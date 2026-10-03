"""
!!!!! deprecated -> use notebooks/simulate.ipynb !!!!!
"""

from os.path import dirname, abspath, join
import sys

# add paths to import from simulator and equities
ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)
sys.path.append(join(ROOT_DIR, 'simulator'))
sys.path.append(join(ROOT_DIR, 'equities/data_processing'))

# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import pandas as pd
import random
from typing import List, Tuple, Union
from copy import deepcopy
from tqdm import tqdm
from glob import glob
# from matplotlib import pyplot as plt
from contextlib import nullcontext
import torch

# for profiling
import pstats

from simulator.core import Message
# from simulator.core.utils import str_to_ns, fmt_ts
from simulator.markets.order_book import OrderBook
from simulator.markets.orders import LimitOrder, Side, MarketOrder
from equities.data_processing import itch_preproc
from equities.data_processing import itch_encoding
from equities.model import GPTConfig, GPT
from equities.data_processing import itch_encoding

# INIT PARAMS
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# out_dir = parent_folder_path + '/out' # ignored if init_from is not 'resume'
out_dir = abspath(join(ROOT_DIR, 'out'))
# dataset = '12302019.NASDAQ_ITCH50_AAPL_message_proc.npy' # dataset to use for initial prompt
num_context_msgs = 100 # 400 # number of messages from dataset to use as context
num_samples = 1 # number of samples to draw (think of like monte carlo paths)
max_new_tokens = 1 # number of tokens generated in each sample (think of like time steps)
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 42
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# exec(open('equities/configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# set up Nasdaq Exchange Agent
TIME = 0
WORLD_AGENT_ID = 1
class FakeExchangeAgent:
    def __init__(self):
        self.messages = []
        self.current_time = TIME
        self.mkt_open = TIME
        self.book_logging = None
        self.stream_history = 10

    def reset(self):
        self.messages = []

    def send_message(self, recipient_id: int, message: Message, _: int = 0):
        self.messages.append((recipient_id, message))

    def logEvent(self, *args, **kwargs):
        pass

# define load paths
raw_itch_load_path = abspath(join(ROOT_DIR, 'dataset/raw/ITCH/'))
processed_dataset = '03272019.NASDAQ_ITCH50_AAPL_message_proc.npy'
proc_data_dir = join('dataset/proc/ITCH/full_view/', processed_dataset)
proc_data_dir = abspath(join(ROOT_DIR, proc_data_dir))
symbols_load_path = abspath(join(ROOT_DIR, 'dataset/symbols'))
symbols_file = sorted(glob(symbols_load_path + '/*sp500*.txt'))[0]

# locate raw ITCH data
itch_message_files = sorted(glob(raw_itch_load_path + '/*message*.csv'))
itch_book_files = sorted(glob(raw_itch_load_path + '/*book*.csv'))
print('found', len(itch_message_files), 'ITCH message files')
print('found', len(itch_book_files), 'ITCH book files')

# create reverse ticker symbol mapping (key is index, value is ticker)
tickers = {}
with open(symbols_file) as f:
    idx = 0
    for line in f:
        idx += 1
        tickers[idx] = line.strip()

# load raw ITCH data (book)
symbols = []
for m_f, b_f in tqdm(zip(itch_message_files, itch_book_files)):
    if '03272019' not in m_f:
        continue
    print(m_f)
    
    # load first message to init simulator
    first_message = (itch_preproc.load_message_df(m_f)).iloc[0]

    # symbol to store in list and use to create OB objects in loop later
    symbol = m_f.rsplit('/', maxsplit=1)[-1][:-12].rsplit('_', maxsplit=1)[-1]
    print("Adding symbol:", symbol)
    symbols.append(symbol)

# load processed ITCH data (messages)
proc_messages = np.array(np.load(proc_data_dir, mmap_mode='r')[0:(15700 + num_context_msgs)])
print("proc_messages.shape:", proc_messages.shape)

# init new book under nasdaq agent
nasdaq_agent = FakeExchangeAgent()

# create a dictionary of order books based on each symbol in symbols
print("Creating order books for symbols:", symbols)
order_books = {}
for symbol in symbols:
    order_books[symbol] = OrderBook(nasdaq_agent, symbol)

# empty book
assert order_books[tickers[proc_messages[0][0]]].bids == order_books[tickers[proc_messages[0][0]]].asks == []

# first message is missing in proc_messages, so we'll use the raw message file to start the book
# print(first_message)

# insert bid order
bid_order = LimitOrder(
    order_id=first_message['id'],
    agent_id=1, # world agent, leave alone for now
    time_placed=first_message['time'],
    symbol=symbols[0],
    quantity=int(first_message['size']),
    side=Side.BID if first_message['side'] == 0 else Side.ASK,
    limit_price=int(first_message['price']*100),
)
order_books[symbols[0]].handle_limit_order(bid_order)

# INIT ORDER BOOKS FROM PROCESSED CONTEXT DATA
# [ "ticker", "order_id", "event_type", "direction", "price_abs", "price",
# "fill_size", "remain_size", "delta_t_s", "delta_t_ns", "time_s", "time_ns",
# "old_id", "price_ref", "fill_size_ref", "time_s_ref", "time_ns_ref", "old_price_abs"]

# init variables to keep track of previous time, price, etc.
prev_time = first_message['time']
prev_price = int(first_message['price']*100)
L1 = [] # list to store L1 data for plotting
last_prices = [] # list to store last prices for plotting

# iterate through messages and update order books
for msg in proc_messages:
    symbol = tickers[msg[0]]
    order_id = msg[1]
    event_type = msg[2]
    price = msg[4]

    # verify time correctness
    assert prev_time + (msg[8]*1000000000) + msg[9] == (msg[10] * 1000000000) + msg[11]
    time = prev_time + (msg[8]*1000000000) + msg[9]

    # handle order based on event type
    if event_type == 1:
        # ADD LIMIT ORDER
        direction = Side.BID if msg[3] == 0 else Side.ASK
        fill_size = msg[6]
        order = LimitOrder(
            order_id=order_id,
            agent_id=1, # world agent, leave alone for now
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction,
            limit_price=price,
        )
        order_books[symbol].handle_limit_order(order)
    elif event_type == 2:
        # EXECUTE ORDER
        fill_size = msg[6]
        direction = Side.BID if msg[3] == 1 else Side.ASK # opposite of direction in non-execution messages
        order = MarketOrder(
            order_id=order_id,
            agent_id=1, # world agent, leave alone for now
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction, # Buy Order if Side.BID (remove liquidity from ask side), Sell Order if Side.ASK (remove liquidity from bid side)
        )
        order_books[symbol].handle_market_order(order)
    elif event_type == 3:
        # EXECUTE ORDER WITH PRICE DIFFERENT THAN DISPLAY
        # This order type is most likely an execution of a price-to-comply order, which is handled by the simulator
        # but this not encoded in the ITCH data beforehand, so we cannot know whether an order is price-to-comply at the time of submission
        # therefore, we handle this event type as a modifed order and then regular execution order (for now, until we revise the data processing)

        # modfify the matched limit order
        direction = Side.BID if msg[3] == 0 else Side.ASK
        ref_order_time = (msg[15] * 1000000000) + msg[16]
        ref_order_size = msg[14]
        ref_order_price = msg[17] # old_price_abs (not mid_price so we cannot calculate using price_ref msg[13])
        # define original order
        original_order = LimitOrder(
            order_id=order_id,
            agent_id=1, # world agent, leave alone for now
            time_placed=ref_order_time,
            symbol=symbol,
            quantity=ref_order_size,
            side=direction,
            limit_price=ref_order_price,
        )
        # define modified order
        # modified_price = msg[4]
        modified_order = LimitOrder(
            order_id=order_id,
            agent_id=1, # world agent, leave alone for now
            time_placed=ref_order_time,
            symbol=symbol,
            quantity=ref_order_size,
            side=direction,
            limit_price=price,
        )
        order_books[symbol].modify_order(original_order, modified_order)
        # execute the modified order
        fill_size = msg[6]
        direction = Side.BID if msg[3] == 1 else Side.ASK # opposite of direction in non-execution messages
        order = MarketOrder(
            order_id=order_id,
            agent_id=1, # world agent, leave alone for now
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction, # Buy Order if Side.BID (remove liquidity from ask side), Sell Order if Side.ASK (remove liquidity from bid side)
        )
        order_books[symbol].handle_market_order(order)
    elif event_type == 4:
        # CANCEL ORDER
        direction = Side.BID if msg[3] == 0 else Side.ASK
        ref_order_time = (msg[15] * 1000000000) + msg[16]
        if msg[7] == 0:
            # FULL DELETION
            fill_size = msg[6]
            order = LimitOrder(
                order_id=order_id,
                agent_id=1, # world agent, leave alone for now
                time_placed=ref_order_time,
                symbol=symbol,
                quantity=fill_size,
                # quantity=msg[14], # total size of order when placed
                side=direction,
                limit_price=price,
            )
            order_books[symbol].cancel_order(order)
        else:
            # PARTIAL CANCELLATION
            cancel_size = msg[6]
            ref_order_size = msg[7] + cancel_size # total size of order before partial cancel
            order = LimitOrder(
                order_id=order_id,
                agent_id=1, # world agent, leave alone for now
                time_placed=ref_order_time,
                symbol=symbol,
                quantity=ref_order_size,
                side=direction,
                limit_price=price,
            )
            order_books[symbol].partial_cancel_order(order, cancel_size)
    elif event_type == 5:
        # REPLACE ORDER
        direction = Side.BID if msg[3] == 0 else Side.ASK
        old_order_id = msg[12]
        old_order_time = (msg[15] * 1000000000) + msg[16]
        old_order_size = msg[14]
        old_order_price = msg[17] # old_price_abs (not mid_price so we cannot calculate using price_ref msg[13])
        # define old order
        old_order = LimitOrder(
            order_id=old_order_id,
            agent_id=1, # world agent, leave alone for now
            time_placed=old_order_time,
            symbol=symbol,
            quantity=old_order_size,
            side=direction,
            limit_price=old_order_price,
        )
        new_order_size = msg[6]
        # define new order
        new_order = LimitOrder(
            order_id=order_id,
            agent_id=1, # world agent, leave alone for now
            time_placed=time,
            symbol=symbol,
            quantity=new_order_size,
            side=direction,
            limit_price=price,
        )
        order_books[symbol].replace_order(1, old_order, new_order) # first arg is agent_id (world agent)
    else:
        raise NotImplementedError("Event type not implemented")

    # update previous time and price
    prev_time = time
    prev_price = price

    # update plotting variables
    L1.append((time, order_books[symbol].get_l1_bid_data(), order_books[symbol].get_l1_ask_data()))
    if event_type in [2, 3]:
        last_prices.append((time, price))

print("len(L1):", len(L1))
print("len(last_prices):", len(last_prices))
    
# INIT MODEL
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()
model.to(device)

# encode the context data
vocab = itch_encoding.Vocab()
# take the last 'num_context_msgs' messages as context
X_raw = proc_messages[-num_context_msgs:]
print("X_raw.shape:", X_raw.shape)
X = itch_encoding.encode_msgs(X_raw, vocab.ENCODING)
print("X.shape:", X.shape)
time = itch_encoding.decode_msg(X[-1], vocab.ENCODING)[10] * 1000000000 + itch_encoding.decode_msg(X[-1], vocab.ENCODING)[11]
print("Simulation start time:", time)
gen_start_time = time # for computing simulation time elapsed in generation
encoded_tok_len = X.shape[1]
# prepare context tensor
x = (torch.tensor(X.reshape(-1), dtype=torch.long, device=device)[None, ...])
print("x.shape:", x.shape)

def find_matching_order(order_book: OrderBook, direction: Side, price: int, fill_size: int, ref_order_time: int) -> LimitOrder:    
    # Error correction: find matching order in order book or re-run timestep if None
    book = order_book.bids if direction.is_bid() else order_book.asks

    for i, price_level in enumerate(book):
        if not price_level.price == price:
            continue

        # print("price match found at level:", i)
        # print("orders at this price level:", price_level.visible_orders)

        # compile candidate orders
        candidate_orders = []
        for order in price_level.visible_orders:
            # print("order:", order[0])
            candidate_orders.append(order[0]) if order[0].agent_id == WORLD_AGENT_ID else None

        # find matching order
        # print("candidate orders:", candidate_orders)
        candidiate_size_only = []
        candidiate_time_only = []
        for order in candidate_orders:
            if order.quantity == fill_size and order.time_placed == ref_order_time:
                # print("matching order (size & time) found:", order)
                return deepcopy(order)
            elif order.quantity == fill_size and order.time_placed != ref_order_time:
                # print("matching order (size only) found:", order)
                candidiate_size_only.append(order)
            elif order.quantity != fill_size and order.time_placed == ref_order_time:
                # print("matching order (time only) found:", order)
                candidiate_time_only.append(order)
            # else:
                # print("no matching order found")

        if len(candidiate_size_only) >= 1:
            # print("return matching order (size only):", candidiate_size_only[0])
            return deepcopy(candidiate_size_only[0])
        elif len(candidiate_time_only) >= 1:
            # print("return matching order (time only):", candidiate_time_only[0])
            return deepcopy(candidiate_time_only[0])
        elif len(candidate_orders) >= 1:
            # print("no matching order found. Return initial volume at price level:", candidate_orders[0])
            return deepcopy(candidate_orders[0])

    # throw error if no match found
    # raise ValueError("No matching order found in order book")
    return None

def process_message(
        symbol: str,
        price: int,
        time: int,
        event_type: int,
        msg: np.ndarray,
        order_books: dict,
        agent_id: int = 1, # world agent by default
# ) -> Union[LimitOrder, MarketOrder, Tuple[LimitOrder, LimitOrder], None]:
):
    """Handles ITCH message processing and order book updates.

    Arguments:
        price (int): Price of order.
        time (int): Time of order.
        event_type (int): Type of event.
        msg (np.ndarray): ITCH message.
        order_book (OrderBook): Order book object.
    """

    # handle message based on event type
    if event_type == 1:
        # ADD LIMIT ORDER
        direction = Side.BID if msg[3] == 0 else Side.ASK
        # print("Direction:", direction)
        fill_size = msg[6]
        # print("Fill Size:", fill_size)
        new_order = LimitOrder(
            agent_id=agent_id,
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction,
            limit_price=price,
        )
        order_books[symbol].handle_limit_order(new_order)
        ref_order = None
    elif event_type == 2:
        # raise NotImplementedError("Event type 2 not implemented")
        # EXECUTE ORDER
        direction = Side.BID if msg[3] == 1 else Side.ASK # opposite of direction in non-execution messages
        # print("Direction:", direction)
        fill_size = msg[6]
        # print("Fill Size:", fill_size)
        new_order = MarketOrder(
            agent_id=agent_id,
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction, # Buy Order if Side.BID (remove liquidity from ask side), Sell Order if Side.ASK (remove liquidity from bid side)
        )
        order_books[symbol].handle_market_order(new_order)
        ref_order = None
    elif event_type == 3:
        # EXECUTE ORDER WITH PRICE DIFFERENT THAN DISPLAY
        raise NotImplementedError("Event type 3 not implemented")

    elif event_type == 4:
        # CANCEL ORDER
        direction = Side.BID if msg[3] == 0 else Side.ASK
        # print("Direction:", direction)
        ref_order_time = (msg[15] * 1000000000) + msg[16]
        if msg[7] == 0:
            # FULL DELETION
            fill_size = msg[6]
            # print("Cancel Size:", fill_size)
            # This is a reference order type, so we must undergo the error correction procedure
            ref_order = find_matching_order(order_books[symbol], direction, price, fill_size, ref_order_time)
            if ref_order is None:
                return None, None
            
            order_books[symbol].cancel_order(ref_order)
            new_order = None
        else:
            # PARTIAL CANCELLATION
            cancel_size = msg[6]
            # print("Partial Cancel Size:", cancel_size)
            ref_order_size = msg[7] + cancel_size # total size of order before partial cancel
            # This is a reference order type, so we must undergo the error correction procedure
            ref_order = find_matching_order(order_books[symbol], direction, price, ref_order_size, ref_order_time)
            if ref_order is None:
                return None, None
            if ref_order.quantity is not ref_order_size:
                # preserve intent of generated partial cancel order
                cancel_ratio = cancel_size / ref_order_size
                cancel_size = int(ref_order.quantity * cancel_ratio)

            order_books[symbol].partial_cancel_order(ref_order, cancel_size)
            # make dummy order to track new cancel_size
            new_order = LimitOrder(
                order_id=-1,
                agent_id=agent_id,
                time_placed=ref_order_time,
                symbol=symbol,
                quantity=cancel_size, # partial cancel size
                side=direction,
                limit_price=0,
            )
    elif event_type == 5:
        # REPLACE ORDER
        # raise NotImplementedError("Event type 5 not implemented")
        direction = Side.BID if msg[3] == 0 else Side.ASK
        # print("Direction:", direction)
        # This is a reference order type, so we must undergo the error correction procedure
        # get reference order details
        bid_price, ask_price = order_books[symbol].bids[0].price, order_books[symbol].asks[0].price
        mid_price = ((bid_price + ask_price) / 2) // 1
        ref_price = int(mid_price) + msg[13]
        ref_size = msg[14]
        ref_order_time = (msg[15] * 1000000000) + msg[16]
        ref_order = find_matching_order(order_books[symbol], direction, ref_price, ref_size, ref_order_time)
        if ref_order is None:
            return None, None

        # define new order
        fill_size = msg[6]
        new_order = LimitOrder(
            agent_id=agent_id,
            time_placed=time,
            symbol=symbol,
            quantity=fill_size,
            side=direction,
            limit_price=price,
        )
        order_books[symbol].replace_order(agent_id, old_order, new_order)
        # return old_order, new_order
    else:
        raise NotImplementedError(f"Event type {event_type} not implemented")
    
    # return order objects for next message encoding step
    return new_order, ref_order

num_generation_steps = 500
num_errors = 0
L1_gen = []
last_prices_gen = []
num_add_order_msgs = 0
num_exec_order_msgs = 0
num_full_cancel_order_msgs = 0
num_partial_cancel_order_msgs = 0
num_replace_order_msgs = 0

for t in tqdm(range(num_generation_steps)):
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens*encoded_tok_len, temperature=temperature, top_k=top_k)


    # decode the generated sequence (will be missing order id, price_abs, old_id, and old_price_abs)
    decoded_msg = itch_encoding.decode_msg(np.array(y[0][-24:].tolist()), vocab.ENCODING)
    # [ "ticker", "NA_VAL", "event_type", "direction", "NA_VAL", "price",
    #  "fill_size", "remain_size", "delta_t_s", "delta_t_ns", "time_s", "time_ns",
    #  "NA_VAL", "price_ref", "fill_size_ref", "time_s_ref", "time_ns_ref", "NA_VAL"]

    # set variables to process new message
    msg = decoded_msg
    symbol = tickers[msg[0]]
    event_type = msg[2]
    # get mid-price of symbol
    bid_price, ask_price = order_books[symbol].bids[0].price, order_books[symbol].asks[0].price
    mid_price = ((bid_price + ask_price) / 2) // 1
    price = int(mid_price) + decoded_msg[5]
    time = prev_time + (msg[8]*1000000000) + msg[9]

    # handle order based on event type
    new_order, ref_order = process_message(symbol, price, time, event_type, msg, order_books, agent_id=WORLD_AGENT_ID)
    if new_order is None and ref_order is None:
        num_errors += 1
        continue

    # update previous time and price
    prev_time = time

    # update plotting variables
    L1_gen.append((time, order_books[symbol].get_l1_bid_data(),
                   order_books[symbol].get_l1_ask_data()))
    if event_type in [2, 3]:
        last_prices_gen.append((time, price))

    # [ "ticker", "order_id", "event_type", "direction", "price_abs", "price",
    #  "fill_size", "remain_size", "delta_t_s", "delta_t_ns", "time_s", "time_ns",
    #  "old_id", "price_ref", "fill_size_ref", "time_s_ref", "time_ns_ref", "old_price_abs"]
    time_s = (prev_time // 1000000000)
    time_ns = (prev_time % 1000000000)

    # set new_msg fields based on event type
    if event_type == 1:
        num_add_order_msgs += 1
        # set new order fields
        order_id = new_order.order_id
        fill_size = msg[6]
        remain_size = msg[7]
        # set conditional fields - these should all be NA_VAL
        old_id = msg[12]
        fill_size_ref = msg[14]
        time_s_ref = msg[15]
        time_ns_ref = msg[16]
        old_price_abs = msg[17]
    elif event_type == 2:
        num_exec_order_msgs += 1
        # set new order fields
        order_id = new_order.order_id
        # TODO: compute price term (msg[5]) from new_order object
        fill_size = msg[6]
        remain_size = msg[7]
        # locate ref_order, then set conditional fields
        ref_order = nasdaq_agent.messages[-2][1].order
        old_id = msg[12]
        # TODO: compute price_ref term (msg[13]) from ref_order object
        # Note: in the training data, the 'fill_size_ref' field is the original order size,
        # whereas here it is the size that was filled by the market order (due to the way
        # the data is logged in agent.messages). We could fix this by tracking the previous
        # L3 state from timestep to timestep, but for now we will leave it as is.
        fill_size_ref = ref_order.quantity
        time_s_ref = (ref_order.time_placed // 1000000000)
        time_ns_ref = (ref_order.time_placed % 1000000000)
        old_price_abs = msg[17]
    elif event_type == 4:
        # set new order fields
        order_id = ref_order.order_id
        # set order size fields based on whether it was a full or partial cancel
        if msg[7] == 0:
            num_full_cancel_order_msgs += 1
            fill_size = ref_order.quantity
            remain_size = msg[7]
        else:
            num_partial_cancel_order_msgs += 1
            fill_size = new_order.quantity # partial cancel size
            remain_size = ref_order.quantity - fill_size
        # set conditional fields
        old_id = msg[12]
        # TODO: set price_ref term (msg[13]), must be == msg[5]
        fill_size_ref = ref_order.quantity
        time_s_ref = (ref_order.time_placed // 1000000000)
        time_ns_ref = (ref_order.time_placed % 1000000000)
        old_price_abs = msg[17]
    elif event_type == 5:
        num_replace_order_msgs += 1
        # set new order fields
        order_id = new_order.order_id
        fill_size = msg[6]
        remain_size = msg[7]
        # set conditional fields
        old_id = ref_order.order_id
        fill_size_ref = ref_order.quantity
        time_s_ref = (ref_order.time_placed // 1000000000)
        time_ns_ref = (ref_order.time_placed % 1000000000)
        old_price_abs = ref_order.limit_price

    new_msg = np.array([msg[0], order_id, msg[2], msg[3], price, msg[5],
            fill_size, remain_size, msg[8], msg[9], time_s, time_ns,
            old_id, msg[13], fill_size_ref, time_s_ref, time_ns_ref, old_price_abs])

    # encode the new message and convert to torch tensor
    x_new = itch_encoding.encode_msg(new_msg, vocab.ENCODING)
    x_new = torch.tensor(x_new, dtype=torch.long, device=device)
    x_new = torch.unsqueeze(x_new, 0) # add batch dimension for concatenation purposes

    # append sampled index to the running sequence and continue
    x = torch.cat((x, x_new), dim=1)

    # if the sequence context is growing too long we must crop it at block_size
    if (x.size(1) + encoded_tok_len) > model.config.block_size:
        x = x[:, encoded_tok_len:]

print("Total number of errors:", num_errors)
print("Error percentage:", round((num_errors / num_generation_steps) * 100, 2), "%")

print("gen_start_time:", gen_start_time)
print("gen_end_time:", time)

sim_time_elapsed = time - gen_start_time
print("Simulation time elapsed (nanoseconds):", sim_time_elapsed)
print("Simulation time elapsed (seconds):", sim_time_elapsed / 1e9)
print("Simulation time elapsed (minutes):", sim_time_elapsed / 1e9 / 60)

print("len(L1_gen):", len(L1_gen))
print("len(last_prices_gen):", len(last_prices_gen))

print('done')