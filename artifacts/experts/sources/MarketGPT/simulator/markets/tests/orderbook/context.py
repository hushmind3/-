# # This file is used to give the individual tests import context
# from typing import List, Tuple
# from os.path import dirname, abspath
# import sys
# SIM_DIR = dirname(dirname(dirname(dirname(abspath(__file__)))))
# sys.path.append(SIM_DIR)

# # test package imports
# from core import Message
# from markets.order_book import OrderBook
# from markets.orders import LimitOrder, MarketOrder, Side, Order
# from markets.messages.orderbook import OrderExecutedMsg
# from markets.messages.orderbook import OrderAcceptedMsg
# from markets.messages.orderbook import OrderModifiedMsg
# from markets.messages.orderbook import OrderReplacedMsg
# from markets.price_level import PriceLevel


# SYMBOL = "X"
# TIME = 0


# class FakeExchangeAgent:
#     def __init__(self):
#         self.messages = []
#         self.current_time = TIME
#         self.mkt_open = TIME
#         self.book_logging = None
#         self.stream_history = 10

#     def reset(self):
#         self.messages = []

#     def send_message(self, recipient_id: int, message: Message, _: int = 0):
#         self.messages.append((recipient_id, message))

#     def logEvent(self, *args, **kwargs):
#         pass


# def setup_book_with_orders(
#     bids: List[Tuple[int, List[int]]] = [], asks: List[Tuple[int, List[int]]] = []
# ) -> Tuple[OrderBook, FakeExchangeAgent, List[LimitOrder]]:
#     agent = FakeExchangeAgent()
#     book = OrderBook(agent, SYMBOL)
#     orders = []

#     for price, quantities in bids:
#         for quantity in quantities:
#             order = LimitOrder(1, TIME, SYMBOL, quantity, Side.BID, price)
#             book.handle_limit_order(order)
#             orders.append(order)

#     for price, quantities in asks:
#         for quantity in quantities:
#             order = LimitOrder(1, TIME, SYMBOL, quantity, Side.ASK, price)
#             book.handle_limit_order(order)
#             orders.append(order)

#     agent.reset()

#     return book, agent, orders

# def reset_env():
#     Order._order_id_counter = 0