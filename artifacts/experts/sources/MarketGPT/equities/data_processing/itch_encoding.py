"""
This code was built on the work of the authors of the paper "Generative AI for End-to-End Limit Order Book Modelling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network"

The original code can be found at: https://github.com/peernagy/LOBS5/blob/main/lob/encoding.py

I refactored the code to work with the ITCH datasets (with my custom fields), multiple tickers, and without the use of JAX.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd


NA_VAL = -9999
SINK_VAL = -20000
MASK_VAL = -10000


def encode(ar, ks, vs):
    """ replace values in ar with values in vs at indices in ks 
        (mapping from vs to ks)
    """
    return vs[np.searchsorted(ks, ar)]

def decode(ar, ks, vs):
    return encode(ar, vs, ks)

def is_special_val(x):
    return np.isin(x, np.array([MASK_VAL, SINK_VAL, NA_VAL])).any()

def expand_special_val(x, n_tokens):
    return np.tile(x, (n_tokens,))

def split_int(x, n_tokens, tok_len, prepend_sign_token=False):

    if prepend_sign_token:
        sign = np.sign(x)
        # only allow pos or neg sign, counting zero as negative (-0)
        sign = np.where(sign == 0, -1, sign)
    
    x = np.abs(x)
    base = 10
    n_digits = n_tokens * tok_len
    div_exp = np.flip(
        np.arange(0, n_digits, tok_len))
    splits = (x // (base ** div_exp)) % (10**tok_len)
    if prepend_sign_token:
        splits = np.hstack([sign, splits])
    return splits

def combine_int(x, tok_len, sign=1):
    base = 10
    n_digits = np.expand_dims(x, axis=0).shape[-1] * tok_len
    exp = np.flip(
        np.arange(0, n_digits, tok_len))
    return sign * np.sum(x * (base ** exp), axis=-1)

def split_field(x, n_tokens, tok_len, prepend_sign_token=False):
    total_tokens = n_tokens + int(prepend_sign_token)
    if is_special_val(x):
        return expand_special_val(x, total_tokens)
    else:
        return split_int(x, n_tokens, tok_len, prepend_sign_token)

def combine_field(
        x: np.array,
        tok_len: int,
        sign: np.array = np.array(1)
    ):
    if is_special_val(np.concatenate((sign.flatten(), x.flatten()))):
        return NA_VAL
    else:
        return combine_int(x, tok_len, sign)

def encode_msg(
        msg: np.array,
        encoding: Dict[str, Tuple[np.array, np.array]],
    ) -> np.array:
    ticker = encode(msg[0], *encoding['ticker'])
    
    # NOTE: leave out id in msg[1]
    event_type = encode(msg[2], *encoding['type'])
    
    direction = encode(msg[3], *encoding['side'])
    # NOTE: leave out price_abs in msg[4]
    price = split_field(msg[5], 1, 3, True)
    # CAVE: temporary fix to switch tokens for + and - sign
    price_sign = encode(price[0], *encoding['sign'])
    price = encode(price[1], *encoding['price'])
    
    fill_size = encode(msg[6], *encoding['size'])
    remain_size = encode(msg[7], *encoding['size'])
    
    time_comb = encode_time(
        time_s = msg[10], 
        time_ns = msg[11],
        encoding = encoding,
        delta_t_s = msg[8],
        delta_t_ns = msg[9],
    )

    # NOTE: leave out oldID in msg[12]
    price_ref = split_field(msg[13], 1, 3, True)
    # CAVE: temporary fix to switch tokens for + and - sign
    price_ref_sign = encode(price_ref[0], *encoding['sign'])
    price_ref = encode(price_ref[1], *encoding['price'])

    fill_size_ref = encode(msg[14], *encoding['size'])

    time_ref_comb = encode_time(
        time_s = msg[15], 
        time_ns = msg[16],
        encoding = encoding
    )

    out = [
        ticker, event_type, direction, price_sign, price, fill_size, remain_size, time_comb,
        price_ref_sign, price_ref, fill_size_ref, time_ref_comb]
    return np.hstack(out)

def encode_msgs(msgs, encoding):
    return np.array([encode_msg(msg, encoding) for msg in msgs])

def encode_time(
        time_s: np.array,
        time_ns: np.array,
        encoding: Dict[str, Tuple[np.array, np.array]],
        delta_t_s: Optional[np.array] = None,
        delta_t_ns: Optional[np.array] = None,
    ) -> np.array:
    # convert seconds after midnight to seconds after exchange open
    # --> 9.5 * 3600 = 34200
    time_s = split_field(time_s, 2, 3, False)
    time_ns = split_field(time_ns, 3, 3, False)
    if delta_t_s is None and delta_t_ns is None:
        time_comb = np.hstack([time_s, time_ns])
    else:
        delta_t_ns = split_field(delta_t_ns, 3, 3, False)
        time_comb = np.hstack([delta_t_s, delta_t_ns, time_s, time_ns])
    time_comb = encode(time_comb, *encoding['time'])
    return time_comb


def decode_msg(msg_enc, encoding):
    # TODO: check if fields with same decoder can be combined into one decode call

    ticker = decode(msg_enc[0], *encoding['ticker'])
    
    event_type = decode(msg_enc[1], *encoding['type'])
    
    direction = decode(msg_enc[2], *encoding['side'])

    price_sign =  decode(msg_enc[3], *encoding['sign'])
    price = decode(msg_enc[4], *encoding['price'])
    price = combine_field(price, 3, price_sign)

    fill_size = decode(msg_enc[5], *encoding['size'])
    remain_size = decode(msg_enc[6], *encoding['size'])

    delta_t_s, delta_t_ns, time_s, time_ns = decode_time(msg_enc[7:16], encoding)

    price_ref_sign = decode(msg_enc[16], *encoding['sign'])
    price_ref = decode(msg_enc[17], *encoding['price'])
    price_ref = combine_field(price_ref, 3, price_ref_sign)

    fill_size_ref = decode(msg_enc[18], *encoding['size'])

    time_s_ref, time_ns_ref = decode_time(msg_enc[19:24], encoding)

    # order ID is not encoded, so it's set to NA
    # same for old_id, price_abs, and old_price_abs
    return np.hstack([ ticker, NA_VAL,
        event_type, direction, NA_VAL, price, fill_size, remain_size, delta_t_s, delta_t_ns, time_s, time_ns,
        NA_VAL, price_ref, fill_size_ref, time_s_ref, time_ns_ref, NA_VAL])

def decode_msgs(msgs, encoding):
    return np.array([decode_msg(msg, encoding) for msg in msgs])

def decode_time(time_toks, encoding):
    if time_toks.shape[0] == 0:
        return NA_VAL, NA_VAL
    time = decode(time_toks, *encoding['time'])
    # delta_t and time given
    if time.shape[0] == 9:
        delta_t_s = time[0]
        delta_t_ns = combine_field(time[1:4], 3)
        time_s = combine_field(time[4:6], 3) #+ 34200
        time_ns = combine_field(time[6:], 3)

        return delta_t_s, delta_t_ns, time_s, time_ns
    # only time given
    elif time.shape[0] == 5:
        # convert time_s to seconds after midnight
        time_s = combine_field(time[:2], 3) #+ 34200
        time_ns = combine_field(time[2:], 3)

        return time_s, time_ns

# TODO: REIMPLEMENT
def repr_raw_msg(msg):
    field_names = ['ticker', 'id', 'type', 'side', 'price_abs', 'price',
                   'fill_size', 'remain_size', 'delta_t_s', 'delta_t_ns', 'time_s', 'time_ns',
                   'oldId', 'price_ref', 'fill_size_ref', 'time_s_ref', 'time_ns_ref', 'old_price_abs']
    out = ''
    for name, val in zip(field_names, msg):
        # TODO: format spacing
        out += name + ":\t" + str(val) + "\n"
    return out

class Vocab:

    MASK_TOK = 0
    SINK_TOK = 1
    NA_TOK = 2

    def __init__(self) -> None:
        self.counter = 3  # 0: MSK, 1: SINK, 2: NAN
        self.ENCODING = {}
        self.DECODING = {}
        self.DECODING_GLOBAL = {}
        self.TOKEN_DELIM_IDX = {}

        self._add_field('time', range(1000), [3,6,9,12])
        self._add_field('type', range(1,6), None)
        self._add_field('size', range(10000), [])
        self._add_field('price', range(1000), [1]) # TODO: consider expanding this for high-priced stocks (e.g., pre-split AMZN)
        self._add_field('sign', [-1, 1], None)
        self._add_field('side', [0, 1], None)
        # self._add_field('ticker', range(1,504), None) # limit to S&P500 for now
        self._add_field('ticker', range(1, 100), None) # limit to 100 custom symbols for now

    def __len__(self):
        return self.counter

    def _add_field(self, name, values, delim_i=None):
        enc = [(MASK_VAL, Vocab.MASK_TOK), (SINK_VAL, Vocab.SINK_TOK), (NA_VAL, Vocab.NA_TOK)]
        enc += [(val, self.counter + i) for i, val in enumerate(values)]
        self.counter += len(enc) - 3  # don't count special tokens
        enc = tuple(zip(*enc))
        self.ENCODING[name] = (
            np.array(enc[0], dtype=np.int32),
            np.array(enc[1], dtype=np.int32))

    def _add_special_tokens(self):
        for field, enc in self.ENCODING.items():
            self.ENCODING[field]['MSK'] = Vocab.MASK_TOK
            self.ENCODING[field]['SINK'] = Vocab.SINK_TOK
            self.ENCODING[field]['NAN'] = Vocab.NA_TOK

            self.DECODING[field][Vocab.MASK_TOK] = 'MSK'
            self.DECODING[field][Vocab.SINK_TOK] = 'SINK'
            self.DECODING[field][Vocab.NA_TOK] = 'NAN'
        self.ENCODING['generic'] = {
            'MSK': Vocab.MASK_TOK,
            'SINK': Vocab.SINK_TOK,
            'NAN': Vocab.NA_TOK,
        }
        self.DECODING_GLOBAL[Vocab.MASK_TOK] = ('generic', 'MSK')
        self.DECODING_GLOBAL[Vocab.SINK_TOK] = ('generic', 'SINK')
        self.DECODING_GLOBAL[Vocab.NA_TOK] = ('generic', 'NAN')

# TODO: REIMPLEMENT
class Message_Tokenizer:

    FIELDS = (
        'ticker',
        'type',
        'side',
        'price',
        'fill_size',
        'remain_size',
        'delta_t_s',
        'delta_t_ns',
        'time_s',
        'time_ns',
        # reference fields:
        'price_ref',
        'fill_size_ref',
        'time_s_ref',
        'time_ns_ref',
    )
    N_NEW_FIELDS = 10
    N_REF_FIELDS = 4
    # note: list comps only work inside function for class variables
    FIELD_I = (lambda fields=FIELDS:{
        f: i for i, f in enumerate(fields)
    })()
    TOK_LENS = np.array((1, 1, 1, 2, 1, 1, 1, 3, 2, 3, 2, 1, 2, 3))
    TOK_DELIM = np.cumsum(TOK_LENS[:-1])
    MSG_LEN = np.sum(TOK_LENS)
    # encoded message length: total length - length of reference fields
    NEW_MSG_LEN = MSG_LEN - \
        (lambda tl=TOK_LENS, fields=FIELDS: np.sum(tl[i] for i, f in enumerate(fields) if f.endswith('_ref')))()
    # fields in correct message order:
    FIELD_ENC_TYPES = {
        'ticker': 'ticker',
        'type': 'type',
        'side': 'side',
        'price': 'price',
        'fill_size': 'size',
        'remain_size': 'size',
        'delta_t_s': 'time',
        'delta_t_ns': 'time',
        'time_s': 'time',
        'time_ns': 'time',
        'price_ref': 'price',
        'fill_size_ref': 'size', 
        'time_s_ref': 'time',
        'time_ns_ref': 'time',
    }

    @staticmethod
    def get_field_from_idx(idx):
        """ Get the field of a given index (or indices) in a message
        """
        if isinstance(idx, int) or idx.ndim == 0:
            idx = np.array([idx])
        if np.any(idx > Message_Tokenizer.MSG_LEN - 1):
            raise ValueError("Index ({}) must be less than {}".format(idx, Message_Tokenizer.MSG_LEN))
        field_i = np.searchsorted(Message_Tokenizer.TOK_DELIM, idx, side='right')
        return [Message_Tokenizer.FIELDS[i] for i in field_i]
    
    @staticmethod
    def _generate_col_idx_by_encoder():
        """ Generates attribute dictionary col_idx_by_encoder
            with encoder type as key and a list of column (field)
            indices as value. This is used to efficiently decode tokenized
            data. 
        """
        col_idx_by_encoder = {}
        counter = 0
        for n_toks, (col, enc_type) in zip(
            Message_Tokenizer.TOK_LENS,
            Message_Tokenizer.FIELD_ENC_TYPES.items()):
            add_vals = list(range(counter, counter + n_toks))
            try:
                col_idx_by_encoder[enc_type].extend(add_vals)
            except KeyError:
                col_idx_by_encoder[enc_type] = add_vals
            counter += n_toks
        return col_idx_by_encoder

    #col_idx_by_encoder = _generate_col_idx_by_encoder.__func__()()

    def __init__(self) -> None:
        self.col_idx_by_encoder = self._generate_col_idx_by_encoder()
        pass

    def validate(self, toks, vocab):
        """ checks if toks is syntactically AND semantically valid message
            returns triple of (is_valid, error location, error message)
        """
        valid_synt, res = self._validate_syntax(toks, vocab)
        if not valid_synt:
            return False, res, 'syntax error'
        valid_semant, err = self._validate_semantics(res)
        if not valid_semant:
            return False, None, err

    def _validate_syntax(self, toks, vocab):
        try:
            decoded = self.decode_to_str(toks, vocab, error_on_invalid=True)
            return True, decoded
        except ValueError as e:
            return False, e.err_i

    def _validate_semantics(self, decoded):
        ''' checks if decoded message string is semantically correct
            return tuple of (is_valid, error in field, error message)
        '''
        pass

    def invalid_toks_per_msg(self, toks, vocab):
        return (self.decode_to_str(toks, vocab) == '').sum(axis=-1)
    
    def invalid_toks_per_seq(self, toks, vocab):
        return self.invalid_toks_per_msg(toks, vocab).sum(axis=-1)

    def preproc(self, m, b, allowed_event_types=['A','E','C','D','R'], is_multi=False):
        # TYPE
        # filter out only allowed event types ...
        m = m.loc[m.type.isin(allowed_event_types)].copy()
        # ... and corresponding book changes
        b = b.loc[m.index]

        # TIME
        # DELTA_T: time since previous order --> 4 tokens of length 3
        m.insert(
            loc=1,
            column='delta_t_ns',
            value=m['time'].diff().fillna(0)
        )
        m.insert(
            loc=1,
            column='delta_t_s',
            value=m.delta_t_ns.astype(int)
        )
        m.delta_t_ns = ((m.delta_t_ns % 1) * 1000000000).astype(int)

        m.insert(0, 'time_s', m.time.astype(int))
        m.rename(columns={'time': 'time_ns'}, inplace=True)
        m.time_ns = ((m.time_ns % 1) * 1000000000).astype(int)
        
        # SIZE
        m.loc[m['size'] > 9999, 'size'] = 9999
        m.loc[m['oldSize'] > 9999, 'oldSize'] = 9999
        m.loc[m['execSize'] > 9999, 'execSize'] = 9999
        m.loc[m['cancSize'] > 9999, 'cancSize'] = 9999
        m['size'] = m['size'].astype(int) # cannot do this to all other size fields bc they may be NaN (handled later)
        # consolidate size columns
        m['remain_size'] = m['size']
        n_cols = len(m.columns)
        for i in range(0, len(m)):
            # set remaining size of add order to NaN
            if m['type'].iloc[i] == 'A':
                m.iloc[i, (n_cols-1)] = m['execSize'].iloc[i] # NaN
            # set fill size of execute and cancel orders to execSize and cancSize respectively
            elif m['type'].iloc[i] == 'E':
                m.iloc[i, 7] = m['execSize'].iloc[i]
            elif m['type'].iloc[i] == 'C':
                m.iloc[i, 7] = m['execSize'].iloc[i]
            elif m['type'].iloc[i] == 'D':
                m.iloc[i, 7] = m['cancSize'].iloc[i]
            # set remaining size of replace order to NaN
            elif m['type'].iloc[i] == 'R':
                m.iloc[i, (n_cols-1)] = m['execSize'].iloc[i] # NaN
        m = m.rename(columns={'size': 'fill_size'})

        # PRICE
        m['price_abs'] = m.price  # keep absolute price for later (simulator)
        m['old_price_abs'] = m.oldPrice  # keep absolute old price for later (simulator)
        # mid-price reference, rounded down to nearest tick_size
        tick_size = 1
        p_ref = (((b.iloc[:, 1] * 100) + (b.iloc[:, 3] * 100)) / 2).shift()#.round(-2).astype(int).shift()
        p_ref = (p_ref // tick_size) * tick_size
        m.price = self._preproc_prices(m.price, p_ref, p_lower_trunc=-999, p_upper_trunc=999)
        m = m.iloc[1:]
        m.price = m.price.fillna(999).astype(int) # handle very first (and final?) message with NaN price

        # # DIRECTION
        # m.direction = ((m.direction + 1) / 2).astype(int)

        # add time elements of original message as feature and process NaNs
        # for all referential order types ('E','C','D','R')
        if not is_multi:
            m = self._add_orig_msg_features(
                m,
                modif_fields=['price', 'fill_size', 'time_s', 'time_ns'])
        else:
            m = self._add_orig_msg_features_multi(
                m,
                b,
                modif_fields=['price', 'fill_size', 'time_s', 'time_ns'])
        
        # change column order
        m = m.rename(columns={'oldId': 'old_id'})
        if not is_multi:
            m = m[['id', 'type', 'side', 'price_abs', 'price',
                'fill_size', 'remain_size', 'delta_t_s', 'delta_t_ns', 'time_s', 'time_ns',
                'old_id', 'price_ref', 'fill_size_ref', 'time_s_ref', 'time_ns_ref', 'old_price_abs']]
        else:
            m = m[['ticker_id', 'id', 'type', 'side', 'price_abs', 'price',
                'fill_size', 'remain_size', 'delta_t_s', 'delta_t_ns', 'time_s', 'time_ns',
                'old_id', 'price_ref', 'fill_size_ref', 'time_s_ref', 'time_ns_ref', 'old_price_abs']]
        
        # convert event type to numeric for encoding step
        m.type = m.type.replace({'A': 1, 'E': 2, 'C': 3, 'D': 4, 'R': 5})

        assert len(m) + 1 == len(b), "length of messages (-1) and book states don't align"

        return m.values

    def _preproc_prices(self, p, p_ref, p_lower_trunc=-10, p_upper_trunc=13):
        """ Takes prices series and reference price (best bid or mid price), 
            encoding prices relative to reference price.
            Returns scaled price series
        """
        # encode prices relative to (previous) reference price
        p = p - p_ref
        log_path = '/home/aaron/Documents/Github/MarketSimT/log/data_price_trunc_metrics.txt' # TODO: automate this
        outfile = open(log_path, 'a')
        # truncate price at deviation of x
        # min tick is 100, hence min 10-level diff is 900
        # <= 1000 covers ~99.54% on bid side, ~99.1% on ask size (GOOG)
        # 1000 limit doesn't work as well for high-priced stocks (e.g., pre-split AMZN)
        pct_changed = 100 * len(p.loc[p > p_upper_trunc]) / len(p)
        print(f"truncating {pct_changed:.4f}% of prices > {p_upper_trunc}")
        outfile.write(f"truncating {pct_changed:.4f}% of prices > {p_upper_trunc}\n")
        p.loc[p > p_upper_trunc] = p_upper_trunc
        pct_changed = 100 * len(p.loc[p < p_lower_trunc]) / len(p)
        print(f"truncating {pct_changed:.4f}% of prices < {p_lower_trunc}")
        outfile.write(f"truncating {pct_changed:.4f}% of prices < {p_lower_trunc}\n")
        p.loc[p < p_lower_trunc] = p_lower_trunc
        outfile.close()
        # scale prices to min ticks size differences
        p /= 1
        return p

    def _add_orig_msg_features(
            self,
            m,
            modif_types={'E','D'},
            modif_types_special={'R'},
            modif_type_C={'C'},
            modif_fields=['price', 'fill_size', 'time_s', 'time_ns'],
            modif_fields_special=['price_abs', 'fill_size', 'time_s', 'time_ns'],
            special_cols = ['oldId', 'remain_size', 'old_price_abs'],
            nan_val=-9999
        ):
        """ Changes representation of order cancellation ('D') / replace ('R')
            / execution ('E') / execution at different price ('C'),
            representing them as the original message and new columns containing
            the order modification details.
            This effectively does the lookup step in past data.
            TODO: lookup missing original message data from previous days' data?
        """

        # make df that converts 'R' values to 'A' values so that we can reference add order component of replace orders
        r_m = m.copy()
        r_m['type'] = r_m['type'].replace('R', 'A')

        # find and merge modif_fields of E, C, D events that match with A (and R) event id
        m_changes = pd.merge(
            m.loc[m.type.isin(modif_types)].reset_index(),
            r_m.loc[r_m.type == 'A', ['id'] + modif_fields],
            how='left', on='id', suffixes=['', '_ref']).set_index('index')
        
        # find modif_fields of R events that match with past A and R event oldIds
        m_changes_special = pd.merge(
            m.loc[m.type.isin(modif_types_special)].reset_index(),
            (r_m.loc[r_m.type == 'A', ['id'] + modif_fields_special]).rename(columns={'id': 'oldId'}),
            how='left', on='oldId', suffixes=['', '_ref']).set_index('index')
        # price_ref = old_mid_price - new_mid_price (different from price_ref above)
        m_changes_special.rename(columns={'price_abs_ref': 'price_ref'}, inplace=True)
        m_changes_special['price_ref'] = m_changes_special['price_ref'] - m_changes_special['price_abs']

        # Repeat the above process for 'C' events
        m_changes_C = pd.merge(
            m.loc[m.type.isin(modif_type_C)].reset_index(),
            r_m.loc[r_m.type == 'A', ['id'] + modif_fields_special],
            how='left', on='id', suffixes=['', '_ref']).set_index('index')
        m_changes_C.rename(columns={'price_abs_ref': 'price_ref'}, inplace=True)
        m_changes_C['price_ref'] = m_changes_C['price_ref'] - m_changes_C['price_abs']

        # add new empty columns for referenced order
        modif_cols = [field + '_ref' for field in modif_fields]
        m[modif_cols] = nan_val

        # fill reference order types with new values
        m.loc[m_changes.index] = m_changes
        m.loc[m_changes_special.index] = m_changes_special
        m.loc[m_changes_C.index] = m_changes_C
        m[modif_cols] = m[modif_cols].fillna(nan_val).astype(int)

        # prepare remaining columns for encoding stage (fill NaNs)
        m[special_cols] = m[special_cols].fillna(nan_val).astype(int)

        return m
    
    def _add_orig_msg_features_multi(
            self,
            m,
            modif_types={'E','D'},
            modif_types_special={'R'},
            modif_type_C={'C'},
            modif_fields=['price', 'fill_size', 'time_s', 'time_ns'],
            modif_fields_special=['price_abs', 'fill_size', 'time_s', 'time_ns'],
            special_cols = ['oldId', 'remain_size', 'old_price_abs'],
            nan_val=-9999
        ):
        """ Changes representation of order cancellation ('D') / replace ('R')
            / execution ('E') / execution at different price ('C'),
            representing them as the original message and new columns containing
            the order modification details.
            This effectively does the lookup step in past data.
        """

        # make df that converts 'R' values to 'A' values so that we can reference add order component of replace orders
        r_m = m.copy()
        r_m['type'] = r_m['type'].replace('R', 'A')

        # find and merge modif_fields of E, D events that match with A (and R) event id
        m_changes = pd.merge(
            m.loc[m.type.isin(modif_types)].reset_index(),
            r_m.loc[r_m.type == 'A', ['id'] + ['ticker_id'] + modif_fields],
            how='left', on=['id', 'ticker_id'], suffixes=['', '_ref']).set_index('index')
        
        # find modif_fields of R events that match with past A and R event oldIds
        m_changes_special = pd.merge(
            m.loc[m.type.isin(modif_types_special)].reset_index(),
            (r_m.loc[r_m.type == 'A', ['id'] + ['ticker_id'] + modif_fields_special]).rename(columns={'id': 'oldId'}),
            how='left', on=['oldId', 'ticker_id'], suffixes=['', '_ref']).set_index('index')
        # price_ref = old_mid_price - new_mid_price (different from price_ref above)
        m_changes_special.rename(columns={'price_abs_ref': 'price_ref'}, inplace=True)
        m_changes_special['price_ref'] = m_changes_special['price_ref'] - m_changes_special['price_abs']

        # Repeat the above process for 'C' events
        m_changes_C = pd.merge(
            m.loc[m.type.isin(modif_type_C)].reset_index(),
            r_m.loc[r_m.type == 'A', ['id'] + ['ticker_id'] + modif_fields_special],
            how='left', on=['id', 'ticker_id'], suffixes=['', '_ref']).set_index('index')
        m_changes_C.rename(columns={'price_abs_ref': 'price_ref'}, inplace=True)
        m_changes_C['price_ref'] = m_changes_C['price_ref'] - m_changes_C['price_abs']

        # add new empty columns for referenced order
        modif_cols = [field + '_ref' for field in modif_fields]
        m[modif_cols] = nan_val

        # fill reference order types with new values
        m.loc[m_changes.index] = m_changes
        m.loc[m_changes_special.index] = m_changes_special
        m.loc[m_changes_C.index] = m_changes_C
        m[modif_cols] = m[modif_cols].fillna(nan_val).astype(int)

        # prepare remaining columns for encoding stage (fill NaNs)
        m[special_cols] = m[special_cols].fillna(nan_val).astype(int)

        return m
    
    def _numeric_str(self, num, pad=2):
        if num == 0:
            return '-00'
        elif num > 0:
            return '+' + str(num).zfill(pad)
        else:
            # minus sign counts as character
            return str(num).zfill(pad + 1)