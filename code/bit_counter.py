import numpy as np
import os
import tempfile
from fjcommon import functools_ext as ft
from fjcommon import timer
from fjcommon import no_op
import itertools

import arithmetic_coding as ac
import probclass


def encode_decode_to_file_ctx(syms, prediction_net: probclass.PredictionNetwork, syms_format='HWC', verbose=False):
    """
    Encode symbols with arithmetic coding to disk.
    :param syms: HWC or CHW depending on syms_format, symbols of one image. Or BHWC, BCHW, in which case the number
    of bits needed for all batches is returned.
    :param prediction_net:
    arithmetic coding to be correct).
    :return: number of bits to encode all symbols in `syms`
    """
    _print = print if verbose else no_op.NoOp()

    if len(syms.shape) == 4:
        num_batches = syms.shape[0]
        return np.sum([encode_decode_to_file_ctx(syms[b, ...], prediction_net, syms_format, verbose)
                       for b in range(num_batches)])

    assert len(syms.shape) == 3, 'Expected HWC or CHW'
    assert syms_format in ('HWC', 'CHW')
    if syms_format == 'HWC':
        _print('Transposing symbols for encoding...')
        syms = np.transpose(syms, (2, 0, 1))

    # ---
    _print('Preparing encode...')

    foutid, fout_p = tempfile.mkstemp()
    ctx_shape = prediction_net.input_ctx_shape
    get_freqs = ft.compose(ac.SimpleFrequencyTable, prediction_net.get_freqs)
    get_pr = prediction_net.get_pr

    # encode
    with timer.execute('Encoding time [s]'):
        _print('Encoding symbols of shape {} ({} symbols) with context shape {}...'.format(
                syms.shape, np.prod(syms.shape), ctx_shape))

        syms_padded = prediction_net.pad_symbols_volume(syms)
        virtual_num_bits, first_sym, theoretical_bit_cost = _encode(
            foutid, syms_padded, ctx_shape, get_freqs, get_pr, _print)
        assert abs(virtual_num_bits - theoretical_bit_cost) < 50, 'Virtual: {} -- Theoretical: {}'.format(
            virtual_num_bits, theoretical_bit_cost)

    # bit count
    actual_num_bits = os.path.getsize(fout_p) * 8
    assert actual_num_bits == virtual_num_bits, '{} != {}'.format(
        actual_num_bits, virtual_num_bits)

    # decode
    with timer.execute('Decoding time [s]'):
        _print('Decoding symbols to shape {}, first_sym={}...'.format(
                syms_padded.shape, first_sym))

        syms_dec_padded = _decode(fout_p, syms_padded.shape, ctx_shape, first_sym, get_freqs, _print)
        syms_dec = prediction_net.undo_pad_symbols_volume(syms_dec_padded)

    # checkin' (takes no time)
    np.testing.assert_array_equal(syms, syms_dec)
    _print('Decoded symbols match input!')

    # cleanup
    os.remove(fout_p)

    return actual_num_bits


def _new_ctx_itr(syms, ctx_shape):
    return probclass.iter_over_blocks(syms, ctx_shape)


def _get_num_ctxs(syms_shape, ctx_shape):
    return probclass.num_blocks(syms_shape, ctx_shape)


def _new_ctx_sym_itr(syms, ctx_shape):
    assert len(ctx_shape) == 3
    _, h, w = ctx_shape
    for ctx in _new_ctx_itr(syms, ctx_shape):
        # symbol is in the last depth dimension in the center
        sym = ctx[-1, h // 2, w // 2]
        yield ctx, sym


def _new_sym_idxs_itr(syms_shape, ctx_size):
    D, H, W = syms_shape
    pad = ctx_size // 2
    return itertools.product(
            range(pad, D),  # D dimension is not padded
            range(pad, H - pad),
            range(pad, W - pad))  # yields tuples (d, h, w)


def _encode(foutid, syms, ctx_shape, get_freqs, get_pr, printer):
    """
    :param foutid:
    :param syms: CHW, padded
    :param ctx_shape:
    :param get_freqs:
    :param get_pr:
    :return:
    """
    with open(foutid, 'wb') as fout:
        bit_out = ac.CountingBitOutputStream(
            bit_out=ac.BitOutputStream(fout))
        enc = ac.ArithmeticEncoder(bit_out)
        ctx_sym_itr = _new_ctx_sym_itr(syms, ctx_shape=ctx_shape)
        # First sym is stored separately using log2(L) bits or sth
        first_ctx, first_sym = next(ctx_sym_itr)
        first_pr = get_pr(first_ctx)
        first_bc = -np.log2(first_pr[first_sym])
        theoretical_bit_cost = first_bc
        num_ctxs = _get_num_ctxs(syms.shape, ctx_shape)
        # Encode other symbols
        for i, (ctx, sym) in enumerate(ctx_sym_itr):
            freqs = get_freqs(ctx)
            pr = get_pr(ctx)
            theoretical_bit_cost += -np.log2(pr[sym])
            enc.write(freqs, sym)
            if i % 1000 == 0:
                printer('\rFeeding context for symbol #{}/{}...'.format(i, num_ctxs), end='', flush=True)
        printer('\r\033[K', end='')  # clear line
        enc.finish()
        bit_out.close()
        return bit_out.num_bits, first_sym, theoretical_bit_cost


def _decode(fout_p, symbols_shape_padded, ctx_shape, first_sym, get_freqs, printer):
    # Idea:
    # have a matrix symbols_decoded, initially all zeros.
    # put first_sym into symbols_decoded
    # use a normal ctx_itr to retrieve the current context from symbols_decoded
    # use symbol_idx_itr to get the index of the next decoded symbol
    # write the decoded symbol into symbols_decoded, then advancethe ctx_itr to get the next context
    with open(fout_p, 'rb') as fin:
        bitin = ac.BitInputStream(fin)
        dec = ac.ArithmeticDecoder(bitin)

        symbols_decoded = np.zeros(symbols_shape_padded, dtype=np.int32)
        ctx_itr = _new_ctx_itr(symbols_decoded, ctx_shape)
        ctx_size = probclass.context_size_from_context_shape(ctx_shape)
        sym_idxs_itr = _new_sym_idxs_itr(symbols_shape_padded, ctx_size=ctx_size)

        next(ctx_itr)  # skip first ctx
        symbols_decoded[next(sym_idxs_itr)] = first_sym  # write first_sym
        num_ctxs = _get_num_ctxs(symbols_shape_padded, ctx_shape)
        for i, (current_ctx, next_decoded_sym_idx) in enumerate(zip(ctx_itr, sym_idxs_itr)):
            freqs = get_freqs(current_ctx)
            symbol = dec.read(freqs)
            symbols_decoded[next_decoded_sym_idx] = symbol
            if i % 1000 == 0:
                printer('\rFeeding context for symbol #{}/{}...'.format(i, num_ctxs), end='', flush=True)
        printer('\r\033[K', end='')  # clear line
        return symbols_decoded

