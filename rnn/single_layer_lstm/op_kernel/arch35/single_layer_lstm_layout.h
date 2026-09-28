/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file single_layer_lstm_layout.h
 * \brief Shared host/device on-chip layout and row partitioning.
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_LAYOUT_H
#define OPS_RNN_SINGLE_LAYER_LSTM_LAYOUT_H

#include "onchip_budget.h"

/* Operator-specific namespaces avoid collisions when headers are included together. */
namespace SingleLayerLstmFwd {

/* Fold PROJ_VEC_NACC interleaved FMA sums into a compensated accumulator every PROJ_VEC_GROUP terms. */
constexpr uint32_t PROJ_VEC_NACC = 4;
/* At most two FMA terms per lane before compensation. Long four-lane groups
 * lose errors inside each partial that the outer Kahan sum cannot recover;
 * cancellation in the recurrent cell makes those errors observable. */
constexpr uint32_t PROJ_VEC_GROUP = 8;

/* Compensation between partials cannot recover rounding inside an L0C partial.
 * Bound long reductions to 32 terms per fresh accumulator; reductions of at most
 * 128 terms retain their original instruction order. Host and device share both
 * limits so the number of cross-core handshakes remains identical. */
constexpr uint32_t CUBE_SUM_K_THRESHOLD = 128U;
constexpr uint32_t CUBE_SUM_K_CHUNK = 32U;
CH_BOTH bool CompensateCubeK(uint32_t k) { return k > CUBE_SUM_K_THRESHOLD; }

/* MIX_AIC_1_2 requires both AIVs to receive their half of each Fixpipe drain. */
constexpr bool SPLIT = true;
constexpr uint32_t GATES = 4;
constexpr uint32_t C0F = SingleLayerLstmCube::C0_BYTES / sizeof(float); // 8 for fp32

/* Elements of fp32 in one vector repeat. Every UB plane below is padded to this granularity, not to
 * C0F, because SingleLayerLstmVec::TanhVec's Compares call is contractually a whole number of repeats and it reads
 * and writes over the rounded length. Padding costs 16 planes x at most 63 floats. */
constexpr uint32_t VEC_REPEAT_ELEMS = 256U / sizeof(float);

/* fp32 planes the recurrence epilogue holds at once. EVERY ONE OF THEM IS [rowsMax, nChunk] -- the
 * vector half is tiled on the same H axis the cube is, so none of them grows with hidden_size:
 *   4  the drain destinations, one per gate
 *   4  those gates once the input projection has been added in
 *   1  c_{t-1} on the way in, c_t on the way out
 *   1  h_t
 *   1  tanh(c_t)
 *   4  scratch for the activations
 * Alongside them: one BIT-per-element mask plane and two narrow staging planes. Change the plane
 * list and change this number -- PickNChunk sizes the chunk from it, and a stale value here would
 * be caught by RecurrenceFits rather than by the hardware, but only after picking a chunk that
 * never fits. */
constexpr uint32_t VEC_PLANES = 15;

/* Headroom PickNChunk leaves for the rounding its own estimate cannot see: each plane is padded to
 * a vector repeat and then to a 32-byte Bump block, which together come to under 4 KB across the
 * whole list. The exact figure is L.ubBytes, and RecurrenceFits checks that; this only has to be
 * large enough that the estimate never over-picks. */
constexpr uint32_t VEC_SLACK = 16U * 1024;

/* The operator's dtype reaches the cube in phase A only. x and w[0:I] go to phase A's cube at the
 * caller's width and the cube accumulates into an fp32 L0C, so widening them on the way in would
 * cost traffic and buy nothing.
 *
 * Phase B is fp32 unconditionally, and that is the point of this variant: Mmad's A and B operands
 * must share a width, so the width of W_hh^T is the width h_{t-1} is fed back at, and keeping both
 * at fp32 keeps the recurrence state exact across all T steps. The price is that a narrow W_hh^T is
 * widened once per launch before the recurrence starts -- see `wraw` / `wcvt` below -- and that
 * phase B's on-chip budget is counted in fp32 bytes at every dtype.
 *
 * `inBytes` is the caller's element width in bytes rather than a type, because op_host builds this
 * same Layout under g++ and cannot name `half` or `bfloat16_t`. */

/* UB the W_hh^T widening may hold. It is a CAP, not an allocation: the staging is read in row
 * chunks and this only decides how many rows a chunk carries. Bigger means fewer chunks and more
 * UB; the staging overlaps the recurrence planes, so a value at or below the recurrence's own
 * footprint costs nothing at all. */
constexpr uint32_t WIDEN_STAGE_CAP = 64U * 1024;

/* Elements one widening chunk carries. The sink is always GM and the chunk is always a flat element
 * range: both slabs -- x as [T, B, I] and the fused weight as [I+H, 4H] -- are written to workspace
 * in the same row-major order they were read, so a chunk is one contiguous copy needing no row
 * alignment. That is what keeps the extents unbounded, since at large H one weight row no longer
 * fits the staging budget. Rounded to 64 elements so both the narrow read and the fp32 write start
 * on a 32-byte block at either width. */
CH_BOTH uint32_t PickWidenElems(uint32_t inBytes)
{
    if (inBytes == sizeof(float)) {
        return 0;
    }
    uint32_t e = WIDEN_STAGE_CAP / (static_cast<uint32_t>(sizeof(float)) + inBytes);
    e = e / 64U * 64U;
    return (e == 0) ? 64U : e;
}

/* Does phase A's whole [m, I] A tile stay in L1 for the chunk, or is one [m, kChunk] block reloaded
 * inside the k loop? Resident is not "the tile fits L1": it shares L1 with the [kChunk, nChunk] B
 * tile and the bias table, so a tile that fills L1 alone leaves no column chunk runnable, which is a
 * refusal rather than a slow schedule. Measured at I=8192 H=8192 B=2, where [16, 8192] fp32 is
 * exactly L1's 512 KB and the operator turned the shape away; streaming A costs one x re-read per
 * (gate, column chunk) and runs it.
 *
 * Host and kernel both call this. The two modes give L1 different totals, and a kernel that
 * re-derived the other would size its Bump against a budget nobody checked. */
CH_BOTH bool ProjAResident(uint32_t m, uint32_t inSize, uint32_t kChunk, uint32_t nChunk)
{
    if (m == 0 || inSize == 0 || kChunk == 0 || nChunk == 0) {
        return false;
    }
    constexpr uint32_t FSZ = static_cast<uint32_t>(sizeof(float));
    const uint32_t mAl = SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK);
    const uint32_t nAl = SingleLayerLstmCube::CeilAlign(nChunk, SingleLayerLstmCube::CUBE_BLOCK);
    const uint32_t kAl = SingleLayerLstmCube::CeilAlign(kChunk, SingleLayerLstmCube::CUBE_BLOCK);
    const uint64_t aWhole = static_cast<uint64_t>(SingleLayerLstmCube::CeilDiv(inSize, C0F)) * mAl * C0F;
    const uint64_t bTile = static_cast<uint64_t>(SingleLayerLstmCube::CeilDiv(nAl, C0F)) * kAl * C0F;
    const uint64_t bias = SingleLayerLstmCube::BtElems(nChunk);
    return (aWhole + bTile + bias) * FSZ <= SingleLayerLstmCube::CAP_L1;
}

/* The recurrence GEMM is tiled on both K and N, and the K tiling must not change the numbers.
 *
 * Per gate the product is h[m, H] x W_hh^T[H, H]. Sending a whole gate to L0B cost H*H*4 bytes, so
 * L0B's 64 KB capped H at 128, and keeping four gates resident in L1 capped it at 176. Tiling both
 * axes removes the cap: L0A holds [m, kChunk], L0B holds [kChunk, nChunk], and the weights stream
 * GM -> L1 -> L0B when they no longer fit.
 *
 * The partial sum never leaves L0C and never narrows: each (gate, n-chunk) owns one L0C tile, the
 * first k-chunk seeds it with MmadPlain and every later one adds into it with MmadAccum, and the
 * drain happens once after the whole K axis. So the K tiling is a scheduling change, not a numerical
 * one.
 *
 * kChunk and nChunk are multiples of CUBE_BLOCK, not of C0F: SplitB's fp32 arm drives
 * LoadDataWithTranspose, whose granule is 16x16 elements whatever the type, so a chunk origin off
 * that granule faults on device -- measured at H=176, where rounding to C0F gave 88 and the launch
 * faulted while 112 and 96 ran clean. The last chunk of each axis may still be short. */

/* Rows of W_hh^T per Mmad. Bounded by L0A holding [m, kChunk], and balanced against nChunk so that
 * one [kChunk, nChunk] tile fills L0B rather than one axis starving the other. */
CH_BOTH uint32_t PickKChunk(uint32_t hid, uint32_t m)
{
    if (hid == 0 || m == 0) {
        return 0;
    }
    const uint32_t mAl = SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK);
    uint32_t k = SingleLayerLstmCube::CAP_L0A / (mAl * static_cast<uint32_t>(sizeof(float)));
    const uint32_t balanced = CompensateCubeK(hid) ? CUBE_SUM_K_CHUNK : CUBE_SUM_K_THRESHOLD;
    if (k > balanced) {
        k = balanced;
    }
    k = k / SingleLayerLstmCube::CUBE_BLOCK * SingleLayerLstmCube::CUBE_BLOCK;
    if (k == 0) {
        return 0;
    }
    return (k > hid) ? hid : k;
}

/* Columns of one gate per Mmad, and per cross-core round, which is why UB bounds it too. L0B holds
 * [kChunk, nChunk] and the four L0C accumulators hold [m, nChunk]; the vector half consumes exactly
 * this chunk per handshake, so its VEC_PLANES planes are [rowsMax, nChunk]. UB is what the old
 * kernel's hidden_size ceiling really was once the cube stopped scaling with H: it held the whole
 * [rowsMax, H] epilogue at once, so B=1024 was capped at H=112 while B=8 reached 2048. `inBytes`
 * enters only through the two narrow staging planes. */
CH_BOTH uint32_t PickNChunk(uint32_t hid, uint32_t m, uint32_t kChunk, uint32_t inBytes)
{
    if (hid == 0 || m == 0 || kChunk == 0 || inBytes == 0) {
        return 0;
    }
    const uint32_t mAl = SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK);
    uint32_t n = SingleLayerLstmCube::CAP_L0C / (GATES * mAl * static_cast<uint32_t>(sizeof(float)));
    const uint32_t byL0B = SingleLayerLstmCube::CAP_L0B /
                           (SingleLayerLstmCube::CeilAlign(kChunk, SingleLayerLstmCube::CUBE_BLOCK) *
                            static_cast<uint32_t>(sizeof(float)));
    if (n > byL0B) {
        n = byL0B;
    }
    const uint32_t rowsMax = SPLIT ? SingleLayerLstmCube::CeilDiv(m, 2) : m;
    const uint32_t perCol = rowsMax * (VEC_PLANES * static_cast<uint32_t>(sizeof(float)) + 1U + 2U * inBytes);
    const uint32_t byUb = (perCol == 0) ? n : ((SingleLayerLstmCube::CAP_UB - VEC_SLACK) / perCol);
    if (n > byUb) {
        n = byUb;
    }
    n = n / SingleLayerLstmCube::CUBE_BLOCK * SingleLayerLstmCube::CUBE_BLOCK;
    if (n == 0) {
        return 0;
    }
    return (n > hid) ? hid : n;
}

/* Rows per cluster: the batch spread over at most `maxBlocks` clusters, as evenly as it divides.
 *
 * The split need not be exact, which is what makes a prime batch runnable. Taking the largest
 * divisor of B landed one enormous block on a single cluster -- B=4097 (17 x 241) gave 4097 rows,
 * B=2049 (3 x 683) gave 683 -- and the per-AIV [rowsMax, nChunk] planes then had no chunk narrow
 * enough to fit, so those shapes were refused. Ceiling division caps rows at ceil(B/16): 257 and 129.
 *
 * The old rule was protecting against a short last block giving SplitA a different srcStride. It
 * does, and that is harmless: the stride comes from the Layout each cluster builds for itself from
 * its own row count, and nothing about one cluster's addressing is read by another. The kernel entry
 * must build the Layout with `min(mBlk, B - blockBase)`, and it does. `maxBlocks` still caps the
 * cluster count, so rows per cluster grows with B past 16*maxBlocks -- that bound belongs to
 * RecurrenceFits. */
CH_BOTH uint32_t PickRowsPerBlock(uint32_t b, uint32_t maxBlocks)
{
    if (b < 2 || maxBlocks == 0) {
        return b;
    }
    /* At least 2 rows per cluster: the Fixpipe M split hands one to each AIV, and a cluster with a
     * single row leaves the second subcore idle for the whole sweep. */
    uint32_t nb = (b / 2 < maxBlocks) ? (b / 2) : maxBlocks;
    if (nb == 0) {
        nb = 1;
    }
    return SingleLayerLstmCube::CeilDiv(b, nb);
}

/* Rows the cluster starting at `blockBase` actually owns -- mBlk everywhere but the last one. */
CH_BOTH uint32_t RowsOfBlock(uint32_t b, uint32_t mBlk, uint32_t blockBase)
{
    const uint32_t left = (b > blockBase) ? (b - blockBase) : 0;
    return (left < mBlk) ? left : mBlk;
}

/* ---------------------------------------------------------------------------------------------
 * Layout. Both cores construct this from the same scalars, which is how they agree on every address
 * without exchanging any.
 * ------------------------------------------------------------------------------------------- */
struct Layout {
    uint32_t b, hid, steps;
    uint32_t mAll;    // FULL batch -- the GM row stride. Distinct from `m` on purpose.
    uint32_t m, k, n; // the per-gate GEMM as ONE CLUSTER sees it: [mBlk, H] x [H, H]
    /* `rowsMax` is the batch rows one AIV owns; `plane` is ONE UB tile, [rowsMax, nChunk] padded to
     * a vector repeat. IT IS A CHUNK, NOT A ROW OF THE WHOLE PROBLEM -- the vector half walks the
     * same H chunks the cube does, one cross-core round each, so nothing here scales with
     * hidden_size. `plane` is assigned in the body because it depends on nChunk. */
    uint32_t rowsMax, plane;
    /* The recurrence GEMM's tiling: `kChunk` rows and `nChunk` columns of one gate meet in L0 per
     * Mmad, the last of each axis possibly short. Six sizes, kept apart because they were one number
     * back when a whole gate went to L0B, which is what capped H at 128.
     *   aElems      the whole [m, H] A tile; the residency test is written against it.
     *   aChunkElems one [m, kChunk] chunk of it.
     *   aTileElems  what A costs in L1 and L0A -- one of the two above per `aResident`, one number
     *               for both buffers on purpose.
     *   bElems      the L1 region one gate occupies: the whole [H, H] when resident, one
     *               [kChunk, nChunk] staging tile when streamed.
     *   bChunkElems the L0B tile, [kChunk, nChunk], in both modes.
     *   cElems      one L0C accumulator, [m, nChunk]; there are GATES of them, live across the whole
     *               K axis. */
    uint32_t kChunk, kChunks;
    uint32_t nChunk, nChunks;
    uint32_t aElems, aTileElems, aChunkElems, bElems, bChunkElems, cElems;
    /* Residency is an optimisation, three times over, and each degrades into a stream rather than a
     * refusal -- which is why hidden_size has no ceiling.
     *   wResident   all four gates of W_hh^T in L1 for the whole sweep. Turns T weight reads into
     *               one; needs 4*H*H*4 bytes, so it stops near H=176.
     *   aResident   h_{t-1} whole in L1 as [m, H] and split into L0A in one go per timestep; needs
     *               CeilAlign(m,16)*H*4 in each. Otherwise the cube reads [m, kChunk] out of hAll in
     *               workspace, where the AIVs write h_t anyway.
     *
     * The L1 and L0A halves of `aResident` are one flag for correctness, not brevity. Splitting them
     * gives a third mode -- A in L1 but not in L0A -- where SplitA re-reads the L1 tile once per
     * column chunk, and by the second chunk the AIVs have written that timestep's h_t over the
     * leading columns the cube still reads h_{t-1} from. Measured before they were joined: at
     * B=1024 H=512 fp32 the first 96 output columns were correct and 96..511 wrong, reproducing with
     * x set to zero, which places it in the recurrence operand. Holding A whole in L0A makes the L1
     * copy safe because the cube reads all of it at chunk 0, before it first signals the AIVs. The
     * streaming path is safe for a different reason: it reads hAll slot s while the AIVs write
     * slot s+1. */
    bool aResident, wResident;
    uint32_t bOff[GATES], cOff[GATES], l1Bytes;
    uint32_t cub[GATES];
    uint32_t gi, gf, gg, go, cc, hh, co, t1, t2, t3, t4, msk, ubBytes;
    /* Narrow staging for the outputs, held at the CALLER's width. Every output is computed in fp32
     * and narrowed on the way to GM, so these two carry a Cast result and nothing else. At
     * inBytes == 4 they are never touched. `nzero` is a separate plane rather than a re-Cast of
     * `nbuf` because the zero tail is written on the SAME timesteps as live data on other rows. */
    uint32_t inBytes, nbuf, nzero;
    /* Widening staging: `wraw` takes a narrow chunk from GM, `wcvt` holds its fp32 image. Both are
     * zero-sized at fp32, where there is nothing to widen. One pair serves BOTH slabs -- x and the
     * fused weight -- because the widening is a single flat pass done once per launch, before
     * either phase starts. */
    uint32_t wStageElems, wraw, wcvt;
    /* Element count of the uint8 mask plane, SEPARATE from `plane`. A LocalTensor's byte length must
     * be a whole number of 32-byte blocks, and a uint8 tensor of `plane` elements is `plane` BYTES --
     * so at plane = 8 (hidden_size 8 with one row per AIV) it is 8 bytes and the tensor cannot be
     * constructed at all. The fp32 planes are safe for free: hidden_size is a multiple of 8, so
     * plane * 4 is always a multiple of 32. Only the byte-wide one needs rounding up. */
    uint32_t mskElems;

    /* `mBlk` is the rows ONE CLUSTER owns, and every on-chip extent below is built from it, not
     * from the batch. A blockDim of 1 was measured at 666 us against the 268 us of per-step
     * launches it replaced, because it used 1 of 36 clusters. Removing launches does not pay if it
     * also removes the machine.
     * `mBlk` here is the rows THIS cluster owns, which is PickRowsPerBlock's value everywhere but
     * possibly the last cluster -- see RowsOfBlock. Every extent below is built from it, so a short
     * last block is self-consistent rather than a special case. */
    CH_BOTH Layout(uint32_t bIn, uint32_t hIn, uint32_t tIn, uint32_t mBlk,
                   uint32_t inBytesIn = static_cast<uint32_t>(sizeof(float)))
        : b(bIn),
          hid(hIn),
          steps(tIn),
          mAll(bIn),
          m(mBlk),
          k(hIn),
          n(hIn),
          rowsMax(SPLIT ? SingleLayerLstmCube::CeilDiv(mBlk, 2) : mBlk),
          plane(0)
    {
        const uint32_t mAligned = SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK);
        aElems = SingleLayerLstmCube::CeilDiv(hid, C0F) * mAligned * C0F; // [m, H]
        const uint32_t bFullElems = SingleLayerLstmCube::CeilDiv(n, C0F) *
                                    SingleLayerLstmCube::CeilAlign(hid, SingleLayerLstmCube::CUBE_BLOCK) *
                                    C0F; // [H, H]
        inBytes = inBytesIn;

        /* Residency is decided by measuring the candidate layout with the same allocator that would
         * build it, not by re-deriving its byte count: the two differ by Bump's 32-byte rounding, and
         * a flag off by one block puts a tile past the end of L1 with no error message. The cheap
         * question comes first and doubles as the uint32 guard -- one gate alone has to fit L1, which
         * bounds H at 362, so the Bump below only ever counts numbers that cannot wrap (four gates
         * are 16*H*H bytes, passing 4 GB near H=32768). */
        wResident = (static_cast<uint64_t>(bFullElems) * sizeof(float) <= SingleLayerLstmCube::CAP_L1);
        if (wResident) {
            SingleLayerLstmCube::Bump probe;
            probe.TakeT<float>(aElems);
            for (uint32_t g = 0; g < GATES; ++g) {
                probe.TakeT<float>(bFullElems);
            }
            wResident = (probe.cur <= SingleLayerLstmCube::CAP_L1);
        }

        kChunk = PickKChunk(hid, m);
        nChunk = PickNChunk(hid, m, kChunk, inBytes);

        /* The streaming layout's two tenants compete for L1 and the order of the decisions matters.
         * The four [kChunk, nChunk] weight pads can be any size; the A tile is [m, H] and the whole K
         * loop reads column ranges of it, so it is all or nothing. A's residency is settled first
         * against pads at one fractal each, and the pads then take what is left. The other order --
         * shrink pads until they fit, give A up at the minimum -- leaves the pads at that minimum
         * after A has vacated: measured at B=1024 H=2048 it produced kChunk=nChunk=16 with L1 at 8 KB
         * of 512. Once A streams, nothing in L1 scales with hidden_size. */
        /* L0A first: A cannot stay in L1 unless it also fits L0A whole -- see `aResident` above. */
        const bool aFitsL0A = (static_cast<uint64_t>(aElems) * sizeof(float) <= SingleLayerLstmCube::CAP_L0A);
        if (!wResident && kChunk != 0 && nChunk != 0) {
            SingleLayerLstmCube::Bump minPads;
            for (uint32_t g = 0; g < GATES; ++g) {
                minPads.TakeT<float>(SingleLayerLstmCube::CUBE_BLOCK * SingleLayerLstmCube::CUBE_BLOCK);
            }
            /* uint64 for the same reason: the A tile is CeilAlign(m,16)*H*4 bytes and a large
             * enough product would wrap into a value that looks like it fits. */
            const uint64_t aBytes = static_cast<uint64_t>(aElems) * sizeof(float);
            aResident = aFitsL0A && (aBytes + minPads.cur <= SingleLayerLstmCube::CAP_L1);

            for (;;) {
                const uint32_t kAl = SingleLayerLstmCube::CeilAlign(kChunk, SingleLayerLstmCube::CUBE_BLOCK);
                SingleLayerLstmCube::Bump l1try;
                l1try.TakeT<float>(aResident ? aElems : SingleLayerLstmCube::CeilDiv(kChunk, C0F) * mAligned * C0F);
                for (uint32_t g = 0; g < GATES; ++g) {
                    l1try.TakeT<float>(SingleLayerLstmCube::CeilDiv(nChunk, C0F) * kAl * C0F);
                }
                if (l1try.cur <= SingleLayerLstmCube::CAP_L1) {
                    break;
                }
                if (kChunk > nChunk && kChunk > SingleLayerLstmCube::CUBE_BLOCK) {
                    kChunk = kChunk / 2 / SingleLayerLstmCube::CUBE_BLOCK * SingleLayerLstmCube::CUBE_BLOCK;
                } else if (nChunk > SingleLayerLstmCube::CUBE_BLOCK) {
                    nChunk = nChunk / 2 / SingleLayerLstmCube::CUBE_BLOCK * SingleLayerLstmCube::CUBE_BLOCK;
                } else if (kChunk > SingleLayerLstmCube::CUBE_BLOCK) {
                    kChunk = kChunk / 2 / SingleLayerLstmCube::CUBE_BLOCK * SingleLayerLstmCube::CUBE_BLOCK;
                } else {
                    break; // one fractal each and still over: RecurrenceFits refuses the shape
                }
            }
        } else {
            /* Resident weights were probed with the whole A tile beside them, so L1 is settled;
             * L0A still has to take it, or the cube would re-read L1 per chunk. */
            aResident = aFitsL0A;
        }

        /* The ALLOCATION size of one UB plane, padded to a vector repeat. The logical width the
         * kernel computes over is `stripe.count * nCur`, at most `rowsMax * nChunk` -- `plane` is
         * only ever a size, never a stride, so padding it is free of addressing consequences. */
        plane = SingleLayerLstmCube::CeilAlign(rowsMax * nChunk, VEC_REPEAT_ELEMS);
        kChunks = (kChunk == 0) ? 0 : SingleLayerLstmCube::CeilDiv(k, kChunk);
        nChunks = (nChunk == 0) ? 0 : SingleLayerLstmCube::CeilDiv(n, nChunk);
        aChunkElems = SingleLayerLstmCube::CeilDiv(kChunk, C0F) * mAligned * C0F; // [m, kChunk]
        bChunkElems = SingleLayerLstmCube::CeilDiv(nChunk, C0F) *
                      SingleLayerLstmCube::CeilAlign(kChunk, SingleLayerLstmCube::CUBE_BLOCK) * C0F; // [kChunk, nChunk]
        cElems = mAligned * SingleLayerLstmCube::CeilAlign(nChunk, SingleLayerLstmCube::CUBE_BLOCK);

        bElems = wResident ? bFullElems : bChunkElems;
        /* ONE SIZE FOR BOTH BUFFERS. Resident: the whole [m, H] tile, in L1 and in L0A. Streaming:
         * a [m, kChunk] landing pad in L1 and the same extent in L0A. */
        aTileElems = aResident ? aElems : aChunkElems;

        SingleLayerLstmCube::Bump l1;
        /* h_{t-1} at offset 0: the whole [m, H] tile when it is resident, one [m, kChunk] landing
         * pad refilled from hAll otherwise. */
        l1.TakeT<float>(aTileElems);
        for (uint32_t g = 0; g < GATES; ++g) {
            /* Resident: W_hh^T gate by gate, loaded once for the sweep. Streaming: the landing pad
             * for gate g's CURRENT [kChunk, nChunk] tile, refilled from GM every Mmad. Four pads
             * rather than one so the four loads of a k-chunk can be issued together. */
            bOff[g] = l1.TakeT<float>(bElems);
        }
        l1Bytes = l1.cur;

        /* Four L0C tiles rather than one [m, 4H]: writing gate g at a column offset inside a single
         * NZ accumulator would depend on L0C's internal blocking, and four separate tiles have the
         * SAME shape as the call lstm_grad already uses. */
        SingleLayerLstmCube::Bump l0c;
        for (uint32_t g = 0; g < GATES; ++g) {
            cOff[g] = l0c.TakeT<float>(cElems);
        }

        SingleLayerLstmCube::Bump ub;
        for (uint32_t g = 0; g < GATES; ++g) {
            cub[g] = ub.TakeT<float>(plane); // drain destinations, one per gate
        }
        gi = ub.TakeT<float>(plane);
        gf = ub.TakeT<float>(plane);
        gg = ub.TakeT<float>(plane);
        go = ub.TakeT<float>(plane);
        cc = ub.TakeT<float>(plane); // carried across the whole sweep
        hh = ub.TakeT<float>(plane); // ditto -- neither ever leaves UB
        /* tanh(c_t). A REAL plane rather than an alias of the dead `t4`: the bump allocator exists
         * to make silent aliasing unrepresentable, and 1 plane of 15 is not worth giving that up. */
        co = ub.TakeT<float>(plane);
        t1 = ub.TakeT<float>(plane);
        t2 = ub.TakeT<float>(plane);
        t3 = ub.TakeT<float>(plane);
        t4 = ub.TakeT<float>(plane);
        /* `Compares` writes one BIT per element, i.e. ceil(plane/8) bytes, so rounding the element
         * count up to a 32-byte block is pure headroom rather than a change of meaning. */
        mskElems = SingleLayerLstmCube::CeilAlign(plane, SingleLayerLstmCube::C0_BYTES);
        msk = ub.TakeT<uint8_t>(mskElems);
        nbuf = ub.TakeT<uint8_t>(plane * inBytes);
        nzero = ub.TakeT<uint8_t>(plane * inBytes);
        const uint32_t recurBytes = ub.cur;

        /* The widening staging starts at UB offset 0 and overlaps the recurrence planes on purpose: it
         * is finished before hh / cc are seeded and never read again, so the budget is the max of the
         * two rather than their sum. Written as a second bump rather than offsets into the first, so
         * that a plane which would be live at the same time cannot be taken from it by accident. */
        wStageElems = PickWidenElems(inBytes);
        SingleLayerLstmCube::Bump st;
        wraw = st.TakeT<uint8_t>(wStageElems * inBytes);
        wcvt = st.TakeT<float>(wStageElems);
        const uint32_t stageBytes = (wStageElems == 0) ? 0 : st.cur;

        ubBytes = (recurBytes > stageBytes) ? recurBytes : stageBytes;
    }
};

} // namespace SingleLayerLstmFwd

#endif // OPS_RNN_SINGLE_LAYER_LSTM_LAYOUT_H
