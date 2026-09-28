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
 * \file single_layer_lstm_budget.h
 * \brief Host-side on-chip budget for SingleLayerLstm: phase A's time chunk and K block, and the
 *        feasibility test for phase B. Its own header so tiling and the host unit test call the same
 *        function -- a chunk one step too large overflows L0A/L0B/L0C on chip with no error anywhere.
 *        Plain C++ (no AscendC), so it builds under g++ with op_host.
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_BUDGET_H
#define OPS_RNN_SINGLE_LAYER_LSTM_BUDGET_H

#include <cstdlib>
#include "../../op_kernel/arch35/single_layer_lstm_layout.h"

namespace SingleLayerLstmBudget {

constexpr uint32_t GATES = 4;

/* Upper bound on clusters. ascend950 has 36 AIC, so this leaves parallelism on the table on
 * purpose: a 32-cluster MIX configuration has been observed to hang intermittently on real
 * hardware, reproduced with two unrelated kernels, and the cause is not diagnosed. A hung device
 * needs a reset, so the bound stays where the observed configurations are stable. Raise it only
 * together with a diagnosis of that hang -- not because the part has more cores. */
constexpr uint32_t MAX_CLUSTERS = 16;

/* Phase A computes `igates = x @ w[0:I] + b` for the rows this cluster will recur over. Per pass:
 * A = [tChunk*mBlk, kChunk], B = w[k0:k0+kChunk, g*H:(g+1)*H], accumulating over k into one L0C
 * tile [tChunk*mBlk, H], one gate at a time.
 *
 * Chunked over time, never over batch rows: a cluster must hold the projection for exactly the rows
 * it recurs over, and the single-launch design has no grid barrier that would make reading another
 * cluster's bytes safe. K must be blocked as well -- B is [I, H], so at I=256 H=128 it is 128 KB
 * against L0B's 64 KB, and I is an input dimension. Phases A and B never hold L1 at the same time,
 * so the budget is their max, not their sum. One gate at a time, because phase A's M is tChunk*mBlk
 * and four L0C tiles of that size exceed 256 KB.
 *
 * Phase A widens x and w[0:I] into workspace and computes at fp32 like phase B, so `inBytes` reaches
 * no extent here and both extents keep a multiple-of-8 rule at every dtype; it stays a parameter
 * because the workspace sizing needs it. The kernel counterpart is SingleLayerLstmProj::Layout,
 * which cannot be included here -- every extent below is that struct's and the two must be read
 * together.
 *
 * Sets *tChunk, *kChunk and *nChunk and returns true, or returns false, in which case the caller
 * must refuse. Clamping to something that does not fit is the failure this file exists to avoid. */

/* Phase A on the vector units: is this shape one of them, and how many w rows fit a staging burst.
 *
 * Taken for accuracy, not speed. The cube accumulates K in one fp32 chain whose order is fixed in
 * hardware; the vector path accumulates in groups and compensates across them. Measured on seven
 * float32 backward shapes over three input draws (21 runs): cube fails 16, vector fails 6, and the
 * worst-element relative error on T12_B28_I254_H76 falls from 4.19e-04 to 1.11e-04 against a float64
 * reference, where an exact projection would reach 4.96e-05.
 *
 * It costs time. Measured over thirteen forward shapes, the predictor is steps * batch rather than
 * problem size: at or below 8 the cube's M is mostly padding and the vector path runs at 0.81x to
 * 0.98x, above it the cube amortises and the vector path costs up to 2.05x. There is no band where
 * the accuracy is free -- the backward shapes that need it sit at steps * batch between 228 and 486.
 *
 * Two guards refuse a shape on top of the UB plan below. PROJ_VEC_MAX_WORK caps
 * steps * batch * input_size * 4 * hidden_size, because the instruction count is linear in that
 * product and a shape whose UB plan fits could still run arbitrarily long; the largest shape
 * measured sits at 3.13e7. PROJ_VEC_MAX_W_BYTES caps blocks * input_size * cw * 4, the weight
 * traffic: this path reads all of w once per batch block, measured at T=4 B=16 I=705 H=982 with
 * rowsPerBlock 1, where the projection went from 2.792 ms to 3.504 ms. */
constexpr uint64_t PROJ_VEC_MAX_WORK = 64000000ULL;
constexpr uint64_t PROJ_VEC_MAX_W_BYTES = 32ULL * 1024 * 1024;
constexpr uint32_t PROJ_VEC_MIN_KTILE = 8;
constexpr uint32_t PROJ_VEC_MAX_KTILE = 256;

inline bool PickVecProj(uint32_t mBlk, uint32_t batch, uint32_t inSize, uint32_t hid, uint32_t steps, uint32_t* kTile)
{
    constexpr uint32_t FSZ = static_cast<uint32_t>(sizeof(float));
    *kTile = 0;
    if (mBlk == 0 || batch == 0 || inSize == 0 || hid == 0 || steps == 0) {
        return false;
    }
    /* TEST SCAFFOLDING, INERT WHEN UNSET. LSTM_FWD_VECPROJ=2 skips the two size guards so a shape
     * this function would otherwise refuse can be run on the vector path anyway; the UB plan below
     * still has to fit. LSTM_FWD_VECPROJ=0 forces phase A back onto the cube for a shape this
     * function would otherwise take, so the two paths can be compared on one binary and one input
     * draw. No production path sets either. */
    bool force = false;
    if (const char* env = std::getenv("LSTM_FWD_VECPROJ")) {
        if (env[0] == '0') {
            return false;
        }
        if (env[0] == '2') {
            force = true;
        }
    }
    const uint32_t m = steps * mBlk;
    const uint32_t gw = GATES * hid;
    const uint32_t cw = SingleLayerLstmCube::CeilAlign(SingleLayerLstmCube::CeilDiv(gw, 2U), 8U);
    if (!force) {
        const uint64_t work = static_cast<uint64_t>(steps) * batch * inSize * gw;
        if (work > PROJ_VEC_MAX_WORK) {
            return false;
        }
        const uint64_t blocks = SingleLayerLstmCube::CeilDiv(batch, mBlk);
        const uint64_t wBytes = blocks * inSize * cw * FSZ;
        if (wBytes > PROJ_VEC_MAX_W_BYTES) {
            return false;
        }
    }
    /* The regions VectorProjectDirect allocates, in the order it allocates them; each is rounded to
     * the same 32 bytes Bump rounds to, so this total is the one the kernel will ask for. */
    auto region = [](uint32_t elems) {
        return SingleLayerLstmCube::CeilAlign(elems * FSZ, SingleLayerLstmCube::C0_BYTES);
    };
    /* acc and cmp are [m, cw]; the group accumulator is PROJ_VEC_NACC wide; yv, tmp and the guard's
     * zero plane are one row each; the guard's abs plane and mask are rounded to a whole number of
     * compare repeats. The list must track SingleLayerLstmProj::VecLayout -- the kernel asks for
     * what that struct computes, and a host that under-counts hands it a plan that does not fit. */
    const uint32_t cmpCw = SingleLayerLstmCube::CeilAlign(cw, SingleLayerLstmCube::CMP_REPEAT_ELEMS);
    const uint32_t fixed = region(m * cw) * 2 + region(SingleLayerLstmFwd::PROJ_VEC_NACC * cw) + region(cw) * 3 +
                           region(cmpCw) +
                           SingleLayerLstmCube::CeilAlign(
                               SingleLayerLstmCube::CeilAlign(cmpCw, SingleLayerLstmCube::C0_BYTES),
                               SingleLayerLstmCube::C0_BYTES) +
                           region(m * inSize);
    const uint32_t budget = SingleLayerLstmCube::CAP_UB - 8U * 1024;
    if (fixed >= budget) {
        return false;
    }
    for (uint32_t kt = PROJ_VEC_MAX_KTILE; kt >= PROJ_VEC_MIN_KTILE; kt /= 2) {
        if (fixed + region(kt * cw) <= budget) {
            *kTile = (kt < inSize) ? kt : inSize;
            return true;
        }
    }
    return false;
}

inline bool PickProjTiling(uint32_t mBlk, uint32_t inSize, uint32_t hid, uint32_t steps, uint32_t inBytes,
                           uint32_t* tChunk, uint32_t* kChunk, uint32_t* nChunk)
{
    constexpr uint32_t FSZ = static_cast<uint32_t>(sizeof(float));
    *tChunk = 0;
    *kChunk = 0;
    *nChunk = 0;
    if (mBlk == 0 || inSize == 0 || hid == 0 || steps == 0 || inBytes == 0) {
        return false;
    }
    /* The cube sees fp32 in this phase too, so the fractal is the fp32 one at every dtype and
     * `inBytes` reaches no extent below; the narrow slabs are widened into workspace before the
     * first Mmad, which is what SingleLayerLstmFwd::WidenInputs does.
     *
     * That retired two measured but unexplained narrow-operand bounds -- capL0C halved to 128 KB and
     * capM capping M at 1024. Both were keyed on operand width and reproduced only with 2-byte
     * operands, so neither is reachable at fp32; the shapes that exposed them (T=448 B=8 I=16 H=64
     * fp16, T=704 B=8 I=16 H=16 fp16) are in the device sweep and pass. Reinstate them only with a
     * fault seen at fp32 operands. input_size therefore keeps a multiple-of-8 rule at all three
     * dtypes instead of doubling to 16 at the narrow two. */
    const uint32_t c0 = SingleLayerLstmFwd::C0F;
    const uint32_t capL0C = SingleLayerLstmCube::CAP_L0C;
    /* input_size must be C0-aligned.
     * `kChunk` no longer has to divide it: a short last block is handled in the innermost loop. */
    if (inSize % c0 != 0 || hid % SingleLayerLstmFwd::C0F != 0) {
        return false;
    }
    (void)inBytes;

    /* Three chunks, picked outermost first. `tChunk` is how many timesteps share one cube pass,
     * `kChunk` how much of input_size goes to L0 at once, `nChunk` how many columns of one gate are
     * computed per cross-core round. The last is why hidden_size no longer bounds this phase: the B
     * tile is [kChunk, nChunk], the L0C accumulator [m, nChunk] and the drained UB plane
     * [rowsMax, nChunk], so H enters only as the number of rounds. tChunk descends first because it
     * is throughput; K then takes the largest divisor of input_size that fits L0A and leaves L0B
     * room for one fractal of N, and N takes what all four buffers allow. */
    const uint32_t hAl = SingleLayerLstmCube::CeilAlign(hid, SingleLayerLstmCube::CUBE_BLOCK);
    /* Headroom on the UB estimate for the Bump's 32-byte rounding on the one plane it allocates,
     * plus the bias row phase A reads. Small and fixed; the kernel's own L.ubBytes is the exact
     * figure and this only has to keep the estimate from over-picking. */
    constexpr uint32_t PROJ_SLACK = 8U * 1024;

    for (uint32_t tc = steps; tc >= 1; --tc) {
        if (steps % tc != 0) {
            if (tc == 1) {
                break;
            }
            continue;
        }
        const uint32_t m = tc * mBlk;
        const uint32_t mAl = SingleLayerLstmCube::CeilAlign(m, SingleLayerLstmCube::CUBE_BLOCK);
        const uint32_t rowsMax = SingleLayerLstmFwd::SPLIT ? SingleLayerLstmCube::CeilDiv(m, 2) : m;
        /* A in L1: the whole [m, I] when it fits alongside the B tile and the bias table, so one pass
         * of x serves every gate and column chunk; otherwise one [m, kChunk] block reloaded inside
         * the k loop, which is what makes a large input_size runnable at all -- at I=2056 with 80
         * aligned rows the whole tile is 658 KB against L1's 512. The choice is made by
         * SingleLayerLstmFwd::ProjAResident, which depends on nChunk; this function only has to make
         * the streamed tile fit, being the smaller of the two. */

        /* K needs no divisor: the last block is simply shorter, as both of phase B's axes already
         * are. Requiring one collapsed kChunk to the fractal width whenever input_size had no large
         * factor -- I=2056 (8 x 257) left only 8 and 2056, and 2056 does not fit L0A, so the phase
         * ran 257 blocks of 8 columns. That schedule is slow, and it is also the one on which a
         * ragged batch produced wrong results; the mechanism is not diagnosed, but the shapes
         * reaching it now pick kChunk in the hundreds. */
        /* TEST SCAFFOLDING, INERT WHEN UNSET. LSTM_FWD_KCHUNK caps the K block so one shape can run
         * with a different number of k blocks. On the legacy L0C-continuation path the count does
         * not change the numbers: at T=1 B=1 input_size=840 hidden_size=847 float32, caps of 512,
         * 256, 128 and 64 moved the projection from 1.613 ms to 0.980 ms while `y` came back
         * bit-identical, 3.1948e-07 from the float64 reference each time -- the L0C accumulation
         * across blocks is exactly the cube's continuation. The compensated path below resets L0C
         * and drains every partial, so its K cap does change the arithmetic. Capping downward only;
         * no production path sets it. */
        uint32_t kStart = SingleLayerLstmCube::CeilAlign(inSize, c0);
        const bool compensated = SingleLayerLstmFwd::CompensateCubeK(inSize);
        if (compensated && kStart > SingleLayerLstmFwd::CUBE_SUM_K_CHUNK) {
            kStart = SingleLayerLstmFwd::CUBE_SUM_K_CHUNK;
        }
        if (const char* env = std::getenv("LSTM_FWD_KCHUNK")) {
            const long v = std::atol(env);
            if (v >= static_cast<long>(c0)) {
                const uint32_t cap = static_cast<uint32_t>(v) / c0 * c0;
                if (cap < kStart) {
                    kStart = cap;
                }
            }
        }
        for (uint32_t k = kStart; k >= c0; k -= c0) {
            const uint32_t kAl = SingleLayerLstmCube::CeilAlign(k, SingleLayerLstmCube::CUBE_BLOCK);
            const uint32_t a2Elems = SingleLayerLstmCube::CeilDiv(k, c0) * mAl * c0;
            if (a2Elems * FSZ > SingleLayerLstmCube::CAP_L0A) {
                continue;
            }
            /* The streamed tile is what L1 has to hold whichever mode is chosen: ProjAResident
             * decides residency once nChunk is known and returns true only when the whole tile plus
             * this chunk's B tile and bias table fit, so [m, kChunk] is always the binding one.
             * Sizing against the whole [m, I] tile is what refused I=8192 H=8192 B=2, a shape that
             * streams A perfectly well. */
            const uint32_t a1Bytes = SingleLayerLstmCube::CeilAlign(a2Elems * FSZ, SingleLayerLstmCube::C0_BYTES);
            if (a1Bytes >= SingleLayerLstmCube::CAP_L1) {
                continue;
            }
            /* N per buffer. L0B holds [kChunk, nChunk] at the caller's width, L0C holds [m, nChunk]
             * in fp32 under the halved cap, and one drained plane is [rowsMax, nChunk] in UB. */
            uint32_t n = SingleLayerLstmCube::CAP_L0B / (kAl * FSZ);
            const uint32_t byL0C = capL0C / (mAl * FSZ);
            /* Long-K projection retains sum, correction and two scratch planes
             * beside the drain; a byte-per-lane mask is conservative. Slack
             * covers compare-repeat padding of all five planes. */
            const uint32_t ubBytesPerLane = compensated ? (5U * FSZ + 1U) : FSZ;
            const uint32_t byUb = (SingleLayerLstmCube::CAP_UB - PROJ_SLACK) / (rowsMax * ubBytesPerLane);
            if (n > byL0C) {
                n = byL0C;
            }
            if (n > byUb) {
                n = byUb;
            }
            if (n > hAl) {
                n = hAl;
            }
            n = n / SingleLayerLstmCube::CUBE_BLOCK * SingleLayerLstmCube::CUBE_BLOCK;
            /* L1 last, and it can only make n smaller. The A tile sized above is the STREAMED
             * one; if the whole tile also fits alongside this B tile the kernel will hold it
             * resident instead, and SingleLayerLstmFwd::ProjAResident -- which the kernel's Layout
             * calls with these very numbers -- is what says so. Either way this total is the one
             * that has to fit. */
            while (n >= SingleLayerLstmCube::CUBE_BLOCK) {
                const uint32_t bElems = SingleLayerLstmCube::CeilDiv(n, c0) * kAl * c0;
                const uint32_t l1Bytes = a1Bytes +
                                         SingleLayerLstmCube::CeilAlign(bElems * FSZ, SingleLayerLstmCube::C0_BYTES) +
                                         SingleLayerLstmCube::CeilAlign(SingleLayerLstmCube::BtElems(n) * FSZ,
                                                                        SingleLayerLstmCube::C0_BYTES);
                if (l1Bytes <= SingleLayerLstmCube::CAP_L1) {
                    break;
                }
                n -= SingleLayerLstmCube::CUBE_BLOCK;
            }
            if (n < SingleLayerLstmCube::CUBE_BLOCK) {
                continue;
            }
            if (n > hid) {
                n = hid; // one chunk covers the gate; hid itself need only be a multiple of 8
            }
            *tChunk = tc;
            *kChunk = k;
            *nChunk = n;
            return true;
        }
        if (tc == 1) {
            break;
        }
    }
    return false;
}

/* Phase B: does the persistent recurrence fit? Reads SingleLayerLstmFwd::Layout -- the object the
 * kernel builds -- rather than re-deriving its extents.
 *
 * It no longer bounds hidden_size: both GEMM axes are tiled and the weights stream when L1 cannot
 * hold them, so L0A, L0B, L0C and the weight side of L1 are fixed-size whatever H is. What scales
 * with H is the A tile ([mBlk, H], whole in L1 because every k-chunk reads a column range) and, at
 * the narrow dtypes, the UB the widening needs. Phase B is fp32 at every dtype, so the cube-side
 * budget is counted in fp32 bytes throughout; `inBytes` is passed to Layout only so L.ubBytes
 * accounts for the two output staging planes and the W_hh^T widening chunk. */
inline bool RecurrenceFits(uint32_t batch, uint32_t hid, uint32_t steps, uint32_t mBlk, uint32_t inBytes)
{
    if (hid == 0 || hid % SingleLayerLstmFwd::C0F != 0 || mBlk == 0 || inBytes == 0 || mBlk > batch) {
        return false;
    }
    const SingleLayerLstmFwd::Layout L(batch, hid, steps, mBlk, inBytes);
    if (L.kChunk == 0 || L.nChunk == 0) {
        return false; // not even one fractal fits L0A or L0B -- the chunk pickers said so
    }
    /* L0B holds one [kChunk, nChunk] tile, not a whole gate; checking L.bElems here is what capped
     * hidden_size at 128, since bElems is the L1 region. L0A likewise holds `aTileElems` -- the whole
     * A tile when resident, one [m, kChunk] chunk when streamed -- and the same number sizes the L1
     * tile, because the two buffers are resident or streamed together. L0C is four tiles: cOff[]
     * takes four from the same bump, and comparing a single tile against the whole capacity was a 4x
     * under-count that never bit at H=128. */
    return L.aTileElems * sizeof(float) <= SingleLayerLstmCube::CAP_L0A &&
           L.bChunkElems * sizeof(float) <= SingleLayerLstmCube::CAP_L0B &&
           GATES * L.cElems * sizeof(float) <= SingleLayerLstmCube::CAP_L0C &&
           L.l1Bytes <= SingleLayerLstmCube::CAP_L1 && L.ubBytes <= SingleLayerLstmCube::CAP_UB;
}

/* Batch rows one cluster puts on chip at a time, and the stride the grid walks the batch with.
 *
 * PickRowsPerBlock spreads the batch over at most MAX_CLUSTERS clusters, which fits whenever the
 * result fits on chip. It does not always: rows per cluster grows without bound past
 * 16 * MAX_CLUSTERS, and h_{t-1} is [rows, H] held whole in L1, so B=16384 gave 1024 rows and a 2 MB
 * A tile against L1's 512 KB. Five forward cases were refused that way.
 *
 * The answer is a second loop. Batch rows carry no dependency on one another, so a cluster runs the
 * whole T-step recurrence for one block of rows and then the next; phase A is lifted into the same
 * loop because it is partitioned on the same rows. The kernel strides by blockDim * rowChunk, which
 * keeps blocks disjoint and leaves the ragged tail to RowsOfBlock. The cost is one weight re-read
 * per block when the weights are not L1-resident; the widening prologue runs once per launch, and
 * the block count is 1 for every shape that fitted before.
 *
 * Halving rather than decrementing: both feasibility tests are monotone in the row count and the
 * search is a dozen calls. Returns 0 when even a single row does not fit, so the caller can re-run
 * the two tests at one row and name which failed. */
inline uint32_t PickRowChunk(uint32_t batch, uint32_t hid, uint32_t steps, uint32_t inSize, uint32_t inBytes)
{
    uint32_t rows = SingleLayerLstmFwd::PickRowsPerBlock(batch, MAX_CLUSTERS);
    while (rows != 0) {
        uint32_t tc = 0;
        uint32_t kc = 0;
        uint32_t nc = 0;
        if (RecurrenceFits(batch, hid, steps, rows, inBytes) &&
            PickProjTiling(rows, inSize, hid, steps, inBytes, &tc, &kc, &nc)) {
            return rows;
        }
        if (rows == 1) {
            break;
        }
        rows = SingleLayerLstmCube::CeilDiv(rows, 2);
    }
    return 0;
}

/* Workspace, in fp32 elements, for the image the cube reads: the fused weight as [I+H, 4H]. Zero at
 * fp32, where the caller's tensor already is fp32 and the kernel aliases it. x has no image: it is
 * fp32 at every operator width, because in a stacked LSTM layer l+1's x is layer l's h, which this
 * operator hands over at fp32. ProjInputImageFloats is kept because the unit tests pin its
 * arithmetic, and both tiling paths add zero for it.
 *
 * One copy for the whole grid -- every cluster reads all of w, so a per-cluster copy would multiply
 * this by blockDim. uint64 because these are element counts: 4H*(I+H) passes 4 G elements before the
 * caller's own weight does. */
inline uint64_t ProjInputImageFloats(uint32_t steps, uint32_t batch, uint32_t inSize, uint32_t inBytes)
{
    if (inBytes == sizeof(float)) {
        return 0;
    }
    return static_cast<uint64_t>(steps) * batch * inSize;
}

inline uint64_t WeightImageFloats(uint32_t inSize, uint32_t hid, uint32_t inBytes)
{
    if (inBytes == sizeof(float)) {
        return 0;
    }
    return static_cast<uint64_t>(inSize + hid) * GATES * hid;
}

} // namespace SingleLayerLstmBudget

#endif // OPS_RNN_SINGLE_LAYER_LSTM_BUDGET_H
