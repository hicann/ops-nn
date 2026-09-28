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
 * \file single_layer_lstm_tiling_data.h
 * \brief TilingData for SingleLayerLstm -- fp32, ascend950 (dav_3510). Every field is decided on the
 *        host; the kernel does no capacity test at all, because a kernel that tests on device and
 *        returns leaves the caller's outputs at whatever they held, indistinguishable from a
 *        precision bug. SingleLayerLstmTilingFunc refuses instead, naming which check failed.
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_TILING_DATA_H
#define OPS_RNN_SINGLE_LAYER_LSTM_TILING_DATA_H

#include <cstdint>

struct __attribute__((aligned(8))) SingleLayerLstmTilingData {
    uint32_t batch = 0;      // B
    uint32_t inputSize = 0;  // I
    uint32_t hiddenSize = 0; // H
    uint32_t timeStep = 0;   // T
    uint32_t seqLenMax = 0;  // <= T; steps beyond it are zero-filled in every output

    /* Batch rows one cluster holds on chip at a time (mBlk) -- the only partitioning axis, being the
     * only one with no cross-cluster dependency. It is not the rows a cluster owns: h_{t-1} is
     * [rowsPerBlock, H] held whole in L1, so at B=16384 spreading over 16 clusters alone gave 1024
     * rows and a 2 MB A tile against L1's 512 KB. The grid walks the batch in blocks of this size,
     * striding by blockDim * rowsPerBlock. See SingleLayerLstmBudget::PickRowChunk. */
    uint32_t rowsPerBlock = 0;
    /* min(CeilDiv(B, rowsPerBlock), MAX_CLUSTERS) -- the number of blocks when they fit on the
     * part, and the cluster count when they do not. Every launched cluster gets at least one
     * block, which both the stride loop and the widening prologue's band assignment assume. */
    uint32_t blockDim = 0;

    /* Input projection, phase A. `tChunk` timesteps are projected per cube pass, so the A tile is
     * [tChunk * rowsPerBlock, I]. Chunking is over TIME, never over batch rows WITHIN A BLOCK: a
     * block must end up holding the projection for exactly the rows it will then recur over, or
     * phase B would need to read bytes another block produced. Phase A rides the same block loop as
     * phase B for that reason, so the two always see the same rows. */
    uint32_t tChunk = 0;
    /* K IS BLOCKED TOO, and that is not optional: the projection's B operand is [I, H], which at
     * I=256 H=128 is 128 KB against L0B's 64 KB. I is an INPUT dimension, not something the shapes
     * that happen to get tested keep small. kChunk NEED NOT divide I: the last block is simply
     * shorter, and the innermost loop carries the second fractal count that costs. Requiring a
     * divisor collapsed kChunk to the fractal width whenever input_size had no large factor. */
    uint32_t kChunk = 0;
    /* Columns of ONE GATE per cube pass and per cross-core round. Phase A's B tile is
     * [kChunk, projNChunk], its L0C accumulator [tChunk*rowsPerBlock, projNChunk] and its drained UB
     * plane [rowsMax, projNChunk] -- all three used to be hidden_size wide, which is what capped
     * hidden_size at 1024 in fp32 before this axis was tiled. Named apart from phase B's own N chunk
     * (SingleLayerLstmFwd::Layout::nChunk) because the two are picked against different buffers and
     * are not usually equal. */
    uint32_t projNChunk = 0;

    /* Workspace offsets in ELEMENTS (fp32), from the start of the operator's own area. */
    uint32_t offIgates = 0; // [T, B, 4H]     phase A's output, phase B's input
    uint32_t offHAll = 0;   // [T+1, B, H]    slot 0 = init_h
    uint32_t offCAll = 0;   // [T+1, B, H]    slot 0 = init_c
    uint32_t offStore = 0;  // [T, B, 4H]     i, f, j, o post-activation
    /* THE FP32 IMAGES OF THE TWO SLABS THE CUBE READS, and both are ZERO-SIZED AT FP32 where the
     * caller's own tensors already are fp32.
     *   offXF32   [T, B, I]      phase A's A operand
     *   offWF32   [I+H, 4H]      rows [0, I) are phase A's B operand, rows [I, I+H) are W_hh^T
     * Mmad's operands must share a width and the recurrence is fp32, so the cube reads fp32
     * everywhere; at fp16 and bf16 the AIVs widen both slabs into these once per launch, before
     * either phase starts. One copy for the whole grid, not one per cluster. */
    uint32_t offXF32 = 0;
    uint32_t offWF32 = 0;
    uint32_t offBF32 = 0;

    /* Phase A on the vector units instead of the cube: 1 selects it, 0 leaves phase A on the cube.
     * Chosen on the host for shapes where the cube is both wasteful and imprecise -- at
     * m = timeStep * rowsPerBlock <= PROJ_VEC_MAX_ROWS its M is mostly padding, and at
     * input_size >= PROJ_VEC_MIN_K its single fp32 chain over K costs enough rounding to show.
     * Both halves must agree: the cube half skips phase A entirely when this is 1, so the vector
     * half must skip the cross-core rounds too or the launch hangs. */
    uint32_t projVec = 0;
    uint32_t projVecKTile = 0; // w rows staged in UB per burst on that path

    /* Semantic reduction extents, separate from padded storage strides above. Every host
     * producer must populate these explicitly. logicalInputSize == 0 is a real empty input
     * projection, NOT a sentinel for inputSize. The production no-padding interface sets
     * these equal to inputSize/hiddenSize; the standalone stack preserves pre-padding widths. */
    uint32_t logicalInputSize = 0;
    uint32_t logicalHiddenSize = 0;
    uint32_t hasBiasHh = 0;
};

#endif // OPS_RNN_SINGLE_LAYER_LSTM_TILING_DATA_H
