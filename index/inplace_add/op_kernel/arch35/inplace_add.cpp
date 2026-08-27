/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_add.cpp
 * \brief inplace_add
 */

#include "kernel_operator.h"
#include "inplace_add_tiling_data.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/cpp/kernel_simt_math_intf.h"
#include <limits>
#include <type_traits>

namespace {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;
// Copy-phase staging buffer. Double buffered, so 2 x 16KB of UB; the SIMT
// runtime keeps its own reservation and this has to fit alongside it.
constexpr uint32_t UB_CHUNK_BYTES = 16384;
constexpr int32_t DB_BUFFER = 2;
// Keep every core's slice starting on a 512B boundary so the DMA never straddles.
constexpr int64_t GM_ALIGN_BYTES = 512;

template <typename T>
__simt_callee__ __aicore__ inline T AddValue(T lhs, T rhs)
{
    if constexpr (std::is_integral<T>::value && std::is_signed<T>::value) {
        using UnsignedT = typename std::make_unsigned<T>::type;
        UnsignedT sum = static_cast<UnsignedT>(lhs);
        sum = static_cast<UnsignedT>(sum + static_cast<UnsignedT>(rhs));
        constexpr UnsignedT signedMax = static_cast<UnsignedT>(std::numeric_limits<T>::max());
        if (sum <= signedMax) {
            return static_cast<T>(sum);
        }
        const UnsignedT magnitude = static_cast<UnsignedT>(static_cast<UnsignedT>(0) - sum);
        if (magnitude == signedMax + static_cast<UnsignedT>(1)) {
            return std::numeric_limits<T>::min();
        }
        return static_cast<T>(-static_cast<T>(magnitude));
    }
    return static_cast<T>(lhs + rhs);
}

// Indices are normalized modulo n, so out-of-range values wrap instead of being
// rejected. The in-range test in front is not an optimization of the modulo but
// of the divide: every element of a row re-reads its index, and a runtime 64-bit
// remainder on that path costs more than the add it guards.
__simt_callee__ __aicore__ inline int64_t NormalizeIndex(int32_t index, int32_t n)
{
    if (index >= 0 && index < n) {
        return static_cast<int64_t>(index);
    }
    int64_t normalized = static_cast<int64_t>(index) % static_cast<int64_t>(n);
    if (normalized < 0) {
        normalized += static_cast<int64_t>(n);
    }
    return normalized;
}

// Phase one: y = x, over MTE, as one contiguous slice per core.
//
// The decomposition is flat rather than by rows on purpose. Splitting by rows
// ties thread occupancy to the row width and core occupancy to the row count,
// and both bind in practice: a rank-1 tensor has rowSize == 1, which left 1023
// of the 1024 threads idle, and an n = 16 tensor left 40 of the 56 cores idle
// whatever its row width. A flat range has neither property.
//
// The transport is DataCopy and not a SIMT vector store, and that is the part
// that took the longest to find. SyncAll raises its cross-core signal on
// PIPE_MTE3, so it publishes MTE traffic and only MTE traffic. Writing this
// phase with vf stores left them outside the barrier's reach, and no
// combination of PipeBarrier<PIPE_ALL>, DataCacheCleanAndInvalid and
// DataSyncBarrier<ALL> closed the window: peers kept reading rows this core had
// already written, at a steady ~0.3-0.5% of scattered destinations, every one
// of them holding the pristine x value. inplace_index_add_simt.h stages the
// same handoff through DataCopy for the same reason.
template <typename T>
__aicore__ inline void InplaceAddCopy(int64_t total, int64_t coreIdx, int64_t coreNum, __gm__ T* xGm, __gm__ T* yGm,
                                      TPipe& pipe)
{
    constexpr int64_t chunkElems = static_cast<int64_t>(UB_CHUNK_BYTES / sizeof(T));
    constexpr int64_t alignElems = static_cast<int64_t>(GM_ALIGN_BYTES / sizeof(T));

    int64_t perCore = (total + coreNum - 1) / coreNum;
    perCore = ((perCore + alignElems - 1) / alignElems) * alignElems;
    int64_t start = coreIdx * perCore;
    int64_t end = start + perCore;
    if (start > total) {
        start = total;
    }
    if (end > total) {
        end = total;
    }

    // TQueBind and not TQue: the queue position picks the sync events, and a
    // VECIN-only queue inserts MTE2->V, which does not hold the store back until
    // the load has landed. GM->UB->GM needs VECIN->VECOUT.
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> que;
    pipe.InitBuffer(que, DB_BUFFER, UB_CHUNK_BYTES);
    GlobalTensor<T> xGlobal;
    GlobalTensor<T> yGlobal;
    xGlobal.SetGlobalBuffer(xGm);
    yGlobal.SetGlobalBuffer(yGm);

    for (int64_t offset = start; offset < end; offset += chunkElems) {
        int64_t len = end - offset;
        if (len > chunkElems) {
            len = chunkElems;
        }
        DataCopyExtParams params{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        LocalTensor<T> inLocal = que.AllocTensor<T>();
        DataCopyPad(inLocal, xGlobal[offset], params, padParams);
        que.EnQue(inLocal);
        LocalTensor<T> outLocal = que.DeQue<T>();
        DataCopyPad(yGlobal[offset], outLocal, params);
        que.FreeTensor(outLocal);
    }
}

// Phase two: y[indices[kIdx]] += v[kIdx], as one flat grid-stride pass over the
// k * rowSize elements of v.
//
// This is the split that needs the cross-core barrier: a core writes rows it
// does not own, so every core must have finished phase one first. Before that
// barrier worked, each core had to scan the whole index array and keep only the
// hits landing in its own rows, which made the kernel's cost a function of K
// alone -- a flat ~1.3ns per index that no amount of intra-block parallelism
// removed, because the scan itself was the work.
//
// No atomics: the operator's contract normalizes indices into [0, n) and
// requires them to be unique there, so two elements of v never target the same
// element of y. Duplicate indices are explicitly undefined by the operator
// contract and are only required to stay memory-safe, which plain stores
// are. Sibling operators that do allow duplicates pay for asc_atomic_add, which
// on GM covers only int32/uint32/int64/uint64/float and forces int8/int16 through
// a widened workspace; staying non-atomic keeps all 13 dtypes on one path.
template <typename T, typename COMP_T, bool ROW_SIZE_ONE>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM) inline void InplaceAddAccumulate(
    COMP_T total, COMP_T blockIdx, COMP_T blockNum, COMP_T rowSize, COMP_T magic, COMP_T shift, int32_t n,
    __gm__ int32_t* indicesGm, __gm__ T* vGm, __gm__ T* yGm)
{
    const COMP_T stride = blockNum * static_cast<COMP_T>(THREAD_NUM);
    for (COMP_T element = blockIdx * static_cast<COMP_T>(THREAD_NUM) + static_cast<COMP_T>(threadIdx.x);
         element < total; element += stride) {
        COMP_T kIdx;
        COMP_T offset;
        if constexpr (ROW_SIZE_ONE) {
            // rowSize == 1 is the rank-1 shape family, and also the one where the
            // index cost dominates; the divide is skipped rather than handed a
            // magic pair for divisor 1.
            kIdx = element;
            offset = 0;
        } else {
            kIdx = Simt::UintDiv(element, magic, shift);
            offset = element - kIdx * rowSize;
        }
        const COMP_T dst = static_cast<COMP_T>(NormalizeIndex(indicesGm[kIdx], n)) * rowSize + offset;
        yGm[dst] = AddValue(yGm[dst], vGm[element]);
    }
}

// COMP_T is the flat element index type. Every offset in both phases is bounded
// by max(n, k) * rowSize, so a single 32-bit test covers them all; the 64-bit
// instantiation exists for the shapes that genuinely exceed it, where the wider
// UintDiv is the cheaper of the two problems.
template <typename T, typename COMP_T>
__aicore__ inline void LaunchAccumulate(int64_t accTotal, int64_t rowSize, int32_t n, int64_t coreIdx, int64_t coreNum,
                                        __gm__ int32_t* indicesGm, __gm__ T* vGm, __gm__ T* yGm)
{
    const COMP_T blockIdx = static_cast<COMP_T>(coreIdx);
    const COMP_T blockNum = static_cast<COMP_T>(coreNum);
    if (rowSize == 1) {
        asc_vf_call<InplaceAddAccumulate<T, COMP_T, true>>(dim3(THREAD_NUM), static_cast<COMP_T>(accTotal), blockIdx,
                                                           blockNum, static_cast<COMP_T>(1), static_cast<COMP_T>(0),
                                                           static_cast<COMP_T>(0), n, indicesGm, vGm, yGm);
        return;
    }
    COMP_T magic = 1;
    COMP_T shift = 1;
    GetUintDivMagicAndShift(magic, shift, static_cast<COMP_T>(rowSize));
    asc_vf_call<InplaceAddAccumulate<T, COMP_T, false>>(dim3(THREAD_NUM), static_cast<COMP_T>(accTotal), blockIdx,
                                                        blockNum, static_cast<COMP_T>(rowSize), magic, shift, n,
                                                        indicesGm, vGm, yGm);
}

template <typename T, int64_t ELEMENTS_PER_VALUE = INPLACE_ADD_SCALAR_COMPONENT_COUNT>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR indices, GM_ADDR v, GM_ADDR y, const InplaceAddTilingData* tilingData)
{
    __gm__ T* xGm = reinterpret_cast<__gm__ T*>(x);
    __gm__ int32_t* indicesGm = reinterpret_cast<__gm__ int32_t*>(indices);
    __gm__ T* vGm = reinterpret_cast<__gm__ T*>(v);
    __gm__ T* yGm = reinterpret_cast<__gm__ T*>(y);

    const int64_t coreIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t coreNum = static_cast<int64_t>(tilingData->needCoreNum);
    const int64_t rowSize = tilingData->innerSize * ELEMENTS_PER_VALUE;
    const int64_t n = static_cast<int64_t>(tilingData->n);
    const int64_t k = static_cast<int64_t>(tilingData->k);
    const int64_t copyTotal = n * rowSize;
    const int64_t accTotal = k * rowSize;
    const int64_t widest = copyTotal > accTotal ? copyTotal : accTotal;

    TPipe pipe;
    InplaceAddCopy<T>(copyTotal, coreIdx, coreNum, xGm, yGm, pipe);

    // The cross-core barrier. Two things have to be true for it to hold:
    // the copy above must be MTE traffic (SyncAll signals on PIPE_MTE3), and the
    // host tiling must declare batch mode (SetScheduleMode(1)) so the cores are
    // co-resident -- without it SyncAll returns immediately and every core's
    // index chunk loses its head element to the peer that copied it.
    // Every core reaches this, including the ones with no accumulate work.
    SyncAll();

    if (accTotal <= 0) {
        return;
    }
    if (widest <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        LaunchAccumulate<T, uint32_t>(accTotal, rowSize, tilingData->n, coreIdx, coreNum, indicesGm, vGm, yGm);
    } else {
        LaunchAccumulate<T, uint64_t>(accTotal, rowSize, tilingData->n, coreIdx, coreNum, indicesGm, vGm, yGm);
    }
}

} // namespace

extern "C" __global__ __aicore__ void inplace_add(GM_ADDR x, GM_ADDR indices, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(InplaceAddTilingData);
    GET_TILING_DATA_WITH_STRUCT(InplaceAddTilingData, tilingData, tiling);
    // Every core must take the same branch here: the cores that skip the work
    // must also skip the SyncAll inside Process, or the ones that reach it wait
    // for peers that never arrive.
    if (tilingData.needCoreNum == 0 || tilingData.n <= 0 || tilingData.innerSize <= 0) {
        return;
    }
#if ORIG_DTYPE_X == DT_COMPLEX32
    Process<half, INPLACE_ADD_COMPLEX_COMPONENT_COUNT>(x, indices, v, y, &tilingData);
#elif ORIG_DTYPE_X == DT_COMPLEX64
    Process<float, INPLACE_ADD_COMPLEX_COMPONENT_COUNT>(x, indices, v, y, &tilingData);
#else
    Process<DTYPE_X>(x, indices, v, y, &tilingData);
#endif
}
