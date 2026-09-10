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
 * \file scatter_elements_with_sorted.h
 * \brief Deterministic global-sort kernel: copy, build linear indices, radix sort, then grouped reduction.
 */

#ifndef SORT_LIB_SCATTER_ELEMENTS_WITH_SORTED_H_
#define SORT_LIB_SCATTER_ELEMENTS_WITH_SORTED_H_

#include <type_traits>

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "../../sort_lib/arch35/sort_lib.h"

namespace ScatterElements {

constexpr uint32_t REDU_LAST = 3;
constexpr uint32_t WITH_SORTED_THREAD_NUM = 1024;
constexpr int64_t WITH_SORTED_COPY_DB_BUFFER = 1;
constexpr int64_t WITH_SORTED_PARAM_NUM = 8;

using WithSortedCopyQue = TQueBind<QuePosition::VECIN, QuePosition::VECOUT, WITH_SORTED_COPY_DB_BUFFER>;

template <typename T>
struct WithSortedAccType {
    using type = T;
};

template <>
struct WithSortedAccType<half> {
    using type = float;
};

template <>
struct WithSortedAccType<bfloat16_t> {
    using type = float;
};

template <>
struct WithSortedAccType<int8_t> {
    using type = int32_t;
};

template <>
struct WithSortedAccType<uint8_t> {
    using type = int32_t;
};

template <>
struct WithSortedAccType<int16_t> {
    using type = int32_t;
};

template <typename T>
__simt_callee__ inline typename WithSortedAccType<T>::type WithSortedToAcc(T x)
{
    if constexpr (std::is_same<T, half>::value) {
        return __half2float(x);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        return __bfloat162float(x);
    } else {
        return static_cast<typename WithSortedAccType<T>::type>(x);
    }
}

template <typename T>
__simt_callee__ inline T WithSortedFromAcc(typename WithSortedAccType<T>::type x)
{
    if constexpr (std::is_same<T, half>::value) {
        return __float2half(x);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        return __float2bfloat16(x);
    } else {
        return static_cast<T>(x);
    }
}

template <typename DATA_T>
__aicore__ inline void WithSortedCopyToY(WithSortedCopyQue& copyQueue, const GlobalTensor<DATA_T>& dataGm,
                                         const GlobalTensor<DATA_T>& yGm, int64_t offset, int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(DATA_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<DATA_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                              static_cast<DATA_T>(0)};
    LocalTensor<DATA_T> xLocal = copyQueue.AllocTensor<DATA_T>();
    DataCopyPad(xLocal, dataGm[offset], copyParams, padParams);
    copyQueue.EnQue(xLocal);

    LocalTensor<DATA_T> yLocal = copyQueue.DeQue<DATA_T>();
    DataCopyPad(yGm[offset], yLocal, copyParams);
    copyQueue.FreeTensor(yLocal);
}

template <typename DATA_T>
__aicore__ inline void WithSortedCopyDataToY(WithSortedCopyQue& copyQueue, GM_ADDR data, GM_ADDR y, int64_t dataAxis,
                                             int64_t loopLength)
{
    GlobalTensor<DATA_T> dataGm;
    GlobalTensor<DATA_T> yGm;
    dataGm.SetGlobalBuffer(reinterpret_cast<__gm__ DATA_T*>(data));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DATA_T*>(y));

    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    int64_t normBlockData = ops::CeilDiv(dataAxis, blockNum);
    int64_t usedCoreNum = ops::CeilDiv(dataAxis, normBlockData);
    int64_t tailBlockData = dataAxis - (usedCoreNum - 1) * normBlockData;
    int64_t curCoreData = blockIdx != (usedCoreNum - 1) ? normBlockData : tailBlockData;
    int64_t loopNum = curCoreData / loopLength;
    int64_t tailLoopLength = curCoreData - loopNum * loopLength;

    if (blockIdx < usedCoreNum) {
        int64_t offset = 0;
        for (int64_t idx = 0; idx < loopNum; ++idx) {
            offset = blockIdx * normBlockData + idx * loopLength;
            WithSortedCopyToY<DATA_T>(copyQueue, dataGm, yGm, offset, loopLength);
        }
        if (tailLoopLength > 0) {
            offset = blockIdx * normBlockData + loopNum * loopLength;
            WithSortedCopyToY<DATA_T>(copyQueue, dataGm, yGm, offset, tailLoopLength);
        }
    }
}

template <typename IdxT, bool IsSubset, typename KeyT>
__simt_vf__ __aicore__ __launch_bounds__(WITH_SORTED_THREAD_NUM) inline void WithSortedPhase1LinearIdxKernel(
    int64_t totalIndexNum, int32_t dimNormalized, int32_t rank, __gm__ const IdxT* index, __gm__ KeyT* linearIdxOut,
    __gm__ volatile KeyT* srcPosOut, __ubuf__ const int64_t* dataStrides, __ubuf__ const int64_t* indexStrides,
    __ubuf__ const uint64_t* indexDivMagic, __ubuf__ const uint64_t* indexDivShift,
    __ubuf__ const int64_t* updatesStrides)
{
    for (int64_t pos = static_cast<int64_t>(blockIdx.x) * WITH_SORTED_THREAD_NUM + static_cast<int64_t>(threadIdx.x);
         pos < totalIndexNum; pos += static_cast<int64_t>(gridDim.x) * WITH_SORTED_THREAD_NUM) {
        uint64_t rem = static_cast<uint64_t>(pos);
        int64_t linearIndex = 0;
        int64_t updatePos = 0;
        for (int32_t d = 0; d < rank; ++d) {
            uint64_t coord = Simt::UintDiv<uint64_t>(rem, indexDivMagic[d], indexDivShift[d]);
            rem -= coord * static_cast<uint64_t>(indexStrides[d]);
            if (d != dimNormalized) {
                linearIndex += static_cast<int64_t>(coord) * dataStrides[d];
            }
            if constexpr (IsSubset) {
                updatePos += static_cast<int64_t>(coord) * updatesStrides[d];
            }
        }
        linearIndex += static_cast<int64_t>(index[pos]) * dataStrides[dimNormalized];
        linearIdxOut[pos] = static_cast<KeyT>(linearIndex);
        if constexpr (IsSubset) {
            srcPosOut[pos] = static_cast<KeyT>(updatePos);
        }
    }
}

template <typename T, bool IsSubset, typename KeyT, typename PermT, int ReduMode>
__simt_vf__ __aicore__ __launch_bounds__(WITH_SORTED_THREAD_NUM) inline void WithSortedPhase3ScatterKernel(
    int64_t totalIndexNum, __gm__ const T* updates, __gm__ T* output, __gm__ const KeyT* sortedLinearIdx,
    __gm__ const PermT* perm, __gm__ const KeyT* srcPos)
{
    using AccT = typename WithSortedAccType<T>::type;
    for (int64_t pos = static_cast<int64_t>(blockIdx.x) * WITH_SORTED_THREAD_NUM + static_cast<int64_t>(threadIdx.x);
         pos < totalIndexNum; pos += static_cast<int64_t>(gridDim.x) * WITH_SORTED_THREAD_NUM) {
        int64_t target = static_cast<int64_t>(sortedLinearIdx[pos]);
        if (pos > 0 && static_cast<int64_t>(sortedLinearIdx[pos - 1]) == target) {
            continue;
        }
        if constexpr (ReduMode == REDU_ADD) {
            AccT acc = static_cast<AccT>(0);
            int64_t k = pos;
            while (k < totalIndexNum && sortedLinearIdx[k] == target) {
                if constexpr (IsSubset) {
                    acc += WithSortedToAcc<T>(updates[srcPos[perm[k]]]);
                } else {
                    acc += WithSortedToAcc<T>(updates[perm[k]]);
                }
                ++k;
            }
            AccT base = WithSortedToAcc<T>(output[target]);
            output[target] = WithSortedFromAcc<T>(base + acc);
        } else {
            int64_t k = pos;
            int64_t last = pos;
            while (k < totalIndexNum && sortedLinearIdx[k] == target) {
                last = k;
                ++k;
            }
            if constexpr (IsSubset) {
                output[target] = updates[srcPos[perm[last]]];
            } else {
                output[target] = updates[perm[last]];
            }
        }
    }
}

template <typename T, typename IdxT, typename CountT, bool IsSubset, typename KeyT, typename PermT,
          int ReduMode = REDU_ADD>
__aicore__ inline void WithSortedProcess(AscendC::TPipe* pipe, GM_ADDR data, GM_ADDR indices, GM_ADDR updates,
                                         GM_ADDR output, GM_ADDR workspace, const ScatterElementsTilingData* td)
{
    __gm__ IdxT* indicesGm = reinterpret_cast<__gm__ IdxT*>(indices);
    __gm__ T* updatesGm = reinterpret_cast<__gm__ T*>(updates);
    __gm__ T* outputGm = reinterpret_cast<__gm__ T*>(output);
    __gm__ char* userWorkspace = reinterpret_cast<__gm__ char*>(AscendC::GetUserWorkspace(workspace));

    int64_t indicesTotalNum = td->sortTiling.indicesTotalNum;
    int64_t dataAxis = td->dataAxis;
    if (dataAxis == 0) {
        return;
    }

    {
        WithSortedCopyQue copyQueue;
        pipe->InitBuffer(copyQueue, WITH_SORTED_COPY_DB_BUFFER, static_cast<uint32_t>(td->loopLength * sizeof(T)));
        WithSortedCopyDataToY<T>(copyQueue, data, output, dataAxis, td->loopLength);
        pipe->Reset();
    }
    SyncAll();

    if (indicesTotalNum == 0) {
        return;
    }

    int32_t rank = td->rank;
    int32_t dimNormalized = td->sortTiling.dimNormalized;
    int64_t indexStridesBuf[8];
    int64_t dataStridesBuf[8];
    int64_t updatesStridesBuf[8];
    for (int32_t d = 0; d < 7; ++d) {
        indexStridesBuf[d] = static_cast<int64_t>(td->indicesStride[d]);
        dataStridesBuf[d] = static_cast<int64_t>(td->dataStride[d]);
        updatesStridesBuf[d] = static_cast<int64_t>(td->updatesStride[d]);
    }
    indexStridesBuf[7] = 1;
    dataStridesBuf[7] = 1;
    updatesStridesBuf[7] = 1;

    {
        TBuf<QuePosition::VECCALC> ubDataStridesBuf;
        TBuf<QuePosition::VECCALC> ubIndexStridesBuf;
        TBuf<QuePosition::VECCALC> ubIndexMagicBuf;
        TBuf<QuePosition::VECCALC> ubIndexShiftBuf;
        TBuf<QuePosition::VECCALC> ubUpdatesStridesBuf;
        constexpr uint32_t PARAM_ELEM_BYTES = WITH_SORTED_PARAM_NUM * sizeof(int64_t);
        pipe->InitBuffer(ubDataStridesBuf, PARAM_ELEM_BYTES);
        pipe->InitBuffer(ubIndexStridesBuf, PARAM_ELEM_BYTES);
        pipe->InitBuffer(ubIndexMagicBuf, WITH_SORTED_PARAM_NUM * sizeof(uint64_t));
        pipe->InitBuffer(ubIndexShiftBuf, WITH_SORTED_PARAM_NUM * sizeof(uint64_t));
        pipe->InitBuffer(ubUpdatesStridesBuf, PARAM_ELEM_BYTES);

        __ubuf__ int64_t* ubDataStrides = reinterpret_cast<__ubuf__ int64_t*>(
            ubDataStridesBuf.Get<int64_t>().GetPhyAddr());
        __ubuf__ int64_t* ubIndexStrides = reinterpret_cast<__ubuf__ int64_t*>(
            ubIndexStridesBuf.Get<int64_t>().GetPhyAddr());
        __ubuf__ uint64_t* ubIndexMagic = reinterpret_cast<__ubuf__ uint64_t*>(
            ubIndexMagicBuf.Get<uint64_t>().GetPhyAddr());
        __ubuf__ uint64_t* ubIndexShift = reinterpret_cast<__ubuf__ uint64_t*>(
            ubIndexShiftBuf.Get<uint64_t>().GetPhyAddr());
        __ubuf__ int64_t* ubUpdatesStrides = reinterpret_cast<__ubuf__ int64_t*>(
            ubUpdatesStridesBuf.Get<int64_t>().GetPhyAddr());

        for (int32_t d = 0; d < rank; ++d) {
            ubDataStrides[d] = dataStridesBuf[d];
            ubIndexStrides[d] = indexStridesBuf[d];
            uint64_t magic = 0;
            uint64_t shift = 0;
            if (d < 7) {
                GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(indexStridesBuf[d]));
            } else {
                magic = 1;
                shift = 0;
            }
            ubIndexMagic[d] = magic;
            ubIndexShift[d] = shift;
            if constexpr (IsSubset) {
                ubUpdatesStrides[d] = updatesStridesBuf[d];
            }
        }
        DataSyncBarrier<MemDsbT::UB>();
        asc_vf_call<WithSortedPhase1LinearIdxKernel<IdxT, IsSubset, KeyT>>(
            dim3(WITH_SORTED_THREAD_NUM), indicesTotalNum, dimNormalized, rank, indicesGm,
            (__gm__ KeyT*)(userWorkspace + td->sortTiling.wsLinearIdxOff),
            (__gm__ volatile KeyT*)(userWorkspace + td->sortTiling.wsSrcPosOff), ubDataStrides, ubIndexStrides,
            ubIndexMagic, ubIndexShift, ubUpdatesStrides);
        pipe->Reset();
    }
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    SyncAll();

    SortLib::SortParams params;
    params.numTileData = td->sortTiling.numTileData;
    params.tileCount = td->sortTiling.tileCount;
    params.activeCores = td->sortTiling.activeCores;
    params.tmpUbSize = td->sortTiling.tmpUbSize;
    params.totalElements = indicesTotalNum;
    params.isSingleCore = td->sortTiling.isSingleCore;

    pipe->Reset();
    if (params.isSingleCore == 1) {
        if (GetBlockIdx() == 0) {
            SortLib::SortInvoke<KeyT, PermT, CountT, false>(
                pipe, (__gm__ KeyT*)(userWorkspace + td->sortTiling.wsLinearIdxOff),
                (__gm__ KeyT*)(userWorkspace + td->sortTiling.wsSortedOff),
                (__gm__ PermT*)(userWorkspace + td->sortTiling.wsPermOff), (__gm__ char*)(userWorkspace), params);
        }
    } else {
        SortLib::SortInvoke<KeyT, PermT, CountT, false>(
            pipe, (__gm__ KeyT*)(userWorkspace + td->sortTiling.wsLinearIdxOff),
            (__gm__ KeyT*)(userWorkspace + td->sortTiling.wsSortedOff),
            (__gm__ PermT*)(userWorkspace + td->sortTiling.wsPermOff), (__gm__ char*)(userWorkspace), params);
    }
    pipe->Reset();
    SyncAll();

    asc_vf_call<WithSortedPhase3ScatterKernel<T, IsSubset, KeyT, PermT, ReduMode>>(
        dim3(WITH_SORTED_THREAD_NUM), indicesTotalNum, (__gm__ const T*)updatesGm, (__gm__ T*)outputGm,
        (__gm__ const KeyT*)(userWorkspace + td->sortTiling.wsSortedOff),
        (__gm__ const PermT*)(userWorkspace + td->sortTiling.wsPermOff),
        (__gm__ const KeyT*)(userWorkspace + td->sortTiling.wsSrcPosOff));
}

template <typename T, typename IDX_T, int ReduMode = REDU_ADD>
class KernelScatterElementsWithSorted {
public:
    __aicore__ inline KernelScatterElementsWithSorted(const ScatterElementsTilingData* tilingData, AscendC::TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR data, GM_ADDR indices, GM_ADDR updates, GM_ADDR output, GM_ADDR workspace)
    {
        data_ = data;
        indices_ = indices;
        updates_ = updates;
        output_ = output;
        workspace_ = workspace;
    }

    __aicore__ inline void Process()
    {
        const int32_t countMode = tilingData_->sortTiling.countMode;
        const int32_t shapeMode = tilingData_->sortTiling.shapeMode;
        const int32_t keySize = static_cast<int32_t>(tilingData_->sortTiling.keySize);
        const int32_t permSize = static_cast<int32_t>(tilingData_->sortTiling.permSize);

        if (countMode == 0) {
            if (shapeMode == 0) {
                DispatchKey<false, uint32_t>(keySize, permSize);
            } else {
                DispatchKey<true, uint32_t>(keySize, permSize);
            }
        } else {
            if (shapeMode == 0) {
                DispatchKey<false, int64_t>(keySize, permSize);
            } else {
                DispatchKey<true, int64_t>(keySize, permSize);
            }
        }
    }

private:
    template <bool IsSubset, typename CountT>
    __aicore__ inline void DispatchKey(int32_t keySize, int32_t permSize)
    {
        if (keySize == 2) {
            DispatchPerm<int16_t, IsSubset, CountT>(permSize);
        } else if (keySize == 4) {
            DispatchPerm<uint32_t, IsSubset, CountT>(permSize);
        } else {
            DispatchPerm<int64_t, IsSubset, CountT>(permSize);
        }
    }

    template <typename KeyT, bool IsSubset, typename CountT>
    __aicore__ inline void DispatchPerm(int32_t permSize)
    {
        if (permSize == 4) {
            WithSortedProcess<T, IDX_T, CountT, IsSubset, KeyT, uint32_t, ReduMode>(pipe_, data_, indices_, updates_,
                                                                                    output_, workspace_, tilingData_);
        } else {
            WithSortedProcess<T, IDX_T, CountT, IsSubset, KeyT, int64_t, ReduMode>(pipe_, data_, indices_, updates_,
                                                                                   output_, workspace_, tilingData_);
        }
    }

    const ScatterElementsTilingData* tilingData_;
    AscendC::TPipe* pipe_;
    GM_ADDR data_ = nullptr;
    GM_ADDR indices_ = nullptr;
    GM_ADDR updates_ = nullptr;
    GM_ADDR output_ = nullptr;
    GM_ADDR workspace_ = nullptr;
};

} // namespace ScatterElements

#endif // SORT_LIB_SCATTER_ELEMENTS_WITH_SORTED_H_
