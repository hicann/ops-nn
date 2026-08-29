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
 * \brief 确定性（WithSorted）三段式 SIMT/SIMD 内核：LinearIndex(SIMT) → Sort(radix) → 顺序分组累加(SIMT)。
 */

#ifndef SORT_LIB_SCATTER_ELEMENTS_V2_WITH_SORTED_H_
#define SORT_LIB_SCATTER_ELEMENTS_V2_WITH_SORTED_H_

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

// REDU_NONE/REDU_ADD/REDU_MUL 已在 scatter_elements.h 定义
constexpr uint32_t REDU_LAST = 3;

constexpr uint32_t WITH_SORTED_THREAD_NUM = 1024;
constexpr int64_t WITH_SORTED_UB_ALIGN = 32;
constexpr int64_t WITH_SORTED_UB_ALIGN_MASK = WITH_SORTED_UB_ALIGN - 1;

constexpr int64_t WITH_SORTED_COPY_DB_BUFFER = 1;
// Phase1 参数项数：rank 最多 8 维，每维 5 个参数
constexpr int64_t WITH_SORTED_PARAM_NUM = 8;

using WithSortedCopyQue = TQueBind<QuePosition::VECIN, QuePosition::VECOUT, WITH_SORTED_COPY_DB_BUFFER>;

__aicore__ inline int64_t WithSortedAlignAllocN(int64_t n, int64_t elemSize)
{
    int64_t totalBytes = n * elemSize;
    return ((totalBytes + WITH_SORTED_UB_ALIGN_MASK) / WITH_SORTED_UB_ALIGN) * WITH_SORTED_UB_ALIGN / elemSize;
}

// 精度类型提升（half / bfloat16_t -> float 累加，单次舍入写回）
// int8_t -> int32_t 累加
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

// 单段搬运
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

// 切核 + 分段循环
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
        for (int64_t idx = 0; idx < loopNum; idx++) {
            offset = blockIdx * normBlockData + idx * loopLength;
            WithSortedCopyToY<DATA_T>(copyQueue, dataGm, yGm, offset, loopLength);
        }

        if (tailLoopLength > 0) {
            offset = blockIdx * normBlockData + loopNum * loopLength;
            WithSortedCopyToY<DATA_T>(copyQueue, dataGm, yGm, offset, tailLoopLength);
        }
    }
}

// Phase 1: 计算 linear_index（单维索引 + dim + 前段子集）
template <typename IdxT, bool IsSubset, typename KeyT>
__simt_vf__ __aicore__ __launch_bounds__(WITH_SORTED_THREAD_NUM) inline void WithSortedPhase1LinearIdxKernel(
    int64_t totalIndexNum, int32_t dimNormalized, int32_t rank, __gm__ const IdxT* index, __gm__ KeyT* linearIdxOut,
    __gm__ volatile KeyT* srcPosOut, __ubuf__ const int64_t* varStrides, __ubuf__ const int64_t* indexStrides,
    __ubuf__ const uint64_t* indexDivMagic, __ubuf__ const uint64_t* indexDivShift, __ubuf__ const int64_t* srcStrides)
{
    for (int64_t pos = static_cast<int64_t>(blockIdx.x) * WITH_SORTED_THREAD_NUM + static_cast<int64_t>(threadIdx.x);
         pos < totalIndexNum; pos += static_cast<int64_t>(gridDim.x) * WITH_SORTED_THREAD_NUM) {
        // 1. 坐标反解：pos -> index 行主序坐标 coord[0..rank-1]（indexStrides 快除）
        uint64_t rem = static_cast<uint64_t>(pos);
        int64_t lin = 0;
        int64_t srcPos = 0;
        for (int32_t d = 0; d < rank; d++) {
            uint64_t coord = Simt::UintDiv<uint64_t>(rem, indexDivMagic[d], indexDivShift[d]);
            rem = rem - coord * static_cast<uint64_t>(indexStrides[d]);
            if (d != dimNormalized) {
                lin += static_cast<int64_t>(coord) * varStrides[d];
            }
            if constexpr (IsSubset) {
                srcPos += static_cast<int64_t>(coord) * srcStrides[d];
            }
        }
        // 2. 读取索引替换 dim 维，拼合线性索引。
        int64_t j = static_cast<int64_t>(index[pos]);
        lin += j * varStrides[dimNormalized];
        linearIdxOut[pos] = static_cast<KeyT>(lin);
        if constexpr (IsSubset) {
            srcPosOut[pos] = static_cast<KeyT>(srcPos);
        }
    }
}

// Phase 3: 分段归约（段首线程单写者，无原子操作）
// ReduMode: REDU_ADD=求和累加；REDU_LAST=最后写者赢（none）
template <typename T, bool IsSubset, typename KeyT, typename PermT, int ReduMode>
__simt_vf__ __aicore__ __launch_bounds__(WITH_SORTED_THREAD_NUM) inline void WithSortedPhase3ScatterAddKernel(
    int64_t totalIndexNum, __gm__ const T* src, __gm__ T* output, __gm__ const KeyT* sortedLinearIdx,
    __gm__ const PermT* perm, __gm__ const KeyT* srcPos)
{
    using AccT = typename WithSortedAccType<T>::type;
    for (int64_t pos = static_cast<int64_t>(blockIdx.x) * WITH_SORTED_THREAD_NUM + static_cast<int64_t>(threadIdx.x);
         pos < totalIndexNum; pos += static_cast<int64_t>(gridDim.x) * WITH_SORTED_THREAD_NUM) {
        int64_t target = static_cast<int64_t>(sortedLinearIdx[pos]);
        // 段首判据：pos 是该 target 段的第一个元素（全局判据，与核边界无关）
        if (pos > 0 && static_cast<int64_t>(sortedLinearIdx[pos - 1]) == target) {
            continue;
        }
        if constexpr (ReduMode == REDU_ADD) {
            // 本线程拥有该段，串行累加至段尾
            AccT acc = static_cast<AccT>(0);
            int64_t k = pos;
            while (k < totalIndexNum && sortedLinearIdx[k] == target) {
                if constexpr (IsSubset) {
                    acc += WithSortedToAcc<T>(src[srcPos[perm[k]]]);
                } else {
                    acc += WithSortedToAcc<T>(src[perm[k]]);
                }
                k++;
            }
            // 读 y 拷贝值作为累加基数（非 inplace：y 已由 Stage-0 复制 data），单写者写回
            AccT base = WithSortedToAcc<T>(output[target]);
            output[target] = WithSortedFromAcc<T>(base + acc);
        } else {
            // REDU_LAST：取本段 pos 最大（= perm 序最后）的更新，与旧确定性路径逐元素
            // 顺序覆盖语义一致（pos 序 = 原 updates 行主序）；多位置更新不再被求和
            int64_t k = pos;
            int64_t last = pos;
            while (k < totalIndexNum && sortedLinearIdx[k] == target) {
                last = k;
                k++;
            }
            if constexpr (IsSubset) {
                output[target] = src[srcPos[perm[last]]];
            } else {
                output[target] = src[perm[last]];
            }
        }
    }
}

// 三段式 Process<CountT, IsSubset, KeyT, PermT>
template <typename T, typename IdxT, typename CountT, bool IsSubset, typename KeyT, typename PermT,
          int ReduMode = REDU_ADD>
__aicore__ inline void WithSortedProcess(AscendC::TPipe* pipe, GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
                                         GM_ADDR output, GM_ADDR workspace, const ScatterElementsV2AscTilingData* td)
{
    __gm__ IdxT* idxGm = reinterpret_cast<__gm__ IdxT*>(indices);
    __gm__ T* srcGm = reinterpret_cast<__gm__ T*>(updates);
    __gm__ T* yGm = reinterpret_cast<__gm__ T*>(output);
    __gm__ char* usrWs = reinterpret_cast<__gm__ char*>(AscendC::GetUserWorkspace(workspace));

    int64_t indicesTotalNum = td->sortTiling.indicesTotalNum;
    int64_t dataAxis = td->dataAxis;
    if (indicesTotalNum == 0 || dataAxis == 0) {
        return;
    }

    // === Stage-0: data(var) -> y(output) 全量拷贝（非 inplace 基数） ===
    // 拷贝队列为满 UB 分配，需 pipe.Reset() 释放后再进入 Phase 1 的小幅参数缓冲。
    {
        WithSortedCopyQue copyQueue;
        pipe->InitBuffer(copyQueue, WITH_SORTED_COPY_DB_BUFFER, static_cast<uint32_t>(td->loopLength * sizeof(T)));
        WithSortedCopyDataToY<T>(copyQueue, var, output, dataAxis, td->loopLength);
        pipe->Reset();
    }
    // 拷贝末尾 MTE3 写 y 后，各核段互不重叠但可能被别核 Phase3 读，故需全核同步；
    SyncAll();

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

    // === Phase 1: 计算 linear_idx（workspace 段2；SUBSET 额外写 srcPos） ===
    // strides + magic/shift 参数缓冲（非满 UB），放同一 pipe 下，用完即 Reset。
    {
        TBuf<QuePosition::VECCALC> ubVarStridesBuf;
        TBuf<QuePosition::VECCALC> ubIndexStridesBuf;
        TBuf<QuePosition::VECCALC> ubIndexMagicBuf;
        TBuf<QuePosition::VECCALC> ubIndexShiftBuf;
        TBuf<QuePosition::VECCALC> ubSrcStridesBuf;
        constexpr uint32_t PARAM_ELEM_BYTES = WITH_SORTED_PARAM_NUM * sizeof(int64_t);
        pipe->InitBuffer(ubVarStridesBuf, PARAM_ELEM_BYTES);
        pipe->InitBuffer(ubIndexStridesBuf, PARAM_ELEM_BYTES);
        pipe->InitBuffer(ubIndexMagicBuf, WITH_SORTED_PARAM_NUM * sizeof(uint64_t));
        pipe->InitBuffer(ubIndexShiftBuf, WITH_SORTED_PARAM_NUM * sizeof(uint64_t));
        pipe->InitBuffer(ubSrcStridesBuf, PARAM_ELEM_BYTES);

        __ubuf__ int64_t* ubVarStrides = reinterpret_cast<__ubuf__ int64_t*>(
            ubVarStridesBuf.Get<int64_t>().GetPhyAddr());
        __ubuf__ int64_t* ubIndexStrides = reinterpret_cast<__ubuf__ int64_t*>(
            ubIndexStridesBuf.Get<int64_t>().GetPhyAddr());
        __ubuf__ uint64_t* ubIndexMagic = reinterpret_cast<__ubuf__ uint64_t*>(
            ubIndexMagicBuf.Get<uint64_t>().GetPhyAddr());
        __ubuf__ uint64_t* ubIndexShift = reinterpret_cast<__ubuf__ uint64_t*>(
            ubIndexShiftBuf.Get<uint64_t>().GetPhyAddr());
        __ubuf__ int64_t* ubSrcStrides = reinterpret_cast<__ubuf__ int64_t*>(
            ubSrcStridesBuf.Get<int64_t>().GetPhyAddr());

        for (int32_t d = 0; d < rank; d++) {
            ubVarStrides[d] = dataStridesBuf[d];
            ubIndexStrides[d] = indexStridesBuf[d];
            uint64_t magic = 0;
            uint64_t shift = 0;
            if (d < 7) {
                GetUintDivMagicAndShift<uint64_t>(magic, shift, static_cast<uint64_t>(indexStridesBuf[d]));
            } else {
                // d==7（末列，仅 rank==8 触达）：stride 隐含为 1，快除参数为 magic=1/shift=0
                magic = 1;
                shift = 0;
            }
            ubIndexMagic[d] = magic;
            ubIndexShift[d] = shift;
            if constexpr (IsSubset) {
                ubSrcStrides[d] = updatesStridesBuf[d];
            }
        }
        DataSyncBarrier<MemDsbT::UB>();
        // DataSyncBarrier 后，TQue/TBuf 的 phy 地址仍有效；SIMT 读 UB 经 DataSyncBarrier 保证可见。
        asc_vf_call<WithSortedPhase1LinearIdxKernel<IdxT, IsSubset, KeyT>>(
            dim3(WITH_SORTED_THREAD_NUM), indicesTotalNum, dimNormalized, rank, idxGm,
            (__gm__ KeyT*)(usrWs + td->sortTiling.wsLinearIdxOff),
            (__gm__ volatile KeyT*)(usrWs + td->sortTiling.wsSrcPosOff), ubVarStrides, ubIndexStrides, ubIndexMagic,
            ubIndexShift, ubSrcStrides);
        pipe->Reset();
    }
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);
    SyncAll();

    // === Phase 2: SortLib radix sort（升序排序 linear_idx） ===
    SortLib::SortParams p;
    p.numTileData = td->sortTiling.numTileData;
    p.tileCount = td->sortTiling.tileCount;
    p.activeCores = td->sortTiling.activeCores;
    p.tmpUbSize = td->sortTiling.tmpUbSize;
    p.totalElements = indicesTotalNum;
    p.isSingleCore = td->sortTiling.isSingleCore;

    // SortLib 的 workspace 基 = usrWs + 0（host wsLinearIdxOff 已跳过 multiSortWsBytes）
    pipe->Reset(); // 调用前释放此前占用的 UB，保证本接口可用满 UB
    SortLib::SortInvoke<KeyT, PermT, CountT, false>(
        pipe, (__gm__ KeyT*)(usrWs + td->sortTiling.wsLinearIdxOff), (__gm__ KeyT*)(usrWs + td->sortTiling.wsSortedOff),
        (__gm__ PermT*)(usrWs + td->sortTiling.wsPermOff), (__gm__ char*)(usrWs + 0), p);
    pipe->Reset(); // 调用后释放本接口占用的 UB，保证后续接口可用
    SyncAll();

    // === Phase 3: 分段归约（REDU_ADD 累加 / REDU_LAST 最后写者赢，读 y 基数写回 y） ===
    asc_vf_call<WithSortedPhase3ScatterAddKernel<T, IsSubset, KeyT, PermT, ReduMode>>(
        dim3(WITH_SORTED_THREAD_NUM), indicesTotalNum, (__gm__ const T*)srcGm, (__gm__ T*)yGm,
        (__gm__ const KeyT*)(usrWs + td->sortTiling.wsSortedOff),
        (__gm__ const PermT*)(usrWs + td->sortTiling.wsPermOff),
        (__gm__ const KeyT*)(usrWs + td->sortTiling.wsSrcPosOff));
}

template <typename T, typename IDX_T, int ReduMode = REDU_ADD>
class KernelScatterElementsWithSorted {
public:
    __aicore__ inline KernelScatterElementsWithSorted(const ScatterElementsV2AscTilingData* tilingData,
                                                      AscendC::TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace)
    {
        var_ = var;
        indices_ = indices;
        updates_ = updates;
        y_ = y;
        workspace_ = workspace;
    }

    __aicore__ inline void Process()
    {
        const int32_t countMode = tilingData_->sortTiling.countMode;                     // 0=uint32, 1=int64 计数
        const int32_t shapeMode = tilingData_->sortTiling.shapeMode;                     // 0=SAME, 1=SUBSET
        const int32_t keySize = static_cast<int32_t>(tilingData_->sortTiling.keySize);   // 2/4/8
        const int32_t permSize = static_cast<int32_t>(tilingData_->sortTiling.permSize); // 4=int32, 8=int64

        // 按 countMode × shapeMode × keySize × permSize 组合 dispatch（编译期实例化，避免运行时 SIMT 分支开销）
        if (countMode == 0) { // uint32_t 计数
            if (shapeMode == 0) {
                DispatchKey<false, uint32_t>(keySize, permSize);
            } else {
                DispatchKey<true, uint32_t>(keySize, permSize);
            }
        } else { // int64_t 计数
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
        // keySize = sortTiling.keySize（2=int16, 4=int32, 8=int64）
        if (keySize == 2) {
            DispatchPerm<int16_t, IsSubset, CountT>(permSize);
        } else if (keySize == 4) {
            DispatchPerm<int32_t, IsSubset, CountT>(permSize);
        } else {
            DispatchPerm<int64_t, IsSubset, CountT>(permSize);
        }
    }

    template <typename KeyT, bool IsSubset, typename CountT>
    __aicore__ inline void DispatchPerm(int32_t permSize)
    {
        // permSize = sortTiling.permSize（4=int32, 8=int64，host 按 countMode 决定）
        if (permSize == 4) {
            WithSortedProcess<T, IDX_T, CountT, IsSubset, KeyT, int32_t, ReduMode>(pipe_, var_, indices_, updates_, y_,
                                                                                   workspace_, tilingData_);
        } else {
            WithSortedProcess<T, IDX_T, CountT, IsSubset, KeyT, int64_t, ReduMode>(pipe_, var_, indices_, updates_, y_,
                                                                                   workspace_, tilingData_);
        }
    }

    const ScatterElementsV2AscTilingData* tilingData_;
    AscendC::TPipe* pipe_;
    GM_ADDR var_ = nullptr;
    GM_ADDR indices_ = nullptr;
    GM_ADDR updates_ = nullptr;
    GM_ADDR y_ = nullptr;
    GM_ADDR workspace_ = nullptr;
};

} // namespace ScatterElements

#endif // SORT_LIB_SCATTER_ELEMENTS_V2_WITH_SORTED_H_
