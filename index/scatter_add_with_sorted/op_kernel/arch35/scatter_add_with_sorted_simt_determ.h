/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License")
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SCATTER_ADD_WITH_SORTED_DETERM_SIMT_H
#define SCATTER_ADD_WITH_SORTED_DETERM_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "scatter_add_with_sorted_struct.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace ScatterAddWithSorted {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_DETERM = 256;
#else
constexpr uint32_t THREAD_NUM_DETERM = 1024;
#endif
constexpr uint32_t TWO = 2;


template <typename T, typename U, typename ADDR_T, bool isStartRowCore, bool isEndRowCore, bool withPos>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_DETERM) inline void ScatterAddWithSortedSimtDetermStep1(
    ADDR_T totalCol, ADDR_T indicesBlockOffset, ADDR_T indicesBlockNum, ADDR_T varFirstDimSize, ADDR_T magic,
    ADDR_T shift, uint32_t blockIdx, __gm__ T* var, __gm__ U* sortedIndices, __gm__ U* pos, __gm__ T* updates,
    __gm__ T* sumWorkspace, __gm__ U* indexWorkspace)
{
    ADDR_T startIndices = indicesBlockOffset;
    ADDR_T endIndices = startIndices + indicesBlockNum;
    ADDR_T updatesBlockOffset = startIndices * totalCol;
    ADDR_T updatesBlockNum = indicesBlockNum * totalCol;
    ADDR_T updatesBlockEnd = updatesBlockOffset + updatesBlockNum;

    for (ADDR_T i = updatesBlockOffset + threadIdx.x; i < updatesBlockEnd; i += blockDim.x) {
        uint32_t flag1 = 0;
        uint32_t flag2 = 0;
        ADDR_T indiceRow = Simt::UintDiv(i, magic, shift);
        U varRow = sortedIndices[indiceRow];
        if (varRow < 0 || varRow >= varFirstDimSize) {   // 跳过无效索引
            continue;
        }

        bool isCoreFirstIndices = indiceRow == startIndices;
        if (!isCoreFirstIndices) {
            if (varRow == sortedIndices[indiceRow - 1]) { // 当前线程对应索引与前一个索引相同，由前一个线程处理
                continue;
            }
        }
        if constexpr (!isStartRowCore) {
            if (isCoreFirstIndices && varRow == sortedIndices[startIndices - 1]) {  // 判断头索引与前一个核的尾索引是否相同
                flag1 = 1;
            }
        }

        T sumRes = 0;
        ADDR_T currentRow = 0;
        ADDR_T colIdx = i - indiceRow * totalCol;
        for (ADDR_T j = indiceRow; j < endIndices; j++) {
            if (sortedIndices[j] != varRow) {
                break;
            }
            currentRow = j;
            if constexpr (withPos) {
                sumRes += updates[pos[j] * totalCol + colIdx];
            } else {
                sumRes += updates[j * totalCol + colIdx];
            }
        }

        bool isCoreLastIndices = currentRow == endIndices - 1;
        if constexpr (!isEndRowCore) {
            if (isCoreLastIndices && varRow == sortedIndices[endIndices]) {         // 判断尾索引与后一个核的头索引是否相同
                flag2 = 1;
            }
        }

        ADDR_T varGmOffset = varRow * totalCol + colIdx;
        U indexWorkspace1 = 0;
        U indexWorkspace2 = 0;
        if (isCoreFirstIndices && isCoreLastIndices) {
            if (flag1 && flag2) {  // 特殊场景，整个核处理的都是相同的索引，且与前后核相邻索引相同
                sumWorkspace[totalCol * TWO * blockIdx + colIdx] = 0;
                sumWorkspace[totalCol * (TWO * blockIdx + 1) + colIdx] = sumRes;
                indexWorkspace1 = varRow;
                indexWorkspace2 = varRow;
            } else if (flag1) {
                sumWorkspace[totalCol * TWO * blockIdx + colIdx] = sumRes;
                indexWorkspace1 = varRow;
                indexWorkspace2 = -1;
            } else if (flag2) {
                sumWorkspace[totalCol * (TWO * blockIdx + 1) + colIdx] = sumRes;
                indexWorkspace1 = -1;
                indexWorkspace2 = varRow;
            } else {
                asc_atomic_add(var + varGmOffset, sumRes);
                indexWorkspace1 = -1;
                indexWorkspace2 = -1;
            }
            if (colIdx == 0) {
                indexWorkspace[TWO * blockIdx] = indexWorkspace1;
                indexWorkspace[TWO * blockIdx + 1] = indexWorkspace2;
            }
        } else if (isCoreFirstIndices) {
            if (flag1) {
                sumWorkspace[totalCol * TWO * blockIdx + colIdx] = sumRes;
                indexWorkspace1 = varRow;
            } else {
                asc_atomic_add(var + varGmOffset, sumRes);
                indexWorkspace1 = -1;
            }
            if (colIdx == 0) {
                indexWorkspace[TWO * blockIdx] = indexWorkspace1;
            }
        } else if (isCoreLastIndices) {
            if (flag2) {
                sumWorkspace[totalCol * (TWO * blockIdx + 1) + colIdx] = sumRes;
                indexWorkspace2 = varRow;
            } else {
                asc_atomic_add(var + varGmOffset, sumRes);
                indexWorkspace2 = -1;
            }
            if (colIdx == 0) {
                indexWorkspace[TWO * blockIdx + 1] = indexWorkspace2;
            }
        } else {
            asc_atomic_add(var + varGmOffset, sumRes);
        }
    }
}

template <typename T, typename U, typename ADDR_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_DETERM) inline void ScatterAddWithSortedSimtDetermStep2(
    uint32_t totalCol, uint32_t blockIdx, uint32_t sumWorkspaceEnd, uint32_t magic, uint32_t shift, __gm__ T* var,
    __gm__ T* sumWorkspace, __gm__ U* indexWorkspace)
{
    if (indexWorkspace[blockIdx * TWO + 1] == indexWorkspace[blockIdx * TWO]) { // 当前核对应的2个workspace索引相等则跳过
        return;
    }
    if (indexWorkspace[blockIdx * TWO + 1] == -1) {
        return;
    }

    T workspaceSumRes = 0;
    uint32_t indicesStartOffset = blockIdx * TWO + 1; // 每个核从对应的第2行开始遍历
    U varRow = indexWorkspace[indicesStartOffset];

    uint32_t dataWorkspaceStart = totalCol * indicesStartOffset;
    for (uint32_t i = dataWorkspaceStart + threadIdx.x; i < sumWorkspaceEnd; i += blockDim.x) {
        uint32_t indiceRow = Simt::UintDiv(i, magic, shift);
        uint32_t colIdx = i - indiceRow * totalCol;
        ADDR_T varGmOffset = varRow * totalCol + colIdx;
        if (indexWorkspace[indiceRow] == varRow) {
            workspaceSumRes += sumWorkspace[i];
        } else {
            asc_atomic_add(var + varGmOffset, workspaceSumRes);
            break;
        }
    }
}

template<typename T, typename U, typename ADDR_T, bool withPos>
class ScatterAddWithSortedDetermSIMT {
public:
    __aicore__ inline ScatterAddWithSortedDetermSIMT(const ScatterAddWithSortedSimtTilingData& tilingData)
        : tilingData_(tilingData){};

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessSimtStep1();
    __aicore__ inline void ProcessSimtStep2();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<U> posGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> sumWorkspace_;
    AscendC::GlobalTensor<U> indexWorkspace_;
    bool isStartRowCore_ = false;
    bool isEndRowCore_ = false;
    const ScatterAddWithSortedSimtTilingData& tilingData_;

    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
    ADDR_T indicesBlockNum_ = 0;
    static constexpr U workspaceIndicesDefault_ = -2;
};

template<typename T, typename U, typename ADDR_T, bool withPos>
__aicore__ inline void ScatterAddWithSortedDetermSIMT<T, U, ADDR_T, withPos>::Init(
                        GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    varGm_.SetGlobalBuffer((__gm__ T *)(var));
    updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
    indicesGm_.SetGlobalBuffer((__gm__ U*)indices);
    posGm_.SetGlobalBuffer((__gm__ U*)pos);

    isStartRowCore_ = blockIdx_ == 0; // 首核
    isEndRowCore_ = blockIdx_ == tilingData_.usedCoreNum - 1; // 尾核

    uint32_t workspaceIndicesOffset = tilingData_.usedCoreNum * TWO * tilingData_.varShape[1] * sizeof(T);
    workspaceIndicesOffset = ops::CeilAlign(workspaceIndicesOffset, static_cast<uint32_t>(sizeof(U)));
    sumWorkspace_.SetGlobalBuffer((__gm__ T*)workspace);
    indexWorkspace_.SetGlobalBuffer((__gm__ U*)(workspace + workspaceIndicesOffset));

    indicesBlockNum_ = tilingData_.normBlockIndices;
    if (blockIdx_ == tilingData_.usedCoreNum - 1) {
        indicesBlockNum_ = tilingData_.tailBlockIndices;
    }
}

template<typename T, typename U, typename ADDR_T, bool withPos>
__aicore__ inline void ScatterAddWithSortedDetermSIMT<T, U, ADDR_T, withPos>::ProcessSimtStep1()
{
    uint32_t blockIdx = blockIdx_;
    ADDR_T totalCol = static_cast<ADDR_T>(tilingData_.varShape[1]);
    ADDR_T varFirstDimSize = static_cast<ADDR_T>(tilingData_.varShape[0]);
    ADDR_T indicesBlockNum = indicesBlockNum_;
    ADDR_T indicesBlockOffset = blockIdx * tilingData_.normBlockIndices;
    ADDR_T magic = 0;
    ADDR_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, totalCol);

    if (isStartRowCore_ && isEndRowCore_) {
        asc_vf_call<ScatterAddWithSortedSimtDetermStep1<T, U, ADDR_T, true, true, withPos>>(dim3(THREAD_NUM_DETERM),
                totalCol, indicesBlockOffset, indicesBlockNum, varFirstDimSize, magic, shift, blockIdx,
                (__gm__ T*)(varGm_.GetPhyAddr()), (__gm__ U*)(indicesGm_.GetPhyAddr()), (__gm__ U*)(posGm_.GetPhyAddr()),
                (__gm__ T*)(updatesGm_.GetPhyAddr()), (__gm__ T*)(sumWorkspace_.GetPhyAddr()), (__gm__ U*)(indexWorkspace_.GetPhyAddr()));
    } else if (isStartRowCore_ && !isEndRowCore_) {
        asc_vf_call<ScatterAddWithSortedSimtDetermStep1<T, U, ADDR_T, true, false, withPos>>(dim3(THREAD_NUM_DETERM),
                totalCol, indicesBlockOffset, indicesBlockNum, varFirstDimSize, magic, shift, blockIdx,
                (__gm__ T*)(varGm_.GetPhyAddr()), (__gm__ U*)(indicesGm_.GetPhyAddr()), (__gm__ U*)(posGm_.GetPhyAddr()),
                (__gm__ T*)(updatesGm_.GetPhyAddr()), (__gm__ T*)(sumWorkspace_.GetPhyAddr()), (__gm__ U*)(indexWorkspace_.GetPhyAddr()));
    } else if (!isStartRowCore_ && isEndRowCore_) {
        asc_vf_call<ScatterAddWithSortedSimtDetermStep1<T, U, ADDR_T, false, true, withPos>>(dim3(THREAD_NUM_DETERM),
                totalCol, indicesBlockOffset, indicesBlockNum, varFirstDimSize, magic, shift, blockIdx,
                (__gm__ T*)(varGm_.GetPhyAddr()), (__gm__ U*)(indicesGm_.GetPhyAddr()), (__gm__ U*)(posGm_.GetPhyAddr()),
                (__gm__ T*)(updatesGm_.GetPhyAddr()), (__gm__ T*)(sumWorkspace_.GetPhyAddr()), (__gm__ U*)(indexWorkspace_.GetPhyAddr()));
    } else {
        asc_vf_call<ScatterAddWithSortedSimtDetermStep1<T, U, ADDR_T, false, false, withPos>>(dim3(THREAD_NUM_DETERM),
                totalCol, indicesBlockOffset, indicesBlockNum, varFirstDimSize, magic, shift, blockIdx,
                (__gm__ T*)(varGm_.GetPhyAddr()), (__gm__ U*)(indicesGm_.GetPhyAddr()), (__gm__ U*)(posGm_.GetPhyAddr()),
                (__gm__ T*)(updatesGm_.GetPhyAddr()), (__gm__ T*)(sumWorkspace_.GetPhyAddr()), (__gm__ U*)(indexWorkspace_.GetPhyAddr()));
    }
}

template<typename T, typename U, typename ADDR_T, bool withPos>
__aicore__ inline void ScatterAddWithSortedDetermSIMT<T, U, ADDR_T, withPos>::ProcessSimtStep2()
{
    uint32_t blockIdx = blockIdx_;
    uint32_t totalCol = static_cast<uint32_t>(tilingData_.varShape[1]);
    uint32_t sumWorkspaceEnd = blockNum_ * TWO * totalCol;
    uint32_t magic = 0;
    uint32_t shift = 0;
    GetUintDivMagicAndShift(magic, shift, totalCol);

    asc_vf_call<ScatterAddWithSortedSimtDetermStep2<T, U, ADDR_T>>(dim3(totalCol),
            totalCol, blockIdx, sumWorkspaceEnd, magic, shift, (__gm__ T*)(varGm_.GetPhyAddr()),
            (__gm__ T*)(sumWorkspace_.GetPhyAddr()), (__gm__ U*)(indexWorkspace_.GetPhyAddr()));
}

template<typename T, typename U, typename ADDR_T, bool withPos>
__aicore__ inline void ScatterAddWithSortedDetermSIMT<T, U, ADDR_T, withPos>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    ProcessSimtStep1();
    SyncAll();
    ProcessSimtStep2();
}

}
#endif