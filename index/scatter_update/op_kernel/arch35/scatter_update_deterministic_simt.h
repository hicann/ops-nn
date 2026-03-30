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
 * \file scatter_update_deterministic_simt.h
 * \brief scatter_update
 */
#ifndef SCATTER_UPDATE_DETERMINISTIC_SIMT_H
#define SCATTER_UPDATE_DETERMINISTIC_SIMT_H

#include "scatter_update_common.h"
#include "scatter_update_struct.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"

namespace ScatterUpdate {
using namespace AscendC;

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
class ScatterUpdateDeterministicSimt : public ScatterUpdateDeterministicCommon<T, U, MASK_T, splitCol, CAST_T, castType> {
public:
    __aicore__ inline ScatterUpdateDeterministicSimt(const ScatterUpdateTilingData& tilingData, TPipe& pipe) : 
        ScatterUpdateDeterministicCommon<T, U, MASK_T, splitCol, CAST_T, castType> (tilingData, pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessSplitRow();
    __aicore__ inline void CopyInIndices(uint64_t indicesGmOffset, uint32_t indicesCount);
};

template<typename T, typename U, typename MASK_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_DETERMINISTIC) inline void ScatterUpdateSimtCompute(
    uint64_t varFirstDimSize, uint64_t updatesBlockCount, uint64_t indicesBlockOffset, uint64_t updatesBlockOffset,
    uint64_t varStride, uint32_t totalCol, uint64_t magic, uint64_t shift, __gm__ MASK_T* workspaceMaskAddr,
    __gm__ U* indicesAddr, __gm__ T* varAddr, __gm__ T* updatesAddr)
{
    for (uint64_t i = Simt::GetThreadIdx(); i < updatesBlockCount; i += Simt::GetThreadNum()) {
        uint64_t indicesBlockRow = Simt::UintDiv(i, magic, shift);        // 当前线程对应的indices行
        U indicesValue = indicesAddr[indicesBlockOffset + indicesBlockRow];
        if (indicesValue < 0 || indicesValue >= varFirstDimSize) {
            continue;
        }
        if (workspaceMaskAddr[indicesValue] != indicesBlockOffset + indicesBlockRow) {
            continue;
        }

        uint64_t varColOffset = i - indicesBlockRow * totalCol;
        uint64_t varGmOffset = indicesValue * varStride + varColOffset;
        varAddr[varGmOffset] = updatesAddr[updatesBlockOffset + i];
    }
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimt<T, U, MASK_T, splitCol, CAST_T, castType>::Init(
            GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    this->InitBase(var, indices, updates);
    this->InitSetBuffer(var, indices, updates, workspace);
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimt<T, U, MASK_T, splitCol, CAST_T, castType>::ProcessSplitRow()
{
    uint64_t varFirstDimSize = this->tilingData_.varShape[0];
    uint32_t totalCol = static_cast<uint32_t>(this->tilingData_.varShape[1]);  // 尾轴较大时走到切分列的分支，此处只处理尾轴较小场景。所以使用uint32即可
    uint64_t varStride = this->tilingData_.varStride;
    uint64_t indicesBlockCount = this->blockIdx_ == this->tilingData_.usedCoreNum - 1 ? this->tilingData_.tailBlockIndices : 
                                                                            this->tilingData_.normBlockIndices;
    uint64_t updatesBlockCount = indicesBlockCount * totalCol;
    uint64_t indicesBlockOffset = this->blockIdx_ * this->tilingData_.normBlockIndices;
    uint64_t updatesBlockOffset = indicesBlockOffset * totalCol;
    uint64_t magic = 0;
    uint64_t shift = 0;
    GetUintDivMagicAndShift(magic, shift, static_cast<uint64_t>(totalCol));

    Simt::VF_CALL<ScatterUpdateSimtCompute<T, U, MASK_T>>(Simt::Dim3(THREAD_NUM_DETERMINISTIC), 
            varFirstDimSize, updatesBlockCount, indicesBlockOffset, updatesBlockOffset, varStride, totalCol, magic, shift,
            (__gm__ MASK_T*)(this->workspaceMask_.GetPhyAddr()), (__gm__ U*)(this->indicesGm_.GetPhyAddr()),
            (__gm__ T*)(this->varGm_.GetPhyAddr()), (__gm__ T*)(this->updatesGm_.GetPhyAddr()));
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicSimt<T, U, MASK_T, splitCol, CAST_T, castType>::Process()
{
    if (GetBlockIdx() >= this->tilingData_.usedCoreNum) {
        return;
    }

    this->CalcMask();
    SyncAll();
    ProcessSplitRow();
}

}
#endif  // SCATTER_UPDATE_DETERMINISTIC_SIMT_H