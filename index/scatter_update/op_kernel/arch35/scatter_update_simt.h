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
 * \file scatter_update_simt.h
 * \brief scatter_update
 */
#ifndef ASCENDC_SCATTER_UPDATE_SIMT_H_
#define ASCENDC_SCATTER_UPDATE_SIMT_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "scatter_update_struct.h"

namespace ScatterUpdate
{
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 256;
#else
constexpr uint32_t THREAD_NUM = 2048;
#endif


template <typename IDX_T, typename VAR_T, typename ADDR_T, bool isUpdateScalar>
class ScatterUpdateSimt
{
public:
    __aicore__ inline ScatterUpdateSimt(const ScatterUpdateTilingData& tilingData) : td_(tilingData){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;

    const ScatterUpdateTilingData& td_;

    ADDR_T blockIdx_{0};
    ADDR_T blockNum_{0};
};

template <typename IDX_T, typename VAR_T, typename ADDR_T, bool isUpdateScalar>
__aicore__ inline void ScatterUpdateSimt<IDX_T, VAR_T, ADDR_T, isUpdateScalar>::Init(GM_ADDR var, GM_ADDR indices,
                                                                                     GM_ADDR updates, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    var_.SetGlobalBuffer((__gm__ VAR_T*)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T*)(updates));
}

template <typename IDX_T, typename VAR_T, typename ADDR_T, bool isUpdateScalar>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterUpdateSimtCompute(
    ADDR_T totalCol, ADDR_T indicesSize, ADDR_T varFirstDimSize, ADDR_T magic, ADDR_T shift, __gm__ VAR_T* var,
    __gm__ IDX_T* indices, __gm__ VAR_T* updates, ADDR_T blockIdx, ADDR_T blockNum, ADDR_T varStride)
{
    ADDR_T totalSize = indicesSize * totalCol;
    VAR_T updateScalarValue = static_cast<VAR_T>(updates[0]);

    for (ADDR_T i = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < totalSize;
         i += blockNum * Simt::GetThreadNum()) {
        ADDR_T indiceRow = Simt::UintDiv(i, magic, shift);        // 当前线程对应indices行
        ADDR_T varRow = static_cast<ADDR_T>(indices[indiceRow]);  // 通过indices索引确定var对应行
        ADDR_T tailRowIdx = i - indiceRow * totalCol;             // 获取当前线程对应updates中的数，在当前行中的索引
        ADDR_T varDataIdx = varRow * varStride + tailRowIdx;       // 获取当前线程对应要更新的var中的数的总索引
        if (!(varRow >= 0 && varRow < varFirstDimSize)) {
            continue;
        }
        if constexpr (isUpdateScalar) {
            var[varDataIdx] = updateScalarValue;
        } else {
            var[varDataIdx] = updates[i];
        }
    }
}

template <typename IDX_T, typename VAR_T, typename ADDR_T, bool isUpdateScalar>
__aicore__ inline void ScatterUpdateSimt<IDX_T, VAR_T, ADDR_T, isUpdateScalar>::Process()
{
    ADDR_T totalCol = static_cast<ADDR_T>(td_.varShape[1]);
    ADDR_T indicesSize = static_cast<ADDR_T>(td_.indicesSize);
    ADDR_T varFirstDimSize = static_cast<ADDR_T>(td_.varShape[0]);
    ADDR_T varStride = static_cast<ADDR_T>(td_.varStride);
    ADDR_T magic = 0;
    ADDR_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, totalCol);

    Simt::VF_CALL<ScatterUpdateSimtCompute<IDX_T, VAR_T, ADDR_T, isUpdateScalar>>(Simt::Dim3(THREAD_NUM), 
            totalCol, indicesSize, varFirstDimSize, magic, shift, (__gm__ VAR_T*)(var_.GetPhyAddr()),
            (__gm__ IDX_T*)(indices_.GetPhyAddr()), (__gm__ VAR_T*)(updates_.GetPhyAddr()), blockIdx_, blockNum_, varStride);
}
}  // namespace ScatterUpdate

#endif