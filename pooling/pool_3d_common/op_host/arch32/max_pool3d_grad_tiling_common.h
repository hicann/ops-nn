/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_grad_tiling_common.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_SCATTER_TILING_TEMPLATE_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_SCATTER_TILING_TEMPLATE_H
#include <cstdint>
#include "log/log.h"
#include "util/math_util.h"
#include "max_pool3d_grad_tiling_constants.h"

namespace optiling {

// 1. All Tensor full size, cut nc between cores, without cut in one core
template<typename ParamsType>
inline bool tryNoCut(
    ParamsType& params,
    uint64_t ncPreCore,
    uint64_t doDim,
    uint64_t hoDim,
    uint64_t woDim,
    uint64_t xDtypeSize,
    uint64_t indexDtypeSize,
    uint64_t effectiveUbSize)
{
    uint64_t noCutSize = ncPreCore * doDim * hoDim * woDim * (xDtypeSize + indexDtypeSize);
    if (noCutSize <= effectiveUbSize) {
        params.baseNc = ncPreCore;
        params.baseDo = doDim;
        params.baseHo = hoDim;
        params.baseWo = woDim;
        params.ubCutAxis = TILING_UB_NO_CUT;
        return true;
    }
    return false;
}

// 2. Cut nc
template<typename ParamsType>
inline bool tryCutNc(
    ParamsType& params,
    uint64_t doDim,
    uint64_t hoDim,
    uint64_t woDim,
    uint64_t xDtypeSize,
    uint64_t indexDtypeSize,
    uint64_t effectiveUbSize,
    uint64_t maxBlockCount)
{
    uint64_t perNcSize = 1UL * doDim * hoDim * woDim * (xDtypeSize + indexDtypeSize);
    if (perNcSize <= effectiveUbSize) {
        uint64_t baseNc = effectiveUbSize / perNcSize;
        if (baseNc > maxBlockCount) {
            baseNc = maxBlockCount;
        }
        params.baseNc = baseNc;
        params.baseDo = doDim;
        params.baseHo = hoDim;
        params.baseWo = woDim;
        params.ubCutAxis = TILING_UB_CUT_NC;
        return true;
    }
    params.baseNc = 1UL;
    return false;
}

// 3. Cut do
template<typename ParamsType>
inline bool tryCutDo(
    ParamsType& params,
    uint64_t hoDim,
    uint64_t woDim,
    uint64_t xDtypeSize,
    uint64_t indexDtypeSize,
    uint64_t effectiveUbSize)
{
    uint64_t perDoSize = 1UL * 1UL * hoDim * woDim * (xDtypeSize + indexDtypeSize);
    if (perDoSize <= effectiveUbSize) {
        params.baseDo = effectiveUbSize / perDoSize;
        params.baseHo = hoDim;
        params.baseWo = woDim;
        params.ubCutAxis = TILING_UB_CUT_DO;
        return true;
    }
    params.baseDo = 1UL;
    return false;
}

// 4. Cut ho
template<typename ParamsType>
inline bool tryCutHo(
    ParamsType& params,
    uint64_t woDim,
    uint64_t xDtypeSize,
    uint64_t indexDtypeSize,
    uint64_t effectiveUbSize)
{
    uint64_t perHoSize = 1UL * 1UL * 1UL * woDim * (xDtypeSize + indexDtypeSize);
    if (perHoSize <= effectiveUbSize) {
        params.baseHo = effectiveUbSize / perHoSize;
        params.baseWo = woDim;
        params.ubCutAxis = TILING_UB_CUT_HO;
        return true;
    }
    params.baseHo = 1UL;
    return false;
}

// 5. Cut wo
template<typename ParamsType>
inline bool tryCutWo(
    ParamsType& params,
    uint64_t xDtypeSize,
    uint64_t indexDtypeSize,
    uint64_t effectiveUbSize)
{
    uint64_t perWoSize = 1UL * 1UL * 1UL * 1UL * (xDtypeSize + indexDtypeSize);
    if (perWoSize <= effectiveUbSize) {
        params.baseWo = effectiveUbSize / perWoSize;
        params.ubCutAxis = TILING_UB_CUT_WO;
        return true;
    }
    params.baseWo = 1UL;
    return false;
}

template<typename ParamsType>
inline bool CalculateScatterTilingParams(
    ParamsType& params,
    uint64_t doDim,
    uint64_t hoDim,
    uint64_t woDim,
    uint64_t xDtypeSize,
    uint64_t indexDtypeSize,
    uint64_t maxBlockCount,
    uint64_t blockAdjustment = 0)
{
    uint64_t effectiveUbSize = params.maxUbSize;
    if (blockAdjustment > 0 && params.maxUbSize > blockAdjustment) {
        effectiveUbSize = params.maxUbSize - blockAdjustment;
    }
    
    uint64_t ncPreCore = Ops::Base::CeilDiv(params.ncDim, params.totalCoreNum);
    params.usedCoreNum = Ops::Base::CeilDiv(params.ncDim, ncPreCore);

    // 尝试五种切分场景
    if (tryNoCut(params, ncPreCore, doDim, hoDim, woDim, 
                 xDtypeSize, indexDtypeSize, effectiveUbSize)) {
        return true;
    }
    
    if (tryCutNc(params, doDim, hoDim, woDim, xDtypeSize, 
                 indexDtypeSize, effectiveUbSize, maxBlockCount)) {
        return true;
    }
    
    if (tryCutDo(params, hoDim, woDim, xDtypeSize, 
                 indexDtypeSize, effectiveUbSize)) {
        return true;
    }
    
    if (tryCutHo(params, woDim, xDtypeSize, indexDtypeSize, 
                 effectiveUbSize)) {
        return true;
    }
    
    if (tryCutWo(params, xDtypeSize, indexDtypeSize, 
                 effectiveUbSize)) {
        return true;
    }
    
    return false;
}
template<typename ParamsType>
inline void CalculateRoundParams(
    ParamsType& params,
    bool isOverLap,
    uint64_t diDim,
    uint64_t hiDim,
    uint64_t wiDim)
{
    params.ncRound = Ops::Base::CeilDiv(params.ncCnt, params.usedCoreNum);
    params.preCoreNum = params.ncCnt % params.usedCoreNum;
    params.ncRoundTail = params.preCoreNum == 0UL ? 
                       params.ncRound : params.ncRound - 1UL;
    params.totalRound = params.ncRound * params.doCnt * 
                       params.hoCnt * params.woCnt;
    
    if (params.xDtypeSize != DTYPE_LEN_B32 && isOverLap) {
        params.workspaceSize = params.ncDim * diDim * 
                             hiDim * wiDim * sizeof(float);
    } else {
        params.workspaceSize = 0UL;
    }
}

template<typename TilingDataType, typename ParamsType>
inline void SetScatterTilingDataCommon(
    TilingDataType& tilingData,
    ParamsType& params)
{
    tilingData.set_ncRound(params.ncRound);
    tilingData.set_ncRoundTail(params.ncRoundTail);
    tilingData.set_totalRound(params.totalRound);
    tilingData.set_preCoreNum(params.preCoreNum);
}

template<typename TilingDataType>
inline void PrintScatterTilingDataCommon(
    const char* contextName,
    TilingDataType& tilingData)
{
    OP_LOGI(contextName, 
            "TilingData ncRound: %lu, ncRoundTail: %lu, totalRound: %lu.",
            tilingData.get_ncRound(), 
            tilingData.get_ncRoundTail(), 
            tilingData.get_totalRound());
}

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_MAX_POOL3D_GRAD_SCATTER_TILING_TEMPLATE_H
