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
 * \file common_tiling.h
 * \brief
 */
#ifndef COMMON_TILING_H
#define COMMON_TILING_H

namespace optiling {
#define UINT_MAX  (__INT_MAX__  *2U +1U)

constexpr int64_t EPSILON_ATTR_INDEX = 0;
constexpr int64_t NORM_MODE_ATTR_INDEX = 0;     // 预留

struct Tiling4LayerNormQuantCompileInfo {
  uint32_t coreNum;
  uint64_t ubSize;
  uint32_t sysWorkspaceSize;
  bool isRegbase = false;
};

template <typename T>
auto CeilDiv(T dividend, T divisor) -> T
{
    return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
}

template <typename T>
inline __attribute__((always_inline)) ge::graphStatus PostLayerNormPtrFunc(T* tilingDataPtr,
                                                                           NormTilingDataPtrCon& ptrCon,
                                                                           gert::TilingContext* context)
{
    OP_CHECK_IF(tilingDataPtr == nullptr,
                    OP_LOGE(context->GetNodeName(), "tilingDataPtr should not be empty"),
                    return ge::GRAPH_FAILED);

    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const Tiling4LayerNormQuantCompileInfo*>(context->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
        ptrCon.maxUbSize = compileInfoPtr->ubSize;
        ptrCon.maxCoreNum = compileInfoPtr->coreNum;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        ptrCon.maxCoreNum = ascendcPlatform.GetCoreNum();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ptrCon.maxUbSize);
    }

    ptrCon.maxEleFp16 = ptrCon.maxUbSize / 2;  // buffer * 2
    int64_t tmpNumRow = 1;
    auto x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto dimsSize = x_shape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(dimsSize <= 0 || dimsSize > 8,
                    OP_LOGE(context->GetNodeName(), "dimsSize is invalid!"),
                    return ge::GRAPH_FAILED);
    for (size_t i = 0; i < dimsSize - 1; i++) {
        OP_CHECK_IF(
            x_shape->GetStorageShape().GetDim(i) <= 0 || tmpNumRow > UINT_MAX / x_shape->GetStorageShape().GetDim(i),
            OP_LOGE(context->GetNodeName(), "tmpNumRow is invalid!"), return ge::GRAPH_FAILED);
        tmpNumRow *= x_shape->GetStorageShape().GetDim(i);
    }

    ptrCon.numRow = static_cast<uint32_t>(tmpNumRow);
    tilingDataPtr->set_numFirstDim(ptrCon.numRow);
    OP_CHECK_IF(x_shape->GetStorageShape().GetDim(dimsSize - 1) <= 0 ||
                        x_shape->GetStorageShape().GetDim(dimsSize - 1) > UINT_MAX,
                    OP_LOGE(context->GetNodeName(), "numCol is invalid!"),
                    return ge::GRAPH_FAILED);
    ptrCon.numCol = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(dimsSize - 1));

    tilingDataPtr->set_numLastDim(ptrCon.numCol);
    ptrCon.numCore = CeilDiv(ptrCon.numRow, CeilDiv(ptrCon.numRow, ptrCon.maxCoreNum));
    tilingDataPtr->set_numCore(ptrCon.numCore);
    context->SetBlockDim(ptrCon.numCore);
    ptrCon.rowWork = CeilDiv(ptrCon.numRow, ptrCon.numCore);
    tilingDataPtr->set_nlFirstdimPerCore(ptrCon.rowWork);
    ptrCon.nlFirstdimPerCoreNum = tilingDataPtr->get_nlFirstdimPerCore();
    tilingDataPtr->set_lFirstdimPerCore(ptrCon.numRow - ptrCon.nlFirstdimPerCoreNum * (ptrCon.numCore - 1));
    return ge::GRAPH_SUCCESS;
}

template <typename T>
inline __attribute__((always_inline)) ge::graphStatus CheckSplit(T* tilingDataPtr, const int32_t& totalMemNeed,
                                                                 const int32_t& sumData,
                                                                 const NormTilingDataPtrCon& ptrCon,
                                                                 gert::TilingContext* context)
{
    OP_CHECK_IF(tilingDataPtr == nullptr,
                    OP_LOGE(context->GetNodeName(), "tilingDataPtr should not be empty"),
                    return ge::GRAPH_FAILED);
    if (totalMemNeed > sumData) {
        OP_CHECK_IF(sumData <= 0, OP_LOGE(context->GetNodeName(), "sumData is invalid!"),
                        return ge::GRAPH_FAILED);
        uint32_t timeCopyIn = static_cast<uint32_t>(CeilDiv(totalMemNeed, sumData));
        tilingDataPtr->set_firstDimPerTimes(
            (ptrCon.nlFirstdimPerCoreNum / timeCopyIn == 0) ? 1 : ptrCon.nlFirstdimPerCoreNum / timeCopyIn);
    } else {
        tilingDataPtr->set_firstDimPerTimes(ptrCon.nlFirstdimPerCoreNum);
    }
    OP_CHECK_IF(tilingDataPtr->get_firstDimPerTimes() >= UINT_MAX - tilingDataPtr->get_nlFirstdimPerCore(),
                    OP_LOGE(context->GetNodeName(), "row_work + row_step is invalid!"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
}
#endif
