/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flat_quant_tiling.cpp
 * \brief
 */
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "flat_quant_tiling.h"

namespace optiling {
constexpr uint8_t INPUT_TENSOR_NUM = 3;
constexpr uint8_t INDEX_ZERO = 0;
constexpr uint8_t INDEX_ONE = 1;
constexpr uint8_t INDEX_TWO = 2;

constexpr uint8_t DIM_THREE = 3;
constexpr uint8_t DIM_TWO = 2;

constexpr uint8_t HALF_TYPE = 1;
constexpr uint8_t BFLOAT_TYPE = 2;
constexpr uint8_t BYTE_LEN_2 = 2;
constexpr uint8_t CEIL_SIZE = 16;
constexpr int32_t MAX_K_SIZE = 32768;
constexpr int32_t MAX_MN_SIZE = 128;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

class FlatQuantTiling {
public:
    explicit FlatQuantTiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    bool CheckShapes() const;
    bool CheckClipRatio() const;
    uint8_t GetDataTypeVal() const;

    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1;

private:
    ge::DataType dataType = ge::DT_UNDEFINED;

    gert::TilingContext *tilingContext = nullptr;
    gert::Shape xShape;
    gert::Shape p1Shape;
    gert::Shape p2Shape;

    const float *clipRatio = nullptr;
    FlatQuantTilingData tilingData;
};

ge::graphStatus FlatQuantTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto xTensor = tilingContext->GetInputTensor(INDEX_ZERO);
    auto p1Tensor = tilingContext->GetInputTensor(INDEX_ONE);
    auto p2Tensor = tilingContext->GetInputTensor(INDEX_TWO);
    const gert::RuntimeAttrs *attrs = tilingContext->GetAttrs();
    if (xTensor == nullptr || p1Tensor == nullptr || p2Tensor == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的数据类型，输入Tensor的数据类型保持一致
    for (int i = 0; i < INPUT_TENSOR_NUM; i++) {
        auto temp = tilingContext->GetInputDesc(i);
        if (dataType == ge::DT_UNDEFINED) {
            dataType = temp->GetDataType();
        } else if (dataType != temp->GetDataType()) {
            return ge::GRAPH_FAILED;
        }
    }
    tilingData.set_dataType(GetDataTypeVal());

    // 获取输入的shape
    xShape = tilingContext->GetInputShape(INDEX_ZERO)->GetOriginShape();
    p1Shape = tilingContext->GetInputShape(INDEX_ONE)->GetOriginShape();
    p2Shape = tilingContext->GetInputShape(INDEX_TWO)->GetOriginShape();
    clipRatio = attrs->GetAttrPointer<float>(0);
    if (!CheckShapes() || !CheckClipRatio()) {
        return ge::GRAPH_FAILED;
    }
    tilingData.set_K(xShape.GetDim(INDEX_ZERO));
    tilingData.set_M(xShape.GetDim(INDEX_ONE));
    tilingData.set_N(xShape.GetDim(INDEX_TWO));
    tilingData.set_clipRatio(*clipRatio);

    int64_t K = xShape.GetDim(INDEX_ZERO);
    int64_t Mceil = CeilA2B(xShape.GetDim(INDEX_ONE), CEIL_SIZE) * CEIL_SIZE;
    int64_t Nceil = CeilA2B(xShape.GetDim(INDEX_TWO), CEIL_SIZE) * CEIL_SIZE;
    size_t *workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = (K * Mceil * Nceil + K * Mceil * Nceil + Mceil * Mceil + Nceil * Nceil) * BYTE_LEN_2 + WORK_SPACE_SIZE;

    auto compileInfo = tilingContext->GetCompileInfo<FlatQuantCompileInfo>();
    tilingContext->SetBlockDim(compileInfo->coreNum);
    tilingContext->SetTilingKey(1);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

bool FlatQuantTiling::CheckShapes() const
{
    if (xShape.GetDimNum() != DIM_THREE || p1Shape.GetDimNum() != DIM_TWO || p2Shape.GetDimNum() != DIM_TWO) {
        return false;
    }
    int64_t K = xShape.GetDim(INDEX_ZERO);
    int64_t M = xShape.GetDim(INDEX_ONE);
    int64_t N = xShape.GetDim(INDEX_TWO);
    if (K > MAX_K_SIZE || M > MAX_MN_SIZE || N > MAX_MN_SIZE) {
        return false;
    }
    if(p1Shape.GetDim(INDEX_ZERO) != M || p1Shape.GetDim(INDEX_ONE) != M || p2Shape.GetDim(INDEX_ZERO) != N || p2Shape.GetDim(INDEX_ONE) != N) {
        return false;
    }
    return true;
}

bool FlatQuantTiling::CheckClipRatio() const
{
    return clipRatio != nullptr && *clipRatio > ZERO_FLOAT && *clipRatio <= ONE_FLOAT;
}

uint8_t FlatQuantTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return HALF_TYPE;
        case ge::DT_BF16:
            return BFLOAT_TYPE;
        default:
            return 0;
    }
}

template <typename T1, typename T2>
inline auto FlatQuantTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

static ge::graphStatus Tiling4FlatQuantTiling(gert::TilingContext *context)
{
    FlatQuantTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext *context)
{
    auto compileInfo =context->GetCompiledInfo<FlatQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(compileInfo->coreNum <= 0,
        OP_LOGE(context->GetNodeName(), "FlatQuanrtTiling GetHardwareInfo Failed"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FlatQuant).Tiling(Tiling4FlatQuantTiling).TilingParse<FlatQuantCompileInfo>(TilingPrepareTiling);
}  // namespace optiling
