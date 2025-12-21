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
 * \file heaviside_tiling.cpp
 * \brief heaviside_tiling source file
 */
#include "heaviside_tiling.h"
#include <vector>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {

constexpr int32_t MAX_ELEMENT_NUM_EACH_CORE = 8 * 1024;

constexpr uint64_t TILING_KEY_DEFAULT = 1;

const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;

class HeavisideTiling {
public:
    explicit HeavisideTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunBigKernelTiling();

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    gert::Shape valuesShape;
    HeavisideTilingData tilingData;

    int64_t inputShapeSize = 0;
    int64_t valuesShapeSize = 0;
    bool valuesType = false;

    const int32_t workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    inline int64_t CeilA2B(const int64_t a, const int64_t b) const
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    int32_t GetNeedCoreNum(const int32_t coreNumPlatform) const
    {
        int64_t needCoreNum = CeilA2B(inputShapeSize, MAX_ELEMENT_NUM_EACH_CORE);
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        if (needCoreNum >= coreNumPlatform) {
            return coreNumPlatform;
        } else {
            return needCoreNum;
        }
    }

    bool GetValuesType()
    {
        if (valuesShapeSize == 1) {
            valuesType = true;
        } else if (valuesShapeSize > 0 && valuesShape == inputShape) {
            valuesType = false;
        } else {
            return false;
        }
        return true;
    }
};

ge::graphStatus HeavisideTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto valuesTensor = tilingContext->GetInputTensor(1);
    if (valuesTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = tilingContext->GetInputDesc(0)->GetDataType();

    tilingContext->SetTilingKey(TILING_KEY_DEFAULT);

    // 获取input和values的shape
    auto srcShape0 = tilingContext->GetInputShape(0);
    auto srcShape1 = tilingContext->GetInputShape(1);
    inputShape = srcShape0->GetOriginShape();
    inputShapeSize = inputShape.GetShapeSize();
    valuesShape = srcShape1->GetOriginShape();
    valuesShapeSize = valuesShape.GetShapeSize();
    if (GetValuesType() == false) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;

    tilingData.set_elementNum(inputShapeSize);
    tilingData.set_valuesType(valuesType);
    tilingData.set_needCoreNum(needCoreNum);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    tilingContext->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4HeavisideTiling(gert::TilingParseContext* /*context*/)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingHeavisideTiling(gert::TilingContext* context)
{
    HeavisideTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(Heaviside)
    .Tiling(TilingHeavisideTiling)
    .TilingParse<HeavisideCompileInfo>(TilingPrepare4HeavisideTiling);
} // namespace optiling