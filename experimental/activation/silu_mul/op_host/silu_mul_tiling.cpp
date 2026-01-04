/*
 * Copyright (c) 2025 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file silu_mul_tiling.cpp
 * \brief silu_mul_tiling source file
 */
#include "silu_mul_tiling.h"
#include <vector>
#include <iostream>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

static constexpr int32_t UB_SIZE = 184 * 1024;
static constexpr int32_t ONE_BLOCK_SIZE = 32;
static constexpr int32_t CALC_BUF_NUM = 8;
static constexpr int32_t HALF_SIZE = 2;
static constexpr int32_t BF16_SIZE = 2;
static constexpr int32_t SIZE_2 = 2;
// static constexpr int32_t TILING_KEY_HALF = 1;
// static constexpr int32_t TILING_KEY_FLOAT = 2;
// static constexpr int32_t TILING_KEY_BFLOAT16 = 3;

static constexpr int32_t SIZE_16 = 16;
static constexpr int32_t LENGTH_1024 = 1024;
static constexpr int32_t LENGTH_LIMIT = 200000;

class SiluMulTiling {
public:
    explicit SiluMulTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunBigKernelTiling();
    ge::graphStatus FillTilingKey();

private:
    ge::graphStatus ShapeCheck();
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    SiluMulTilingData tilingData;
    int32_t batchSize = 0;
    int32_t inputShapeSize = 0;
    int32_t lastDimSize = 0;
    int32_t oneBlockNum = 0;
    int32_t PPMaxCalNum = 0;
    const int32_t workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    static inline int32_t CeilA2B(const int32_t a, const int32_t b)
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    int32_t GetNeedCoreNum(const int32_t coreNumPlatform)
    {
        int32_t needCoreNum = 1;
        if (lastDimSize / SIZE_2 > PPMaxCalNum) {
            needCoreNum = batchSize;
        } else {
            const int32_t d = lastDimSize / SIZE_2;
            auto dAlign = (d + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
            const int32_t n = PPMaxCalNum / dAlign;
            needCoreNum = CeilA2B(batchSize, n);
        }
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        if (needCoreNum >= coreNumPlatform) {
            return coreNumPlatform;
        } else {
            return needCoreNum;
        }
    }
};

ge::graphStatus SiluMulTiling::ShapeCheck()
{
    OP_CHECK_IF(
        (lastDimSize > LENGTH_1024),
        OP_LOGE(tilingContext->GetNodeName(), "Last dim size should be no more than 1024."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (lastDimSize % SIZE_2 == 1), OP_LOGE(tilingContext->GetNodeName(), "Last dim size should be even."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (batchSize > LENGTH_LIMIT),
        OP_LOGE(tilingContext->GetNodeName(), "Batch dim size should be no more than 200000."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SiluMulTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, srcTensor);

    PPMaxCalNum = UB_SIZE / CALC_BUF_NUM / static_cast<int32_t>(sizeof(float));

    FillTilingKey();

    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();
    size_t inputShapeDim = inputShape.GetDimNum();
    OP_CHECK_IF(
        (inputShapeDim < static_cast<size_t>(SIZE_2)),
        OP_LOGE(tilingContext->GetNodeName(), "Input shape dim should be no less than 2."), return ge::GRAPH_FAILED);
    lastDimSize = inputShape.GetDim(inputShapeDim - 1);
    inputShapeSize = inputShape.GetShapeSize();

    if (lastDimSize == 0) {
        OP_LOGE(tilingContext->GetNodeName(),
            "Last dim elements can not be zero.");
        return ge::GRAPH_FAILED;
    }

    batchSize = inputShapeSize / lastDimSize;

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    int32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(workspaceSize_);
    OP_CHECK_IF(
        (ShapeCheck() == ge::GRAPH_FAILED), OP_LOGE(tilingContext->GetNodeName(), "ShapeCheck failed!"),
        return ge::GRAPH_FAILED);
    tilingData.set_lastDimSize(lastDimSize);
    tilingData.set_batchSize(batchSize);
    tilingData.set_PPMaxCalNum(PPMaxCalNum);
    tilingData.set_needCoreNum(needCoreNum);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    tilingContext->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SiluMulTiling::FillTilingKey()
{
    auto temp = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, temp);
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    if (dataType == ge::DT_FLOAT16) {
        oneBlockNum = ONE_BLOCK_SIZE / HALF_SIZE;
    } else if (dataType == ge::DT_FLOAT) {
        oneBlockNum = ONE_BLOCK_SIZE / static_cast<int32_t>(sizeof(float));
    } else if (dataType == ge::DT_BF16) {
        oneBlockNum = ONE_BLOCK_SIZE / BF16_SIZE;
    } else {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4SiluMulTiling([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingSiluMulTiling(gert::TilingContext* context)
{
    SiluMulTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(SiluMul).Tiling(TilingSiluMulTiling).TilingParse<SiluMulCompileInfo>(TilingPrepare4SiluMulTiling);
} // namespace optiling