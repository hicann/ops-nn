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
 * \file logit_tiling.cpp
 * \brief
 */
#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "logit_tiling.h"

namespace optiling {

constexpr int64_t MAX_ELEMENT_NUM_EACH_CORE = 8 * 1024;

constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_BFLOAT16 = 3;

const static int64_t SIZE_16 = 16;
const static int64_t LENGTH_1024 = 1024;

// SINGLE BUFFER
#define UB_NUM_INT16_ONE 8U
#define UB_NUM_INT8_UINT8_ONE 14U
// DOUBLE BUFFER
#define UB_NUM_INT16_TWO 12U
#define UB_NUM_INT8_UINT8_TWO 20U
#define BLOCK_SIZE 256U

class LogitTiling {
public:
    explicit LogitTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunBigKernelTiling(gert::TilingContext* context);

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    LogitTilingData tilingData;

    int64_t inputShapeSize = 0;

    const int64_t workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    int64_t GetNeedCoreNum(const int64_t coreNumPlatform)
    {
        int64_t needCoreNum = Ops::Base::CeilDiv(inputShapeSize, MAX_ELEMENT_NUM_EACH_CORE);
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

ge::graphStatus LogitTiling::RunBigKernelTiling(gert::TilingContext* context)
{
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的参数
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    float epsilon = *(attrs->GetFloat(0));

    if (epsilon <= 0) {
        epsilon = 1e-6;
    }
    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = tilingContext->GetInputDesc(0)->GetDataType();

    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = TILING_KEY_HALF;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = TILING_KEY_FLOAT;
    } else if (dataType == ge::DT_BF16) {
        tilingKey = TILING_KEY_BFLOAT16;
    } else if (dataType == ge::DT_INT16) {
        tilingKey = 10;
    } else if (dataType == ge::DT_INT8) {
        tilingKey = 11;
    } else if (dataType == ge::DT_UINT8) {
        tilingKey = 12;
    } else {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(tilingKey);

    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();
    inputShapeSize = inputShape.GetShapeSize();

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint64_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;

    tilingData.set_elementNum(inputShapeSize);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.set_eps(epsilon);

    // 增加int16、int8、uint8处理分支
    uint64_t coreNum = platformInfo.GetCoreNumAiv();
    uint64_t ubSize = 0U; // init ubSize
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint64_t inputNum = tilingContext->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 2U;
    if (tilingContext->GetInputDesc(0)->GetDataType() == ge::DT_INT8 ||
        tilingContext->GetInputDesc(0)->GetDataType() == ge::DT_UINT8) {
        typeLength = 1U;
    }

    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == inputNum) {
        OP_LOGE(context, "BLOCK_SIZE or coreNum or inputNum is 0");
        return ge::GRAPH_FAILED;
    }

    uint64_t inputLength = typeLength * inputNum;
    uint64_t inputBytes = inputLength / inputNum;
    uint64_t inputLengthAlgin = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    uint64_t ubDataNumber = 0U;
    // bufferOpen = 1 is OPEN DOUBLE BUFFER
    uint64_t bufferOpen = 1;
    if (tilingContext->GetInputDesc(0)->GetDataType() == ge::DT_INT16) {
        if ((inputLengthAlgin < coreNum * (((ubSize / BLOCK_SIZE) * BLOCK_SIZE) / UB_NUM_INT16_ONE))) {
            bufferOpen = 0;
            ubDataNumber = UB_NUM_INT16_ONE;
        } else {
            ubDataNumber = UB_NUM_INT16_TWO;
        }
    } else {
        if ((inputLengthAlgin < coreNum * (((ubSize / BLOCK_SIZE) * BLOCK_SIZE) / UB_NUM_INT8_UINT8_ONE))) {
            bufferOpen = 0;
            ubDataNumber = UB_NUM_INT8_UINT8_ONE;
        } else {
            ubDataNumber = UB_NUM_INT8_UINT8_TWO;
        }
    }

    if (0 == inputBytes) {
        OP_LOGE(context, "inputBytes is 0");
        return ge::GRAPH_FAILED;
    }

    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlgin / BLOCK_SIZE) ? coreNum :
                                                                                     inputLengthAlgin / BLOCK_SIZE;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlgin / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin / BLOCK_SIZE) % coreNum;
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    tilingData.set_smallCoreDataNum(smallCoreDataNum);
    tilingData.set_bigCoreDataNum(bigCoreDataNum);
    tilingData.set_finalSmallTileNum(finalSmallTileNum);
    tilingData.set_finalBigTileNum(finalBigTileNum);
    tilingData.set_tileDataNum(tileDataNum);
    tilingData.set_smallTailDataNum(smallTailDataNum);
    tilingData.set_bigTailDataNum(bigTailDataNum);
    tilingData.set_tailBlockNum(tailBlockNum);
    tilingData.set_bufferOpen(bufferOpen);
    // 以上是int16、int8、uint8类型的logit增加的代码
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    if (dataType == ge::DT_UINT8 || dataType == ge::DT_INT8 || dataType == ge::DT_INT16) {
        needCoreNum = coreNum;
    }
    tilingContext->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4LogitTiling([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingLogitTiling(gert::TilingContext* context)
{
    LogitTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling(context);
}

IMPL_OP_OPTILING(Logit).Tiling(TilingLogitTiling).TilingParse<LogitCompileInfo>(TilingPrepare4LogitTiling);
} // namespace optiling
