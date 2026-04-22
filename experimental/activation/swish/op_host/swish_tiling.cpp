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
 * \file swish_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "swish/op_kernel/swish_tiling_data.h"
#include "swish/op_kernel/swish_tiling_key.h"

namespace optiling {

struct SwishCompileInfo {};

const uint64_t BUFFER_NUM = 2;

// tiling 分发入口
static ge::graphStatus SwishTilingFunc(gert::TilingContext* context)
{
    uint64_t blockSize = 0;
    uint64_t ubSize = 0;

    blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);

    SwishTilingData* tiling = context->GetTilingData<SwishTilingData>();
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto coreNum = ascendcPlatform.GetCoreNum();
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND910B && socVersion != platform_ascendc::SocVersion::ASCEND310B && context->GetInputDesc(0)->GetDataType() == ge::DT_BF16) {
        OP_LOGE(context, "socVersion error.");
        return ge::GRAPH_FAILED;
    }

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    uint64_t inputNum = inputShape->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(inputNum == 0, OP_LOGE(context, "inputNum is 0"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();    
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);

    uint64_t inputLength = inputNum * typeLength;
    uint64_t inputBytes = inputLength / inputNum;
    OP_CHECK_IF(inputNum == 0, OP_LOGE(context, "inputNum is 0"), return ge::GRAPH_FAILED);

    uint64_t ubDataNumber = (dataType == ge::DT_FLOAT) ? 2 : 4;
    uint64_t tileBlockNum = (ubSize / blockSize / BUFFER_NUM) / ubDataNumber;
    uint64_t tileDataNum = (tileBlockNum * blockSize) / inputBytes;

    uint64_t inputLengthAlgin32 = (((inputLength + blockSize - 1) / blockSize) * blockSize);
    if(tileDataNum >= inputNum)
    {
        coreNum=1;
    }
    else
    {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
        coreNum = (coreNum <  inputLengthAlgin32 / blockSize) ? coreNum : inputLengthAlgin32 / blockSize;
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputBytes == 0, OP_LOGE(context, "inputBytes is 0"), return ge::GRAPH_FAILED);

    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / blockSize / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin32 / blockSize) % coreNum;
    
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * blockSize / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint64_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
    
    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * blockSize / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint64_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum; 
    
    tiling->smallCoreDataNum = (uint32_t)smallCoreDataNum;
    tiling->bigCoreDataNum = (uint32_t)bigCoreDataNum;
    tiling->tileDataNum = (uint32_t)tileDataNum;
    tiling->smallTailDataNum = (uint32_t)smallTailDataNum;
    tiling->bigTailDataNum = (uint32_t)bigTailDataNum;
    tiling->finalSmallTileNum = (uint32_t)finalSmallTileNum;
    tiling->finalBigTileNum = (uint32_t)finalBigTileNum;
    tiling->tailBlockNum = (uint32_t)tailBlockNum;

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
    float scale = scaleValueAttr == nullptr ? 1.0f : *scaleValueAttr;
    tiling->scale = scale;

    uint64_t attrWork = TPL_SCALE_OTHER;
    constexpr float NEG_ONE = -1.0f;
    constexpr float ZERO = 0.0f;
    if (scale == NEG_ONE) {
        attrWork = TPL_SCALE_NEG_ONE;
    } else if (scale == ZERO) {
        attrWork = TPL_SCALE_ZERO;
    } else {
        attrWork = TPL_SCALE_OTHER;
    }

    context->SetBlockDim(coreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(TPL_SCH_MODE_0, attrWork));
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
static ge::graphStatus TilingPrepare4Swish([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Swish).Tiling(SwishTilingFunc).TilingParse<SwishCompileInfo>(TilingPrepare4Swish);
} // namespace optiling
