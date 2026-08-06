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
 * \file apply_adam_w_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/apply_adam_w_tiling_data.h"
#include "../op_kernel/apply_adam_w_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

const uint32_t WS_SYS_SIZE = 0;
const uint32_t DATA_NUM_FP32 = 5;
const uint32_t DATA_NUM_FP16 = 5;
const uint32_t DATA_NUM_BF16 = 12;
const uint32_t BETA1_POWER_INDEX = 3;
const uint32_t BETA2_POWER_INDEX = 4;
const uint32_t LR_INDEX = 5;
const uint32_t WD_INDEX = 6;
const uint32_t BETA1_INDEX = 7;
const uint32_t BETA2_INDEX = 8;
const uint32_t ESPILON_INDEX = 9;
const uint32_t TILE_SPLIT_NUM = 1024;
const uint32_t SINGLE_BUFFER_NUM = 1;
const uint32_t DOUBLE_BUFFER_NUM = 2;
struct ApplyAdamWCompileInfo {};

static ge::graphStatus TilingParseForApplyAdamW([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, uint64_t coreNum,
                                         uint64_t& inputNum, uint64_t& bytesPerElement, uint64_t& tileBlockNum,
                                         uint64_t& tileDataNum, uint64_t& inputLengthAlgin, uint32_t& bufferNum,
                                         bool amsgrad, uint32_t blockSize)
{
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        OP_LOGE(context->GetNodeName(), "inputNum is 0");
        return ge::GRAPH_FAILED;
    }
    bytesPerElement = typeLength;
    auto dataType = context->GetInputDesc(0)->GetDataType();

    // 根据数据类型确定每个元素需要的UB数据量倍数
    uint64_t ubDataNumber = DATA_NUM_FP32;
    switch (dataType) {
        case ge::DT_FLOAT:
            ubDataNumber = DATA_NUM_FP32 + (amsgrad ? 1 : 0); // 输入1 + 输入2 + 输出
            break;
        case ge::DT_BF16:
            ubDataNumber = DATA_NUM_BF16 + (amsgrad ? 1 : 0); // 可能需要更多中间缓冲区
            break;
        case ge::DT_FLOAT16:
            ubDataNumber = DATA_NUM_BF16 + (amsgrad ? 3 : 0); // 可能需要更多中间缓冲区
            break;
        default:
            break;
    }

    inputLengthAlgin = (((inputLength + blockSize - 1) / blockSize) * blockSize);

    // 动态决定buffer数量：如果UB空间足够，使用单流水；否则使用双流水
    uint64_t singleBufferNeedSize = inputLengthAlgin * ubDataNumber;
    if (singleBufferNeedSize <= coreNum * ubSize) {
        // UB空间足够，使用单流水以获得更好性能
        bufferNum = SINGLE_BUFFER_NUM;
    } else {
        // UB空间不足，使用双流水
        bufferNum = DOUBLE_BUFFER_NUM;
    }

    if (ubDataNumber == 0 || bufferNum == 0) {
        OP_LOGE(context->GetNodeName(), "ubDataNumber or bufferNum is 0");
        return ge::GRAPH_FAILED;
    }

    // 根据选择的buffer数量计算tile参数
    tileBlockNum = (ubSize / bufferNum / blockSize) / ubDataNumber;
    if (bytesPerElement == 0) {
        OP_LOGE(context->GetNodeName(), "bytesPerElement is 0");
        return ge::GRAPH_FAILED;
    }
    tileDataNum = (tileBlockNum * blockSize) / bytesPerElement;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(gert::TilingContext* context, uint64_t inputLengthAlgin, int64_t coreNum,
                                              uint64_t tileBlockNum, uint64_t bytesPerElement, uint64_t tileDataNum,
                                              uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum,
                                              uint64_t& smallTailDataNum, uint64_t& bigTailDataNum,
                                              uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum,
                                              uint64_t& tailBlockNum, uint32_t blockSize)
{
    if (0 == blockSize || 0 == coreNum || 0 == tileBlockNum || 0 == bytesPerElement) {
        OP_LOGE(context->GetNodeName(), "blockSize or coreNum or tileBlockNum or bytesPerElement is 0");
        return ge::GRAPH_FAILED;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlgin / blockSize / coreNum;
    tailBlockNum = (inputLengthAlgin / blockSize) % coreNum;

    smallCoreDataNum = everyCoreInputBlockNum * blockSize / bytesPerElement;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * blockSize / bytesPerElement;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyAdamWTilingFunc(gert::TilingContext* context)
{
    ApplyAdamWTilingData* tiling = context->GetTilingData<ApplyAdamWTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(ApplyAdamWTilingData), 0, sizeof(ApplyAdamWTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 获取平台运行信息
    // 获取输入数据信息和动态决定buffer数量
    uint64_t ubSize, inputNum, bytesPerElement, tileBlockNum, tileDataNum, inputLengthAlgin;
    int64_t coreNum;
    uint32_t bufferNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ret);

    // 动态获取UB block size，兼容不同芯片的UB对齐粒度
    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    bool amsgrad = *(attrs->GetBool(0));
    bool maximize = *(attrs->GetBool(1));

    ret = GetShapeAttrsInfo(context, ubSize, coreNum, inputNum, bytesPerElement, tileBlockNum, tileDataNum,
                            inputLengthAlgin, bufferNum, amsgrad, blockSize);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ret);

    // 计算coreNum
    uint64_t calcCoreNum = inputNum / TILE_SPLIT_NUM;

    if (inputNum % TILE_SPLIT_NUM)
        calcCoreNum = calcCoreNum + 1;
    coreNum = (calcCoreNum < static_cast<uint64_t>(coreNum)) ? calcCoreNum : coreNum;

    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum,
        tailBlockNum;
    ret = CalculateCoreBlockNums(context, inputLengthAlgin, coreNum, tileBlockNum, bytesPerElement, tileDataNum,
                                 smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum,
                                 finalBigTileNum, tailBlockNum, blockSize);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ret);

    // 设置tiling数据
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    tiling->amsgrad = amsgrad ? 1 : 0;
    tiling->maximize = maximize ? 1 : 0;

    // 计算workspace大小
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);
    uint64_t tilingKey = 0;
    if (bufferNum == DOUBLE_BUFFER_NUM) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);

    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(ApplyAdamW).Tiling(ApplyAdamWTilingFunc).TilingParse<ApplyAdamWCompileInfo>(TilingParseForApplyAdamW);
} // namespace optiling
