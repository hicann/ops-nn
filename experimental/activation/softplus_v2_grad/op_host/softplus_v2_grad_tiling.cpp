/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../op_kernel/softplus_v2_grad_tiling_data.h"
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static const uint32_t BLOCK_SIZE = 512; // 改为512字节对齐
static const uint32_t BUFFER_NUM = 2;
static const uint32_t DATA_COPY_ALIGNMENT = 512; // DataCopy改为512字节对齐
static const uint32_t COMPUTE_ALIGNMENT = 256;   // 计算操作保持256字节对齐

// 对齐计算函数（保留，用于host侧对齐计算）
static uint32_t AlignSize(uint32_t size, uint32_t alignment)
{
    if (alignment == 0) {
        return size; // 避免除0
    }
    return (size + alignment - 1) / alignment * alignment;
}

struct SoftplusV2GradCompileInfo {};

static ge::graphStatus TilingParseForSoftplusV2Grad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    // 计算Workspace大小，确保512字节对齐
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);

    // Workspace用于临时存储对齐后的数据
    size_t usrSize = COMPUTE_ALIGNMENT * 2;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SoftplusV2GradTilingFunc(gert::TilingContext* context)
{
    SoftplusV2GradTilingData* tiling = context->GetTilingData<SoftplusV2GradTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // 获取输入数据信息
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);

    uint32_t UB_DATA_NUMBER = 0;

    // 设置数据类型标识
    if (dataType == ge::DT_FLOAT16) {
        // float16 输入 + float32多个缓冲区  --> 24
        UB_DATA_NUMBER = 24;
    } else if (dataType == ge::DT_FLOAT) {
        // float32 输入 + float32多个缓冲区   --> 13
        UB_DATA_NUMBER = 13;
    } else if (dataType == ge::DT_BF16) {
        // bfloat16 输入 + float32多个缓冲区  -- > 24
        UB_DATA_NUMBER = 24;
    }

    // host侧进行512字节对齐处理
    uint32_t inputLength = inputNum * typeLength;
    uint32_t inputBytes = inputLength / inputNum;
    uint32_t tileBlockNum = (ubSize / BLOCK_SIZE) / UB_DATA_NUMBER;
    uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    // host侧确保数据长度512字节对齐
    uint32_t inputLengthAlign512 = AlignSize(inputLength, BLOCK_SIZE);

    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlign512 / BLOCK_SIZE) ?
                      coreNum :
                      inputLengthAlign512 / BLOCK_SIZE;
    }

    uint32_t everyCoreInputBlockNum = inputLengthAlign512 / BLOCK_SIZE / coreNum;
    uint32_t tailBlockNum = (inputLengthAlign512 / BLOCK_SIZE) % coreNum;
    uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;

    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    // 计算对齐长度使用512字节对齐
    uint32_t smallprocessDataNum_computes = AlignSize(tileDataNum * inputBytes, COMPUTE_ALIGNMENT) / inputBytes;
    uint32_t tailsmallprocessDataNum_computes = AlignSize(smallTailDataNum * inputBytes, COMPUTE_ALIGNMENT) /
                                                inputBytes;

    everyCoreInputBlockNum += 1;
    uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;
    uint32_t bigprocessDataNum_computes = AlignSize(tileDataNum * inputBytes, COMPUTE_ALIGNMENT) / inputBytes;
    uint32_t tailbigprocessDataNum_computes = AlignSize(bigTailDataNum * inputBytes, COMPUTE_ALIGNMENT) / inputBytes;

    // 填充Tiling参数
    tiling->typeLength = typeLength;
    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<uint64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);
    tiling->bigprocessDataNum_computes = static_cast<uint64_t>(bigprocessDataNum_computes);
    tiling->smallprocessDataNum_computes = static_cast<uint64_t>(smallprocessDataNum_computes);
    tiling->tailbigprocessDataNum_computes = static_cast<uint64_t>(tailbigprocessDataNum_computes);
    tiling->tailsmallprocessDataNum_computes = static_cast<uint64_t>(tailsmallprocessDataNum_computes);

    // 获取SoftplusBackward算子属性
    float beta = 1.0f;
    float threshold = 20.0f;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* beta_ptr = attrs->GetFloat(0);
        if (beta_ptr != nullptr)
            beta = *beta_ptr;
        const float* threshold_ptr = attrs->GetFloat(1);
        if (threshold_ptr != nullptr)
            threshold = *threshold_ptr;
    }
    tiling->beta = beta;
    tiling->threshold = threshold;

    // 设置Workspace大小
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);

    context->SetTilingKey(0);
    context->SetBlockDim(coreNum);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SoftplusV2Grad)
    .Tiling(SoftplusV2GradTilingFunc)
    .TilingParse<SoftplusV2GradCompileInfo>(TilingParseForSoftplusV2Grad);
} // namespace optiling