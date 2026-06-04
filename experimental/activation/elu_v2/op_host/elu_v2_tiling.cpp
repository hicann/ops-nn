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
 * \file elu_v2_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/elu_v2_tiling_data.h"
#include "../op_kernel/elu_v2_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 256;
static const gert::Shape g_vec_1_shape = {1};

struct EluV2CompileInfo {};

inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取平台信息指针
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    // 创建AscendC平台对象
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    // 获取AI Core数量
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    // 获取UB(统一缓冲区)大小
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& inputDataNum, ge::DataType& dataType)
{
    // 获取输入x的shape信息
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    
    // 如果输入shape是标量，转换为{1}，否则保持原shape不变
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    
    
    // 获取输出y的shape信息
    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    // shape校验
    OP_CHECK_IF(
        inputShapeX.GetDimNum() != outShapeY.GetDimNum(),
        OP_LOGE(
            context, "EluV2: inputx,outputy shape dim = %zu, %zu, should be equal",
            inputShapeX.GetDimNum(), outShapeY.GetDimNum()),
        return ge::GRAPH_FAILED);

    // 获取shape的维度值
    inputDataNum = inputShapeX.GetShapeSize();
    
    // 数据类型校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    // 获取工作空间大小数组
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}


// tiling 分发入口
static ge::graphStatus EluV2TilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行时信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, 
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    
    // 2、获取shape、属性信息
    int64_t inputDataNum;
    ge::DataType dataType;

    OP_CHECK_IF(
        GetShapeAttrsInfo(context, inputDataNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), 
        return ge::GRAPH_FAILED);
    
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    EluV2TilingData* tiling = context->GetTilingData<EluV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    
    // 初始化tiling数据为0
    OP_CHECK_IF(
        memset_s(tiling, sizeof(EluV2TilingData), 0, sizeof(EluV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), 
        return ge::GRAPH_FAILED);

    const float alpha = *(context->GetAttrs()->GetFloat(0));
    const float scale = *(context->GetAttrs()->GetFloat(1));
    const float inputScale = *(context->GetAttrs()->GetFloat(2));

    int64_t dataTypeLength = (dataType == ge::DT_FLOAT) ? 4 : 2;
    // 每个block中元素的数量
    int64_t blockElemNum = BLOCK_SIZE / dataTypeLength;
    // ub中能容纳的block数量    
    int64_t ubBlockNum = ubSize / BLOCK_SIZE;

    int64_t ubPartNum = (dataTypeLength == 4) ? 5 : 9;
    // 单次处理最大的block数量
    int64_t tileBlockNum = ubBlockNum / ubPartNum;
    // 单次处理最大的元素数量
    int64_t tileDataNum = tileBlockNum * blockElemNum;
    // 输入元素数量对应的block数量
    int64_t inputBlockNum = (inputDataNum + blockElemNum - 1) / blockElemNum;//向上取整
    
    int64_t maxCoreNum = (inputBlockNum + tileBlockNum - 1) / tileBlockNum;//向上取整

    int64_t usedCoreNum = (coreNum < maxCoreNum) ? coreNum : maxCoreNum;
    usedCoreNum = (usedCoreNum >= 1) ? usedCoreNum : 1;

    int64_t smallCoreBlockNum = inputBlockNum / usedCoreNum;
    int64_t bigCoreNum = inputBlockNum % usedCoreNum;
    int64_t smallCoreDataNum = smallCoreBlockNum * blockElemNum;
    int64_t bigCoreDataNum = smallCoreDataNum + blockElemNum;

    context->SetBlockDim(usedCoreNum);

    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->bigCoreNum = bigCoreNum;

    tiling->alpha = alpha;
    tiling->scale = scale;
    tiling->inputScale = inputScale;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForEluV2([[maybe_unused]] gert::TilingParseContext* context)
{   
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(EluV2).Tiling(EluV2TilingFunc).TilingParse<EluV2CompileInfo>(TilingParseForEluV2);
} // namespace optiling
