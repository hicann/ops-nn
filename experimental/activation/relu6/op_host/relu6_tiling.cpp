/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file relu6_tiling.cpp
 * \brief Relu6 Tiling 实现（arch35 = Ascend950）
 *
 * Tiling 策略：
 * 1. 获取平台信息（UB 大小、AI Core 数量）
 * 2. 获取 shape/dtype 信息
 * 3. 多核切分：blockFactor = ceil(totalNum / coreNum)
 * 4. UB 切分：ubFactor = floor_align(floor_div(ubSize / typeSize / 3), ubBlockSize)
 *    Relu6 需要 3 个 LocalTensor: inputLocal, tmpLocal, outputLocal
 * 5. 设置 TilingKey（由模板参数选择 dtype）
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu6_tiling_data.h"
#include "../op_kernel/relu6_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
// Relu6 需要 3 个 LocalTensor：inputLocal + tmpLocal（Maxs输出）+ outputLocal
constexpr int64_t BUFFER_NUM = 3;
// 双缓冲阈值：数据量大于此值时启用双缓冲（迭代三扩展）
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

/// \brief 确保 shape 不为标量（0 维），若为标量则转换为 {1}
static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

/// \brief 获取平台信息：UB 大小、AI Core 数量
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

/// \brief 获取 shape 和 dtype 信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    // Shape 校验：输入输出元素数一致
    OP_CHECK_IF(
        inputShapeX.GetShapeSize() != outShapeY.GetShapeSize(),
        OP_LOGE(context, "Relu6: input and output shape size mismatch"), return ge::GRAPH_FAILED);

    totalNum = inputShapeX.GetShapeSize();

    // dtype 校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/// \brief 设置 workspace 大小（Relu6 不需要额外 workspace）
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

/// \brief Tiling 分发入口
static ge::graphStatus Relu6TilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取 shape、dtype 信息
    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    // 3. 获取 WorkspaceSize
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. 设置 TilingData
    Relu6TilingData* tiling = context->GetTilingData<Relu6TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(Relu6TilingData), 0, sizeof(Relu6TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 多核切分：将总元素按核数均分
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // UB 切分：
    // 公式：(UB总大小 / 类型大小) / buffer数量，然后按 UB 块大小对齐
    // bufferNum = 3（inputLocal + tmpLocal + outputLocal）
    int64_t typeSize = ge::GetSizeByDataType(dataType);
    OP_CHECK_IF(typeSize == 0, OP_LOGE(context, "typeSize is 0, invalid dataType"), return ge::GRAPH_FAILED);
    int64_t ubBlockSize = GetUbBlockSize(context);
    tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, BUFFER_NUM), ubBlockSize);

    // 设置 dataType 字段：保存原始 ge::DataType 枚举值，供 Kernel 侧模板参数实例化使用
    tiling->dataType = static_cast<int32_t>(dataType);

    context->SetBlockDim(usedCoreNum);

    // 5. 设置 TilingKey（dtype 模板参数选择）
    uint32_t dType = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dType);
    return ge::GRAPH_SUCCESS;
}

/// \brief Tiling 解析（Relu6 不需要额外解析）
static ge::graphStatus TilingParseForRelu6([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct Relu6CompileInfo {}; // 必须定义，入图场景依赖

// Tiling 注册入口
IMPL_OP_OPTILING(Relu6).Tiling(Relu6TilingFunc).TilingParse<Relu6CompileInfo>(TilingParseForRelu6);

} // namespace optiling
