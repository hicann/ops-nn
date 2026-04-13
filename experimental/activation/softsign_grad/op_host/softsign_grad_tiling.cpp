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
 * \file softsign_grad_tiling.cpp
 * \brief SoftsignGrad 算子 Tiling 实现 (arch35)
 *
 * Tiling 策略：
 *   - 按总元素数均匀切分到多核
 *   - 每个核按 ubFactor 分块处理
 *   - 通过 TilingKey (dtype + bufferMode) 选择不同 Kernel 模板
 *
 * 全量覆盖（迭代三完成）：支持 FP16/FP32/BF16 x 单/双缓冲（6 个 TilingKey 组合）
 *   - 双缓冲在 totalNum > MIN_SPLIT_THRESHOLD 时自动启用
 *   - bufferNum 按 dtype 路径精确计算
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/softsign_grad_tiling_data.h"
#include "../op_kernel/softsign_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t FP32_SIZE = 4;   // float 字节数（内部计算基准）
// 双缓冲阈值：元素数大于此值时启用双缓冲
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

// 获取平台信息：UB 大小和 AIV 核数
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

// 获取 shape 和 dtype 信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    // gradients (input 0)
    auto gradientsDesc = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradientsDesc);
    auto gradientsShape = EnsureNotScalar(gradientsDesc->GetStorageShape());

    // features (input 1)
    auto featuresDesc = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresDesc);
    auto featuresShape = EnsureNotScalar(featuresDesc->GetStorageShape());

    // output
    auto outputDesc = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    auto outputShape = EnsureNotScalar(outputDesc->GetStorageShape());

    // Shape 校验：gradients 和 features 的 shape 必须一致
    OP_CHECK_IF(
        gradientsShape != featuresShape || gradientsShape != outputShape,
        OP_LOGE(
            context, "SoftsignGrad: shape mismatch: gradients=%s, features=%s, output=%s",
            Ops::Base::ToString(gradientsShape).c_str(), Ops::Base::ToString(featuresShape).c_str(),
	    Ops::Base::ToString(outputShape).c_str()),
        return ge::GRAPH_FAILED);

    totalNum = gradientsShape.GetShapeSize();

    // dtype 校验：支持 FP16/FP32/BF16
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "SoftsignGrad: invalid dtype=%d, supported: FP16/FP32/BF16", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }

    // 校验 gradients 与 features dtype 一致
    auto featuresDescInfo = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresDescInfo);
    OP_CHECK_IF(
        featuresDescInfo->GetDataType() != dataType,
        OP_LOGE(context, "SoftsignGrad: gradients and features dtype mismatch"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// Tiling 入口
static ge::graphStatus SoftsignGradTilingFunc(gert::TilingContext* context)
{
    // 1、平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2、shape / dtype
    int64_t totalNum;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    // 3、workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、计算 Tiling 参数
    SoftsignGradTilingData* tiling = context->GetTilingData<SoftsignGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftsignGradTilingData), 0, sizeof(SoftsignGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    // 多核切分
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // UB 切分：根据 dtype 和 buffer 模式计算
    // 以 FP32 大小为基准计算 ubFactor，保证任何 dtype 路径 UB 使用不超限
    //
    // 双缓冲决策：totalNum > MIN_SPLIT_THRESHOLD 时启用
    int64_t ubBlockSize = GetUbBlockSize(context);
    uint64_t useDoubleBuffer = (totalNum > MIN_SPLIT_THRESHOLD) ? 1 : 0;

    // bufferNum 计算（以 float 为单位的等价 buffer 数）
    // FP32 路径 (NEEDS_CAST=false):
    //   - 单缓冲: 2 input(float,*1) + 1 tmp(float) + 1 output(float,*1) = 4
    //   - 双缓冲: 2 input(float,*2) + 1 tmp(float) + 1 output(float,*2) = 7
    //
    // FP16/BF16 路径 (NEEDS_CAST=true):
    //   - 单缓冲: 2 input(half,*1=1 float eq) + 2 cast(float) + 1 tmp(float) + 1 output(half,*1=0.5 float eq) = 4.5 -> ceil to 5
    //   - 双缓冲: 2 input(half,*2=2 float eq) + 2 cast(float) + 1 tmp(float) + 1 output(half,*2=1 float eq) = 6
    //
    // Use dtype-specific bufferNum for tighter UB utilization
    int64_t bufferNum;
    bool needsCast = (dataType != ge::DT_FLOAT);
    if (needsCast) {
        // FP16/BF16: half queues + float cast/tmp buffers
        bufferNum = useDoubleBuffer ? 6 : 5;
    } else {
        // FP32: all float buffers
        bufferNum = useDoubleBuffer ? 7 : 4;
    }

    tiling->ubFactor = FloorAlign(
        FloorDiv(static_cast<int64_t>(ubSize) / FP32_SIZE, bufferNum),
        ubBlockSize);

    context->SetBlockDim(usedCoreNum);

    // 5、设置 TilingKey
    // 参数顺序与 softsign_grad_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 一致
    // 映射到 softsign_grad_arch35.cpp 中 kernel 模板参数：softsign_grad<D_T, BUFFER_MODE>
    uint32_t dType = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dType, useDoubleBuffer);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSoftsignGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SoftsignGradCompileInfo {};  // 入图场景依赖

IMPL_OP_OPTILING(SoftsignGrad)
    .Tiling(SoftsignGradTilingFunc)
    .TilingParse<SoftsignGradCompileInfo>(TilingParseForSoftsignGrad);

} // namespace optiling
