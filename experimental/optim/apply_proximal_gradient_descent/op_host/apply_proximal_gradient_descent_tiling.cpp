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
 * \file apply_proximal_gradient_descent_tiling.cpp
 * \brief ApplyProximalGradientDescent tiling (arch35)
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/apply_proximal_gradient_descent_tiling_data.h"
#include "../op_kernel/apply_proximal_gradient_descent_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 32U;     // workspace 占位 32B（无实际使用）
// 双缓冲阈值：数据量 > 阈值启用双缓冲
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 系统/流水线开销预留（防止 UB 满分配后 pipe 控制结构溢出）
constexpr int64_t UB_RESERVE_BYTES = 8 * 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

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

// 获取输入 shape / dtype
// 输入顺序：0=var, 1=alpha, 2=l1, 3=l2, 4=delta
// 输出顺序：0=varOut
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    auto varStorage = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, varStorage);
    auto varShape = EnsureNotScalar(varStorage->GetStorageShape());

    auto deltaStorage = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, deltaStorage);
    auto deltaShape = EnsureNotScalar(deltaStorage->GetStorageShape());

    auto outStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outStorage);
    auto outShape = EnsureNotScalar(outStorage->GetStorageShape());

    OP_CHECK_IF(
        varShape.GetShapeSize() != deltaShape.GetShapeSize() ||
            varShape.GetShapeSize() != outShape.GetShapeSize(),
        OP_LOGE(
            context, "ApplyProximalGradientDescent: var/delta/varOut shape size mismatch: var=%ld, delta=%ld, varOut=%ld",
            varShape.GetShapeSize(), deltaShape.GetShapeSize(), outShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    totalIdx = varShape.GetShapeSize();

    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype: %d", static_cast<int>(dataType));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyProximalGradientDescentTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    int64_t totalIdx;
    ge::DataType dataType;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ApplyProximalGradientDescentTilingData* tiling =
        context->GetTilingData<ApplyProximalGradientDescentTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ApplyProximalGradientDescentTilingData), 0,
                 sizeof(ApplyProximalGradientDescentTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    tiling->totalNum = totalIdx;

    // 空 tensor 分支：blockDim=1，blockFactor=0，ubFactor=0；Kernel 检测 blockLength_==0 直接 return
    if (totalIdx == 0) {
        tiling->blockFactor = 0;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        // 仍需选择一个合法 TilingKey，缺省选 FP32+DB 即可（不进 kernel）
        uint32_t dTypeX = static_cast<uint32_t>(dataType);
        uint64_t useDoubleBuffer = 1;
        ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
        return ge::GRAPH_SUCCESS;
    }

    // 多核切分：按 DMA 最小粒度对齐
    tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
    int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

    // UB 切分（按每元素实际占用字节数计算，预留系统开销 8KB）
    // FP32 路径每元素占用：
    //   单缓冲 SB：(in_var + in_delta + out) * 4B + 3 * tmp(FP32 4B) = 24B
    //   双缓冲 DB：6 * queue(FP32 4B) + 3 * tmp(FP32 4B) = 36B
    // FP16 路径每元素占用：
    //   单缓冲 SB：(in_var + in_delta + out) * 2B + 3 * tmp(FP32 4B) + 2 * cast(FP32 4B) = 26B
    //   双缓冲 DB：6 * queue(FP16 2B) + 3 * tmp(FP32 4B) + 2 * cast(FP32 4B) = 32B
    // 取每 (dtype, buffer_mode) 真实最大值，避免 UB 越界
    uint64_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bytesPerElem;
    if (dataType == ge::DT_FLOAT) {
        bytesPerElem = useDoubleBuffer ? 36 : 24;
    } else { // DT_FLOAT16
        bytesPerElem = useDoubleBuffer ? 32 : 26;
    }
    int64_t usableUbSize =
        (static_cast<int64_t>(ubSize) > UB_RESERVE_BYTES) ?
        (static_cast<int64_t>(ubSize) - UB_RESERVE_BYTES) : static_cast<int64_t>(ubSize);
    tiling->ubFactor = Ops::Base::FloorAlign(
        Ops::Base::FloorDiv(usableUbSize, bytesPerElem), ubBlockSize);

    context->SetBlockDim(usedCoreNum);

    // TilingKey 参数：D_T_X + BUFFER_MODE（与 ASCENDC_TPL_ARGS_DECL 顺序一致）
    // 编码：tilingKey = dtype | (buffer_mode << 8)
    //   FP32+SB=0   FP16+SB=1   FP32+DB=256   FP16+DB=257
    // 与 binary 索引（_0/_1/_256/_257）一一对齐
    uint32_t dTypeX = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, dTypeX, useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForApplyProximalGradientDescent(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyProximalGradientDescentCompileInfo {}; // 入图场景依赖

IMPL_OP_OPTILING(ApplyProximalGradientDescent)
    .Tiling(ApplyProximalGradientDescentTilingFunc)
    .TilingParse<ApplyProximalGradientDescentCompileInfo>(TilingParseForApplyProximalGradientDescent);

} // namespace optiling
