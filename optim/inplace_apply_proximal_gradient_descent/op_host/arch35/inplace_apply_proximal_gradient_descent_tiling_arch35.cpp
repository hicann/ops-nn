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
 * =============================================================================
 * inplace_apply_proximal_gradient_descent_package/op_host/arch35/inplace_apply_proximal_gradient_descent_tiling_arch35.cpp
 * =============================================================================
 * Role: DESIGN §9.9 TilingFuncInplaceApplyProximalGradientDescent 的 TilingContext
 *       glue：平台查询（§9.9 ReadPlatform，GetCoreNumAiv/GetVecRegLen/
 *       GetCoreMemSize）→ StorageShape 逐维同形与标量载体校验（§9.3）→ dim0 展平 →
 *       BUFFER_MODE 阈值（§9.4）→ 空 Tensor 短路 / CalcUbFactor + CalcMultiCore +
 *       CalcLoopTail（§9.4/§9.5）→ typed GetTilingData<T> + GetWorkspaceSizes(1)
 *       + SetBlockDim 提交（§9.7）→ 仅按 BUFFER_MODE 调用
 *       ASCENDC_TPL_SEL_PARAM（§6）。§9.8 以空 CompileInfo + TilingParse 注册，
 *       不声明 TilingInputsDataDependency（Host 不读取三路 scalar payload）。
 *
 * 失败点与副作用严格服从 §9.9 失败点表：前置校验或切分失败返回
 * GRAPH_FAILED 且不提交 TilingData；typed TilingData/workspace 获取或
 * SetBlockDim 失败时不调用 mode selector。
 * =============================================================================
 */

#include <cstdint>
#include <limits>
#include "tiling/platform/platform_ascendc.h"
#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "../../op_kernel/arch35/inplace_apply_proximal_gradient_descent_tiling_key.h"
#include "../../op_kernel/arch35/inplace_apply_proximal_gradient_descent_tiling_data.h"
#include "inplace_apply_proximal_gradient_descent_tiling_host_arch35.h"
#include "inplace_apply_proximal_gradient_descent_tiling_arch35.h"

namespace optiling {

namespace {
// §9.4 stable binary route contract; changes require a profiling baseline.
constexpr int64_t kMinSplitThreshold = 1024;
constexpr char kOpName[] = "InplaceApplyProximalGradientDescent";
} // namespace

ge::graphStatus TilingFuncInplaceApplyProximalGradientDescent(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(context->GetNodeName(), "Enter InplaceApplyProximalGradientDescentTilingFunc");

    // ==================== §9.9 ReadPlatform（平台查询） ====================
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGI(kOpName, "DBG rc=platform-null");
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t availableCoreNum = platform.GetCoreNumAiv();
    const uint32_t vecRegLen = platform.GetVecRegLen();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (availableCoreNum == 0 || vecRegLen == 0 || vecRegLen % sizeof(float) != 0 || ubSize == 0 ||
        ubSize > static_cast<uint64_t>(INT64_MAX)) {
        OP_LOGI(kOpName, "DBG rc=platform-values aiv=%u vl=%u ub=%lu", availableCoreNum, vecRegLen,
                static_cast<unsigned long>(ubSize));
        return ge::GRAPH_FAILED;
    }
    // =========== §9.2 input 0 descriptor 必须存在；dtype 由 OpDef 外层展开 ===========
    const auto* varInputDesc = context->GetInputDesc(0);
    if (varInputDesc == nullptr) {
        OP_LOGI(kOpName, "DBG rc=input0-desc-null");
        return ge::GRAPH_FAILED;
    }
    // ============ §9.3 shape 指针与 StorageShape 逐维校验 ============
    const auto* varDesc = context->GetInputShape(0);
    const auto* alphaDesc = context->GetInputShape(1);
    const auto* l1Desc = context->GetInputShape(2);
    const auto* l2Desc = context->GetInputShape(3);
    const auto* deltaDesc = context->GetInputShape(4);
    const auto* outDesc = context->GetOutputShape(0);
    if (varDesc == nullptr || alphaDesc == nullptr || l1Desc == nullptr || l2Desc == nullptr || deltaDesc == nullptr ||
        outDesc == nullptr) {
        OP_LOGI(kOpName, "DBG rc=shape-desc-null v=%d a=%d l1=%d l2=%d d=%d o=%d", varDesc != nullptr,
                alphaDesc != nullptr, l1Desc != nullptr, l2Desc != nullptr, deltaDesc != nullptr, outDesc != nullptr);
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& varShape = varDesc->GetStorageShape();
    const gert::Shape& alphaShape = alphaDesc->GetStorageShape();
    const gert::Shape& l1Shape = l1Desc->GetStorageShape();
    const gert::Shape& l2Shape = l2Desc->GetStorageShape();
    const gert::Shape& deltaShape = deltaDesc->GetStorageShape();
    const gert::Shape& outShape = outDesc->GetStorageShape();
    // rank 0–16、var/delta/var_out 逐维完全同形、三路标量载体仅 0-D/[1]
    if (varShape.GetDimNum() > 16 || deltaShape.GetDimNum() > 16 || outShape.GetDimNum() > 16 ||
        !ExactShapeEqual(varShape, deltaShape) || !ExactShapeEqual(varShape, outShape) ||
        !IsSharedScalarShape(alphaShape) || !IsSharedScalarShape(l1Shape) || !IsSharedScalarShape(l2Shape)) {
        OP_LOGI(kOpName,
                "DBG rc=shape-check vr=%zu dr=%zu or=%zu veq=%d oeq=%d "
                "asc=%d l1c=%d l2c=%d",
                varShape.GetDimNum(), deltaShape.GetDimNum(), outShape.GetDimNum(),
                ExactShapeEqual(varShape, deltaShape), ExactShapeEqual(varShape, outShape),
                IsSharedScalarShape(alphaShape), IsSharedScalarShape(l1Shape), IsSharedScalarShape(l2Shape));
        return ge::GRAPH_FAILED;
    }

    // ============ §9.3 展平 dim0 / §9.4 BUFFER_MODE 阈值 ============
    int64_t dim0 = 0;
    if (!CalcDim0(varShape, dim0)) {
        OP_LOGI(kOpName, "DBG rc=calcdim0-fail");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(kOpName, "DBG dim0=%ld bufferMode=%lu", dim0,
            static_cast<unsigned long>((dim0 <= kMinSplitThreshold) ? 0U : 1U));
    const uint64_t bufferMode = (dim0 <= kMinSplitThreshold) ? 0U : 1U;

    // ============ §9.5 切分（空 Tensor 短路：核数 1、其余 0） ============
    int32_t usedCoreNum = 1;
    int64_t blockFactor = 0;
    int64_t blockTail = 0;
    int64_t ubFactor = 0;
    int64_t formerLoop = 0;
    int64_t formerTail = 0;
    int64_t tailLoop = 0;
    int64_t tailTail = 0;
    if (dim0 > 0) {
        // §9 公共链路的切分只依赖 dim0、平台量与 BUFFER_MODE。datatype 不参与
        // UB/多核公式，由 OpDef profile 在构建期生成独立外层二进制。
        const bool splitOk = CalcUbFactor(ubSize, static_cast<uint8_t>(bufferMode), ubFactor) &&
                             CalcMultiCore(dim0, availableCoreNum, usedCoreNum, blockFactor, blockTail) &&
                             CalcLoopTail(blockFactor, ubFactor, formerLoop, formerTail) &&
                             CalcLoopTail(blockTail, ubFactor, tailLoop, tailTail);
        if (!splitOk || ubFactor > static_cast<int64_t>(UINT32_MAX / sizeof(float))) {
            OP_LOGI(kOpName, "DBG rc=split-fail dim0=%ld ubFactor=%ld splitOk=%d", dim0, ubFactor, splitOk ? 1 : 0);
            return ge::GRAPH_FAILED;
        }
        int64_t maxRepeatTimes = 0;
        const int64_t vlF32 = static_cast<int64_t>(vecRegLen / sizeof(float));
        if (!CeilDivPositive(ubFactor, vlF32, maxRepeatTimes) || maxRepeatTimes > UINT16_MAX) {
            return ge::GRAPH_FAILED;
        }
    }

    // ============ §9.7 typed TilingData / workspace / SetBlockDim 提交 ============
    auto* td = context->GetTilingData<InplaceApplyProximalGradientDescentTilingData>();
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (td == nullptr || workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const bool commitOk = CommitTilingData(
        td, workspace, [context](uint32_t coreNum) { return context->SetBlockDim(coreNum) == ge::GRAPH_SUCCESS; }, dim0,
        usedCoreNum, blockFactor, blockTail, ubFactor, formerLoop, formerTail, tailLoop, tailTail);
    if (!commitOk) {
        OP_LOGI(kOpName, "DBG rc=commit-fail");
        return ge::GRAPH_FAILED;
    }

    // §9.7 日志只打印 TilingData 字段与 bufferMode
    OP_LOGI(kOpName,
            "tiling dim0=%ld usedCoreNum=%d reserved=%d blockFactor=%ld "
            "blockTail=%ld ubFactor=%ld ubLoopOfFormerBlock=%ld "
            "ubTailOfFormerBlock=%ld ubLoopOfTailBlock=%ld "
            "ubTailOfTailBlock=%ld bufferMode=%lu",
            td->dim0, td->usedCoreNum, td->reserved, td->blockFactor, td->blockTail, td->ubFactor,
            td->ubLoopOfFormerBlock, td->ubTailOfFormerBlock, td->ubLoopOfTailBlock, td->ubTailOfTailBlock,
            static_cast<unsigned long>(bufferMode));

    // §6 TilingKey 只编码 BUFFER_MODE；dtype 由 def profile 外层二进制承载。
    ASCENDC_TPL_SEL_PARAM(context, bufferMode);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// §9.8 Host 侧注册：空 CompileInfo；不声明 TilingInputsDataDependency
// （Host 不读取三路 scalar payload，负标量/零或负分母均沿普通路径进 Kernel）
// =============================================================================
struct InplaceApplyProximalGradientDescentCompileInfo {};

static ge::graphStatus TilingPrepareForInplaceApplyProximalGradientDescent(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplaceApplyProximalGradientDescent)
    .Tiling(TilingFuncInplaceApplyProximalGradientDescent)
    .TilingParse<InplaceApplyProximalGradientDescentCompileInfo>(TilingPrepareForInplaceApplyProximalGradientDescent);

} // namespace optiling
