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
 * \file apply_fused_ema_adam_tiling.cpp
 * \brief
 */
#include "apply_fused_ema_adam_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

using namespace std;
namespace optiling {
constexpr uint32_t INPUT_GRAD_IDX = 0;
constexpr uint32_t INPUT_VAR_IDX = 1;
constexpr uint32_t INPUT_M_IDX = 2;
constexpr uint32_t INPUT_V_IDX = 3;
constexpr uint32_t INPUT_S_IDX = 4;
constexpr uint32_t INPUT_STEP_IDX = 5;
constexpr uint32_t ATTR_LR_IDX = 0;
constexpr uint32_t ATTR_EMA_DECAY_IDX = 1;
constexpr uint32_t ATTR_BETA1_IDX = 2;
constexpr uint32_t ATTR_BETA2_IDX = 3;
constexpr uint32_t ATTR_EPS_IDX = 4;
constexpr uint32_t ATTR_MODE_IDX = 5;
constexpr uint32_t ATTR_BIAS_CORRECTION_IDX = 6;
constexpr uint32_t ATTR_WEIGHT_DECAY_IDX = 7;
constexpr uint32_t ONE_BLK_NUM = 16;
constexpr uint32_t ONE_BLK_NUM_FP32 = 8;
constexpr uint32_t BYTE_ONE_BLK = 32;
constexpr uint32_t TBUFFER_NUM = 2;
constexpr uint32_t QUEUE_NUM = 9;
constexpr uint32_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint32_t FP32_DTYPE_SIZE = 4;
ApplyFusedEmaAdamTilingData fusedEmaAdamTiling;

static void PrintInfo(const gert::TilingContext* context)
{
    auto nodeName = context;
    OP_LOGD(nodeName, ">>>>>>>>>>>>> Print tiling data begin <<<<<<<<<<<<<");
    OP_LOGD(nodeName, "lr = %f.", fusedEmaAdamTiling.get_lr());
    OP_LOGD(nodeName, "emaDecay = %f.", fusedEmaAdamTiling.get_emaDecay());
    OP_LOGD(nodeName, "beta1 = %f.", fusedEmaAdamTiling.get_beta1());
    OP_LOGD(nodeName, "beta2 = %f.", fusedEmaAdamTiling.get_beta2());
    OP_LOGD(nodeName, "eps = %f.", fusedEmaAdamTiling.get_eps());
    OP_LOGD(nodeName, "mode = %lu.", fusedEmaAdamTiling.get_mode());
    OP_LOGD(nodeName, "biasCorrection = %lu.", fusedEmaAdamTiling.get_biasCorrection());
    OP_LOGD(nodeName, "weightDecay = %f.", fusedEmaAdamTiling.get_weightDecay());
    OP_LOGD(nodeName, "frontCoreNum = %lu.", fusedEmaAdamTiling.get_frontCoreNum());
    OP_LOGD(nodeName, "tailCoreNum = %lu.", fusedEmaAdamTiling.get_tailCoreNum());
    OP_LOGD(nodeName, "coreCalcNum = %lu.", fusedEmaAdamTiling.get_coreCalcNum());
    OP_LOGD(nodeName, "loopNum = %lu.", fusedEmaAdamTiling.get_loopNum());
    OP_LOGD(nodeName, "coreCalcMax = %lu.", fusedEmaAdamTiling.get_coreCalcMax());
    OP_LOGD(nodeName, "frontCalcExtra = %lu.", fusedEmaAdamTiling.get_frontCalcExtra());
    OP_LOGD(nodeName, "tailCalcExtra = %lu.", fusedEmaAdamTiling.get_tailCalcExtra());
    OP_LOGD(nodeName, ">>>>>>>>>>>>> Print tiling data end <<<<<<<<<<<<<");
}

static ge::graphStatus GetTilingAttr(const gert::TilingContext* context)
{
    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const float* attrLr = attrs->GetAttrPointer<float>(ATTR_LR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrLr);
    fusedEmaAdamTiling.set_lr(static_cast<float>(*attrLr));

    const float* attrEmaDecay = attrs->GetAttrPointer<float>(ATTR_EMA_DECAY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrEmaDecay);
    fusedEmaAdamTiling.set_emaDecay(static_cast<float>(*attrEmaDecay));

    const float* attrBeta1 = attrs->GetAttrPointer<float>(ATTR_BETA1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBeta1);
    fusedEmaAdamTiling.set_beta1(static_cast<float>(*attrBeta1));

    const float* attrBeta2 = attrs->GetAttrPointer<float>(ATTR_BETA2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBeta2);
    fusedEmaAdamTiling.set_beta2(static_cast<float>(*attrBeta2));

    const float* attrEps = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrEps);
    fusedEmaAdamTiling.set_eps(static_cast<float>(*attrEps));

    const uint64_t* attrMode = attrs->GetAttrPointer<uint64_t>(ATTR_MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrMode);
    fusedEmaAdamTiling.set_mode(static_cast<uint64_t>(*attrMode));

    const bool* attrBiasCorrection = attrs->GetAttrPointer<bool>(ATTR_BIAS_CORRECTION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrBiasCorrection);
    auto biasCorrection = *attrBiasCorrection;
    fusedEmaAdamTiling.set_biasCorrection(static_cast<uint64_t>(biasCorrection ? 1 : 0));

    const float* attrWeightDecay = attrs->GetAttrPointer<float>(ATTR_WEIGHT_DECAY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrWeightDecay);
    fusedEmaAdamTiling.set_weightDecay(static_cast<float>(*attrWeightDecay));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDtype(gert::TilingContext* context)
{
    auto nodeName = context;

    auto dtypeInput = context->GetInputDesc(INPUT_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto gradDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto varDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_M_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto mDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_V_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto vDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_S_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto sDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetInputDesc(INPUT_STEP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto stepDtype = dtypeInput->GetDataType();

    bool isDiffDtype =
        (gradDtype != varDtype) || (gradDtype != mDtype) || (gradDtype != vDtype) || (gradDtype != sDtype);
    OP_CHECK_IF(
        isDiffDtype, OP_LOGE(nodeName, "Input grad, var, m, v, s should keep same dtype."), return ge::GRAPH_FAILED);

    bool isInvalidType = (gradDtype != ge::DT_FLOAT) && (gradDtype != ge::DT_BF16) && (gradDtype != ge::DT_FLOAT16);
    OP_CHECK_IF(
        isInvalidType, OP_LOGE(nodeName, "Input grad, var, m, v, s dtype only support fp16, fp32 and bf16."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        stepDtype != ge::DT_INT64, OP_LOGE(nodeName, "Input step dtype only support int64."), return ge::GRAPH_FAILED);

    uint32_t tilingKey = DTYPE_BF16_KEY;
    if (gradDtype == ge::DT_FLOAT) {
        tilingKey = DTYPE_FP32_KEY;
    } else if (gradDtype == ge::DT_FLOAT16) {
        tilingKey = DTYPE_FP16_KEY;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(gert::TilingContext* context)
{
    auto shapeInput = context->GetInputShape(INPUT_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInput);
    auto gradSize = shapeInput->GetStorageShape().GetShapeSize();

    shapeInput = context->GetInputShape(INPUT_VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInput);
    auto varSize = shapeInput->GetStorageShape().GetShapeSize();

    shapeInput = context->GetInputShape(INPUT_M_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInput);
    auto mSize = shapeInput->GetStorageShape().GetShapeSize();

    shapeInput = context->GetInputShape(INPUT_V_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInput);
    auto vSize = shapeInput->GetStorageShape().GetShapeSize();

    shapeInput = context->GetInputShape(INPUT_S_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInput);
    auto sSize = shapeInput->GetStorageShape().GetShapeSize();

    shapeInput = context->GetInputShape(INPUT_STEP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInput);

    bool isDiffSize = gradSize != varSize || gradSize != mSize || gradSize != vSize || gradSize != sSize;
    OP_CHECK_IF(
        isDiffSize, OP_LOGE(context, "Input grad, var, m, v, s should keep equal data."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCompute(gert::TilingContext* context)
{
    auto nodeName = context;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((coreNum <= 0), OP_LOGE(nodeName, "Fail to get core num."), return ge::GRAPH_FAILED);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    uint64_t totalDataNum = context->GetInputShape(INPUT_V_IDX)->GetStorageShape().GetShapeSize();
    uint64_t dtypeSize =
        context->GetInputDesc(INPUT_V_IDX)->GetDataType() == ge::DT_FLOAT ? FP32_DTYPE_SIZE : FP16_BF16_DTYPE_SIZE;
    uint64_t frontCoreNum = totalDataNum % coreNum != 0 ? totalDataNum % coreNum : coreNum;
    uint64_t tailCoreNum = totalDataNum <= coreNum ? 0 : coreNum - frontCoreNum;
    uint64_t blockDim = frontCoreNum + tailCoreNum;
    uint64_t coreCalcNum = (totalDataNum + coreNum - 1) / coreNum;

    fusedEmaAdamTiling.set_frontCoreNum(frontCoreNum);
    fusedEmaAdamTiling.set_tailCoreNum(tailCoreNum);
    fusedEmaAdamTiling.set_coreCalcNum(coreCalcNum);
    context->SetBlockDim(blockDim);

    uint64_t tBuffersize = TBUFFER_NUM * BYTE_ONE_BLK;
    uint64_t bufferSize = ubSize - tBuffersize;
    uint64_t coreOnesize =
        (dtypeSize == FP32_DTYPE_SIZE) ? dtypeSize * QUEUE_NUM : (dtypeSize + FP32_DTYPE_SIZE) * QUEUE_NUM;
    if (fusedEmaAdamTiling.get_mode() == 1) {
        coreOnesize -= (dtypeSize == FP32_DTYPE_SIZE) ? dtypeSize : (dtypeSize + FP32_DTYPE_SIZE);
    }
    uint64_t alignSize = dtypeSize == FP32_DTYPE_SIZE ? ONE_BLK_NUM_FP32 : ONE_BLK_NUM;
    uint64_t coreCalcMax = bufferSize / coreOnesize / alignSize * alignSize;
    uint64_t loopNum = coreCalcMax < coreCalcNum ? (coreCalcNum + coreCalcMax - 1) / coreCalcMax : 1;
    uint64_t frontCalcExtra = loopNum == 1 ? coreCalcNum : coreCalcNum - coreCalcMax * (loopNum - 1);
    uint64_t tailCalcExtra = frontCalcExtra - 1;

    fusedEmaAdamTiling.set_coreCalcMax(coreCalcMax);
    fusedEmaAdamTiling.set_loopNum(loopNum);
    fusedEmaAdamTiling.set_frontCalcExtra(frontCalcExtra);
    fusedEmaAdamTiling.set_tailCalcExtra(tailCalcExtra);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4ApplyFusedEmaAdam(gert::TilingContext* context)
{
    auto nodeName = context;
    OP_LOGD(nodeName, "Tiling4ApplyFusedEmaAdam running begin");

    OP_CHECK_IF(
        GetTilingAttr(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "GetTilingAttr failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckInputDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "CheckInputDtype failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        CheckInputShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "CheckInputShape failed."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        TilingCompute(context) != ge::GRAPH_SUCCESS, OP_LOGE(nodeName, "TilingCompute failed."),
        return ge::GRAPH_FAILED);

    fusedEmaAdamTiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(fusedEmaAdamTiling.GetDataSize());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    PrintInfo(context);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4ApplyFusedEmaAdam([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ApplyFusedEmaAdamCompileInfo {
};

IMPL_OP_OPTILING(ApplyFusedEmaAdam)
    .Tiling(Tiling4ApplyFusedEmaAdam)
    .TilingParse<ApplyFusedEmaAdamCompileInfo>(TilingPrepare4ApplyFusedEmaAdam);
} // namespace optiling