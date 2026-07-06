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
 * \file fused_sgd_tiling.cpp
 * \brief
 */
#include "fused_sgd_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

using namespace std;
namespace optiling {
constexpr uint32_t INPUT_PARAMS_IDX = 0;
constexpr uint32_t INPUT_GRADS_IDX = 1;
constexpr uint32_t INPUT_MOMENTUM_BUFFER_IDX = 2;
constexpr uint32_t INPUT_GRAD_SCALE_IDX = 3;
constexpr uint32_t ATTR_WEIGHT_DECAY_IDX = 0;
constexpr uint32_t ATTR_MOMENTUM_IDX = 1;
constexpr uint32_t ATTR_LR_IDX = 2;
constexpr uint32_t ATTR_DAMPENING_IDX = 3;
constexpr uint32_t ATTR_NESTEROV_IDX = 4;
constexpr uint32_t ATTR_MAXIMIZE_IDX = 5;
constexpr uint32_t ATTR_IS_FIRST_STEP_IDX = 6;
constexpr uint32_t ONE_BLK_NUM = 16;
constexpr uint32_t ONE_BLK_NUM_FP32 = 8;
constexpr uint32_t BYTE_ONE_BLK = 32;
constexpr uint32_t TBUFFER_NUM = 3;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint32_t FP32_DTYPE_SIZE = 4;

std::string FusedSgdTiling::TilingDataToString() const
{
    return "weightDecay = " + std::to_string(weightDecay_) + ", momentum = " + std::to_string(momentum_) +
           ", lr = " + std::to_string(lr_) + ", dampening = " + std::to_string(dampening_) +
           ", nesterov = " + std::to_string(nesterov_) + ", maximize = " + std::to_string(maximize_) +
           ", isFirstStep = " + std::to_string(isFirstStep_) + ", useGradScale = " + std::to_string(useGradScale_) +
           ", useMomentum = " + std::to_string(useMomentum_) + ", tensorNum = " + std::to_string(tensorNum_) +
           ", tensorsPerCore = " + std::to_string(tensorsPerCore_) + ", usedCoreNum = " + std::to_string(usedCoreNum_) +
           ", coreCalcMax = " + std::to_string(coreCalcMax_);
}

// 获取硬件信息
ge::graphStatus FusedSgdTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0, OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSgdTiling::GetAttrInfo()
{
    // 获取属性
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* attrWeightDecay = attrs->GetAttrPointer<float>(ATTR_WEIGHT_DECAY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrWeightDecay);
    weightDecay_ = static_cast<float>(*attrWeightDecay);
    const float* attrMomentum = attrs->GetAttrPointer<float>(ATTR_MOMENTUM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrMomentum);
    momentum_ = static_cast<float>(*attrMomentum);
    const float* attrLr = attrs->GetAttrPointer<float>(ATTR_LR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrLr);
    lr_ = static_cast<float>(*attrLr);
    const float* attrDampening = attrs->GetAttrPointer<float>(ATTR_DAMPENING_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrDampening);
    dampening_ = static_cast<float>(*attrDampening);
    const bool* attrNesterov = attrs->GetAttrPointer<bool>(ATTR_NESTEROV_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrNesterov);
    nesterov_ = static_cast<uint32_t>(*attrNesterov ? 1 : 0);
    const bool* attrMaximize = attrs->GetAttrPointer<bool>(ATTR_MAXIMIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrMaximize);
    maximize_ = static_cast<uint32_t>(*attrMaximize ? 1 : 0);
    const bool* attrIsFirstStep = attrs->GetAttrPointer<bool>(ATTR_IS_FIRST_STEP_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrIsFirstStep);
    isFirstStep_ = static_cast<uint32_t>(*attrIsFirstStep ? 1 : 0);

    return ge::GRAPH_SUCCESS;
}

void FusedSgdTiling::CheckOptionalInputs()
{
    // 判断gradScale为空
    auto shapeInput = context_->GetOptionalInputTensor(INPUT_GRAD_SCALE_IDX);
    if (shapeInput != nullptr) {
        const gert::Shape& inputShapeGradScale = shapeInput->GetStorageShape();
        uint32_t gradScaleDims = inputShapeGradScale.GetDimNum();
        bool flag = true;
        if (gradScaleDims > 0) {
            for (uint32_t i = 0; i < gradScaleDims; i++) {
                int64_t dimValue = inputShapeGradScale.GetDim(i);
                if (dimValue == 0) {
                    flag = false;
                    break;
                }
            }
        }
        if (flag) {
            useGradScale_ = 1;
        } else {
            useGradScale_ = 0;
        }
    } else {
        useGradScale_ = 0;
    }

    // 判断tensorlist为空
    auto momentumBufferListInput = context_->GetDynamicInputShape(INPUT_MOMENTUM_BUFFER_IDX, 0);
    if (momentumBufferListInput != nullptr) {
        const gert::Shape& inputShapeMomentumBufferList = momentumBufferListInput->GetStorageShape();
        uint32_t momentumBufferListDims = inputShapeMomentumBufferList.GetDimNum();
        bool flag = true;
        if (momentumBufferListDims < 1) {
            flag = false;
        }
        if (flag) {
            for (uint32_t i = 0; i < momentumBufferListDims; i++) {
                int64_t dimValue = inputShapeMomentumBufferList.GetDim(i);
                if (dimValue == 0) {
                    flag = false;
                    break;
                }
            }
        }
        if (flag) {
            useMomentum_ = 1;
        } else {
            useMomentum_ = 0;
        }
    } else {
        useMomentum_ = 0;
    }
}

static ge::graphStatus CheckInputDtype(gert::TilingContext* context, uint32_t useMomentum_)
{
    auto dtypeInput = context->GetDynamicInputDesc(INPUT_PARAMS_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto paramsDtype = dtypeInput->GetDataType();

    dtypeInput = context->GetDynamicInputDesc(INPUT_GRADS_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto gradsDtype = dtypeInput->GetDataType();

    bool isDiffDtype = (paramsDtype != gradsDtype);
    ge::DataType momentumDtype;
    if (useMomentum_) {
        dtypeInput = context->GetDynamicInputDesc(INPUT_MOMENTUM_BUFFER_IDX, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
        momentumDtype = dtypeInput->GetDataType();
        isDiffDtype = isDiffDtype || (paramsDtype != momentumDtype);
    }

    if (isDiffDtype) {
        std::string dtypeMsg = Ops::Base::ToString(paramsDtype) + ", " + Ops::Base::ToString(gradsDtype);
        if (!useMomentum_) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "params, grads", dtypeMsg.c_str(),
                                                   "params, grads should have the same dtype");
        } else {
            dtypeMsg = dtypeMsg + " and " + Ops::Base::ToString(momentumDtype);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "params, grads and momentum_buffer_list",
                                                   dtypeMsg.c_str(),
                                                   "params, grads and momentum_buffer_list should have the same dtype");
        }
        return ge::GRAPH_FAILED;
    }

    bool isInvalidType = (paramsDtype != ge::DT_FLOAT) && (paramsDtype != ge::DT_BF16) &&
                         (paramsDtype != ge::DT_FLOAT16);
    if (isInvalidType) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "params/grads/momentum_buffer_list",
                                  Ops::Base::ToString(paramsDtype).c_str(), "float16, bfloat16 and float");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSgdTiling::GetInputTensorInfo()
{
    auto computeNodeInfo = context_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, computeNodeInfo);

    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_PARAMS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, anchorInstanceInfo);
    tensorNum_ = static_cast<uint64_t>(anchorInstanceInfo->GetInstanceNum());
    if (tensorNum_ == 0) {
        OP_LOGE(context_, "tensor num can not be 0");
        return ge::GRAPH_FAILED;
    }

    // 检查可选输入是否为空
    CheckOptionalInputs();

    for (uint64_t i = 0; i < tensorNum_; i++) {
        auto paramsShapePtr = context_->GetDynamicInputShape(INPUT_PARAMS_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, paramsShapePtr);
        auto gradsShapePtr = context_->GetDynamicInputShape(INPUT_GRADS_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, gradsShapePtr);

        gert::Shape paramsShape = paramsShapePtr->GetStorageShape();
        gert::Shape gradsShape = gradsShapePtr->GetStorageShape();
        bool isDiffSize = paramsShape != gradsShape;

        gert::Shape momentumShape;
        if (useMomentum_) {
            auto momentumShapePtr = context_->GetDynamicInputShape(INPUT_MOMENTUM_BUFFER_IDX, i);
            OP_CHECK_NULL_WITH_CONTEXT(context_, momentumShapePtr);
            momentumShape = momentumShapePtr->GetStorageShape();
            isDiffSize = isDiffSize || paramsShape != momentumShape;
        }
        if (isDiffSize) {
            std::string shapesMsg = Ops::Base::ToString(paramsShape) + ", " + Ops::Base::ToString(gradsShape);
            if (!useMomentum_) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), "params, grads", shapesMsg.c_str(),
                                                       "params, grads should have the same shape");
            } else {
                shapesMsg = shapesMsg + " and " + Ops::Base::ToString(momentumShape);
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context_->GetNodeName(), "params, grads and momentum_buffer_list", shapesMsg.c_str(),
                    "params, grads and momentum_buffer_list should have the same shape");
            }
            return ge::GRAPH_FAILED;
        }
    }

    return CheckInputDtype(context_, useMomentum_);
}

ge::graphStatus FusedSgdTiling::CalculateOutputInfo()
{
    usedCoreNum_ = tensorNum_ < static_cast<uint64_t>(coreNum_) ? tensorNum_ : static_cast<uint64_t>(coreNum_);
    tensorsPerCore_ = static_cast<uint32_t>((tensorNum_ + usedCoreNum_ - 1) / usedCoreNum_);

    dtypeSize_ = context_->GetDynamicInputDesc(INPUT_PARAMS_IDX, 0)->GetDataType() == ge::DT_FLOAT ?
                     FP32_DTYPE_SIZE :
                     FP16_BF16_DTYPE_SIZE;
    uint64_t tBuffersize = BUFFER_NUM * BYTE_ONE_BLK;
    uint64_t bufferSize = ubSize_ - tBuffersize;
    // 计算处理一个元素所需的ub大小
    uint64_t coreOnesize;
    if (dtypeSize_ == FP32_DTYPE_SIZE) {
        coreOnesize = FP32_DTYPE_SIZE * 3 * 2 * BUFFER_NUM;
    } else {
        coreOnesize = ((dtypeSize_ + FP32_DTYPE_SIZE) * 3 + FP32_DTYPE_SIZE * 3) * BUFFER_NUM;
    }
    uint64_t alignSize = dtypeSize_ == FP32_DTYPE_SIZE ? ONE_BLK_NUM_FP32 : ONE_BLK_NUM;
    OP_LOGI(context_, "bufferSize = %lu", bufferSize);
    OP_LOGI(context_, "coreOnesize = %lu", coreOnesize);
    OP_LOGI(context_, "alignSize = %lu", alignSize);
    // 计算ub一次最多能处理的数据量
    coreCalcMax_ = bufferSize / coreOnesize / alignSize * alignSize;
    OP_LOGI(context_, "coreCalcMax_ = %lu", coreCalcMax_);

    return ge::GRAPH_SUCCESS;
}

void FusedSgdTiling::SetTilingData(FusedSgdTilingData* tilingData)
{
    tilingData->weightDecay = weightDecay_;
    tilingData->momentum = momentum_;
    tilingData->lr = lr_;
    tilingData->dampening = dampening_;
    tilingData->nesterov = nesterov_;
    tilingData->maximize = maximize_;
    tilingData->isFirstStep = isFirstStep_;
    tilingData->useGradScale = useGradScale_;
    tilingData->useMomentum = useMomentum_;
    tilingData->tensorNum = tensorNum_;
    tilingData->tensorsPerCore = tensorsPerCore_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->coreCalcMax = coreCalcMax_;

    size_t* workspaceSize = context_->GetWorkspaceSizes(1);
    *workspaceSize = sysWorkspaceSize_;
    context_->SetTilingKey(0);
    context_->SetBlockDim(usedCoreNum_);
}

ge::graphStatus Tiling4FusedSgd(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4FusedSgd");
    FusedSgdTiling tiling(context);
    OP_CHECK_IF(tiling.GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.GetAttrInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.GetInputTensorInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetInputTensorInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.CalculateOutputInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateOutputInfo error"),
                return ge::GRAPH_FAILED);

    FusedSgdTilingData* tilingData = context->GetTilingData<FusedSgdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(memset_s(tilingData, sizeof(FusedSgdTilingData), 0, sizeof(FusedSgdTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling.SetTilingData(tilingData);
    OP_LOGD(context, "tiling data: %s", tiling.TilingDataToString().c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4FusedSgd([[maybe_unused]] gert::TilingParseContext* context) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(FusedSgd).Tiling(Tiling4FusedSgd).TilingParse<FusedSgdCompileInfo>(TilingPrepare4FusedSgd);
} // namespace optiling