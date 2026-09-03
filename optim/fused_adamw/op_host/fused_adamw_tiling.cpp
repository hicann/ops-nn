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
 * \file fused_adamw_tiling.cpp
 * \brief
 */

#include "fused_adamw_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_infos_def.h"
#include <cmath>

using namespace std;
namespace optiling {
constexpr uint32_t INPUT_PARAMS_IDX = 0;
constexpr uint32_t INPUT_GRADS_IDX = 1;
constexpr uint32_t INPUT_EXP_AVGS_IDX = 2;
constexpr uint32_t INPUT_EXP_AVG_SQS_IDX = 3;
constexpr uint32_t INPUT_MAX_EXP_AVG_SQS_IDX = 4;
constexpr uint32_t INPUT_STATE_STEPS_IDX = 5;
constexpr uint32_t INPUT_GRAD_SCALE_IDX = 6;
constexpr uint32_t INPUT_FOUND_INF_IDX = 7;
constexpr uint32_t ATTR_LR_IDX = 0;
constexpr uint32_t ATTR_BETA1_IDX = 1;
constexpr uint32_t ATTR_BETA2_IDX = 2;
constexpr uint32_t ATTR_WEIGHT_DECAY_IDX = 3;
constexpr uint32_t ATTR_EPS_IDX = 4;
constexpr uint32_t ATTR_AMSGRAD_IDX = 5;
constexpr uint32_t ATTR_MAXIMIZE_IDX = 6;
constexpr uint32_t ONE_BLK_NUM = 16;
constexpr uint32_t ONE_BLK_NUM_FP32 = 8;
constexpr uint32_t BYTE_ONE_BLK = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BUFFER_NUM_8 = 8;
constexpr uint32_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint32_t FP32_DTYPE_SIZE = 4;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t TENSOR_COUNT_BASE = 5;        // params, grads, exp_avg, exp_avg_sq, state_steps
constexpr uint32_t TENSOR_COUNT_AMSGRAD = 6;     // + max_exp_avg_sq
constexpr uint32_t TENSOR_COUNT_BASE_OUT = 3;    // params, exp_avg, exp_avg_sq
constexpr uint32_t TENSOR_COUNT_AMSGRAD_OUT = 4; // + max_exp_avg_sq
constexpr uint32_t UB_EMPTY = 1000;              // ub预留1000字节

std::string FusedAdamWTiling::TilingDataToString() const
{
    return "lr = " + std::to_string(lr_) + ", beta1 = " + std::to_string(beta1_) +
           ", beta2 = " + std::to_string(beta2_) + ", weightDecay = " + std::to_string(weightDecay_) +
           ", eps = " + std::to_string(eps_) + ", amsgrad = " + std::to_string(amsgrad_) +
           ", maximize = " + std::to_string(maximize_) + ", useGradScale = " + std::to_string(useGradScale_) +
           ", useFoundInf = " + std::to_string(useFoundInf_) + ", tensorNum = " + std::to_string(tensorNum_) +
           ", tensorsPerCore = " + std::to_string(tensorsPerCore_) + ", usedCoreNum = " + std::to_string(usedCoreNum_) +
           ", coreCalcMax = " + std::to_string(coreCalcMax_);
}

ge::graphStatus FusedAdamWTiling::GetPlatformInfo()
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

ge::graphStatus FusedAdamWTiling::GetAttrInfo()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const float* attrLr = attrs->GetAttrPointer<float>(ATTR_LR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrLr);
    lr_ = static_cast<float>(*attrLr);

    const float* attrBeta1 = attrs->GetAttrPointer<float>(ATTR_BETA1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrBeta1);
    beta1_ = static_cast<float>(*attrBeta1);

    const float* attrBeta2 = attrs->GetAttrPointer<float>(ATTR_BETA2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrBeta2);
    beta2_ = static_cast<float>(*attrBeta2);

    const float* attrWeightDecay = attrs->GetAttrPointer<float>(ATTR_WEIGHT_DECAY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrWeightDecay);
    weightDecay_ = static_cast<float>(*attrWeightDecay);

    const float* attrEps = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrEps);
    eps_ = static_cast<float>(*attrEps);

    const bool* attrAmsgrad = attrs->GetAttrPointer<bool>(ATTR_AMSGRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrAmsgrad);
    amsgrad_ = static_cast<uint32_t>(*attrAmsgrad ? 1 : 0);

    const bool* attrMaximize = attrs->GetAttrPointer<bool>(ATTR_MAXIMIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrMaximize);
    maximize_ = static_cast<uint32_t>(*attrMaximize ? 1 : 0);

    return ge::GRAPH_SUCCESS;
}

void FusedAdamWTiling::CheckOptionalInputs()
{
    // 判断grad_scale是否为空
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

    // 判断found_inf是否为空
    auto shapeFoundInf = context_->GetOptionalInputTensor(INPUT_FOUND_INF_IDX);
    if (shapeFoundInf != nullptr) {
        const gert::Shape& inputShapeFoundInf = shapeFoundInf->GetStorageShape();
        uint32_t foundInfDims = inputShapeFoundInf.GetDimNum();
        bool flag = true;
        if (foundInfDims > 0) {
            for (uint32_t i = 0; i < foundInfDims; i++) {
                int64_t dimValue = inputShapeFoundInf.GetDim(i);
                if (dimValue == 0) {
                    flag = false;
                    break;
                }
            }
        }
        if (flag) {
            useFoundInf_ = 1;
        } else {
            useFoundInf_ = 0;
        }
    } else {
        useFoundInf_ = 0;
    }
}

static ge::graphStatus CheckInputDtype(gert::TilingContext* context, uint32_t amsgrad_)
{
    auto dtypeInput = context->GetDynamicInputDesc(INPUT_PARAMS_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInput);
    auto paramsDtype = dtypeInput->GetDataType();

    auto checkDtype = [&](uint32_t idx, const char* nameOfOps) -> ge::graphStatus {
        auto dtypeInputInner = context->GetDynamicInputDesc(idx, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, dtypeInputInner);
        auto dtype = dtypeInputInner->GetDataType();
        if (dtype != paramsDtype) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), nameOfOps,
                                                   Ops::Base::ToString(dtype).c_str(),
                                                   "should have the same dtype as params");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    if (checkDtype(INPUT_GRADS_IDX, "grads") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (checkDtype(INPUT_EXP_AVGS_IDX, "exp_avgs") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (checkDtype(INPUT_EXP_AVG_SQS_IDX, "exp_avg_sqs") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (amsgrad_ != 0) {
        if (checkDtype(INPUT_MAX_EXP_AVG_SQS_IDX, "max_exp_avg_sqs") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    bool isInvalidType = (paramsDtype != ge::DT_FLOAT) && (paramsDtype != ge::DT_BF16) &&
                         (paramsDtype != ge::DT_FLOAT16);
    if (isInvalidType) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "params/grads/exp_avgs/exp_avg_sqs",
                                  Ops::Base::ToString(paramsDtype).c_str(), "float16, bfloat16 and float");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedAdamWTiling::GetInputTensorInfo()
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

    CheckOptionalInputs();

    for (uint64_t i = 0; i < tensorNum_; i++) {
        auto paramsShapePtr = context_->GetDynamicInputShape(INPUT_PARAMS_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, paramsShapePtr);
        gert::Shape paramsShape = paramsShapePtr->GetStorageShape();

        auto checkShape = [&](uint32_t idx, const char* shapeNameOfOps) -> ge::graphStatus {
            auto shapePtr = context_->GetDynamicInputShape(idx, i);
            OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
            if (shapePtr->GetStorageShape() != paramsShape) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context_->GetNodeName(), shapeNameOfOps,
                                                       Ops::Base::ToString(shapePtr->GetStorageShape()).c_str(),
                                                       "should have the same shape as params");
                return ge::GRAPH_FAILED;
            }
            return ge::GRAPH_SUCCESS;
        };

        if (checkShape(INPUT_GRADS_IDX, "grads") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (checkShape(INPUT_EXP_AVGS_IDX, "exp_avgs") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (checkShape(INPUT_EXP_AVG_SQS_IDX, "exp_avg_sqs") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (amsgrad_ != 0) {
            if (checkShape(INPUT_MAX_EXP_AVG_SQS_IDX, "max_exp_avg_sqs") != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    }

    return CheckInputDtype(context_, amsgrad_);
}

ge::graphStatus FusedAdamWTiling::CalculateOutputInfo()
{
    usedCoreNum_ = tensorNum_ < static_cast<uint64_t>(coreNum_) ? tensorNum_ : static_cast<uint64_t>(coreNum_);
    tensorsPerCore_ = static_cast<uint32_t>((tensorNum_ + usedCoreNum_ - 1) / usedCoreNum_);
    usedRealCoreNum_ = static_cast<uint32_t>((tensorNum_ + tensorsPerCore_ - 1) / tensorsPerCore_);
    lastCoreTensor_ = tensorNum_ - (usedRealCoreNum_ - 1) * tensorsPerCore_;

    dtypeSize_ = context_->GetDynamicInputDesc(INPUT_PARAMS_IDX, 0)->GetDataType() == ge::DT_FLOAT ?
                     FP32_DTYPE_SIZE :
                     FP16_BF16_DTYPE_SIZE;
    uint64_t tBuffersize = BUFFER_NUM_8 * BYTE_ONE_BLK;
    uint64_t bufferSize = ubSize_ - tBuffersize - UB_EMPTY;

    uint32_t tensorCount = amsgrad_ ? TENSOR_COUNT_AMSGRAD : TENSOR_COUNT_BASE;
    uint32_t tensorCountOut = amsgrad_ ? TENSOR_COUNT_AMSGRAD_OUT : TENSOR_COUNT_BASE_OUT;

    // 计算处理一个元素所需的UB大小
    uint64_t coreOnesize;
    if (dtypeSize_ == FP32_DTYPE_SIZE) {
        // inQue: tensorCount * sizeof(float) * BUFFER_NUM
        // outQue: tensorCountOut * sizeof(float) * BUFFER_NUM
        coreOnesize = (tensorCount + tensorCountOut) * FP32_DTYPE_SIZE * BUFFER_NUM;
    } else {
        // inQue: tensorCount * sizeof(T) + tensorCount * sizeof(float)  (原始 + FP32 cast)
        // outQue: tensorCountOut * sizeof(float)
        coreOnesize = (tensorCount * (dtypeSize_ + FP32_DTYPE_SIZE) + tensorCountOut * FP32_DTYPE_SIZE) * BUFFER_NUM;
    }
    uint64_t alignSize = dtypeSize_ == FP32_DTYPE_SIZE ? ONE_BLK_NUM_FP32 : ONE_BLK_NUM;
    coreCalcMax_ = bufferSize / coreOnesize / alignSize * alignSize;

    return ge::GRAPH_SUCCESS;
}

void FusedAdamWTiling::SetTilingData(FusedAdamWTilingData* tilingData)
{
    tilingData->lr = lr_;
    tilingData->beta1 = beta1_;
    tilingData->beta2 = beta2_;
    tilingData->weightDecay = weightDecay_;
    tilingData->eps = eps_;
    tilingData->amsgrad = amsgrad_;
    tilingData->maximize = maximize_;
    tilingData->useGradScale = useGradScale_;
    tilingData->useFoundInf = useFoundInf_;
    tilingData->tensorNum = tensorNum_;
    tilingData->tensorsPerCore = tensorsPerCore_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->usedRealCoreNum = usedRealCoreNum_;
    tilingData->lastCoreTensor = lastCoreTensor_;
    tilingData->coreCalcMax = coreCalcMax_;
    tilingData->stepCount = 0;

    size_t* workspaceSize = context_->GetWorkspaceSizes(1);
    *workspaceSize = sysWorkspaceSize_;
    context_->SetTilingKey(0);
    context_->SetBlockDim(usedRealCoreNum_);
}

ge::graphStatus Tiling4FusedAdamW(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4FusedAdamW");
    FusedAdamWTiling tiling(context);
    OP_CHECK_IF(tiling.GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.GetAttrInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.GetInputTensorInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetInputTensorInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.CalculateOutputInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateOutputInfo error"),
                return ge::GRAPH_FAILED);

    FusedAdamWTilingData* tilingData = context->GetTilingData<FusedAdamWTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(memset_s(tilingData, sizeof(FusedAdamWTilingData), 0, sizeof(FusedAdamWTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling.SetTilingData(tilingData);
    OP_LOGD(context, "tiling data: %s", tiling.TilingDataToString().c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4FusedAdamW([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedAdamw).Tiling(Tiling4FusedAdamW).TilingParse<FusedAdamWCompileInfo>(TilingPrepare4FusedAdamW);
} // namespace optiling
