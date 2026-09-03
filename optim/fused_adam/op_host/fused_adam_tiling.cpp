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
 * \file fused_adam_tiling.cpp
 * \brief
 */

#include "../op_kernel/fused_adam_tiling_data.h"
#include "fused_adam_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "platform/platform_infos_def.h"
#include <algorithm>
#include <graph/utils/type_utils.h>

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
constexpr uint32_t FP16_BF16_DTYPE_SIZE = 2;
constexpr uint32_t FP32_DTYPE_SIZE = 4;
static constexpr int32_t MAX_SUPPORT_DIM_NUMS = 8;
static constexpr int32_t BLOCK_SIZE = 32;
static constexpr int64_t SINGLE_CORE_PROCESS_BYTES_LOWER_BOUND = 1024;
static constexpr uint64_t TILING_KEY_FLOAT = 0;
static constexpr uint64_t TILING_KEY_HALF = 1;
static constexpr uint64_t TILING_KEY_BF16 = 2;

std::string FusedAdamTiling::TilingDataToString()
{
    std::string ans = "lr = " + std::to_string(lr_) + ", beta1 = " + std::to_string(beta1_) +
                      ", beta2 = " + std::to_string(beta2_) + ", weightDecay = " + std::to_string(weightDecay_) +
                      ", eps = " + std::to_string(eps_) + ", amsgrad = " + std::to_string(amsgrad_) +
                      ", maximize = " + std::to_string(maximize_) +
                      ", useGradScale = " + std::to_string(useGradScale_) +
                      ", useFoundInf = " + std::to_string(useFoundInf_) +
                      ", tensorNum = " + std::to_string(tensorNum_) + ", usedCoreNum = " + std::to_string(usedCoreNum_);
    ans += "\ni\tcnt\t\n";
    for (uint32_t i = 0; i < tensorNum_; i++) {
        ans += std::to_string(i) + "\t" + std::to_string(tensorDataCountList_[i]) + "\n";
    }
    ans += "i\tstart\tend\tso\teo\n";
    for (uint32_t i = 0; i < usedCoreNum_; i++) {
        ans += std::to_string(i) + "\t" + std::to_string(tensorStartList_[i]) + "\t" +
               std::to_string(tensorEndList_[i]) + "\t" + std::to_string(tensorStartOffsetList_[i]) + "\t" +
               std::to_string(tensorEndOffsetList_[i]) + "\n";
    }
    ans += "TilingKey: " + std::to_string(CalcTilingKey()) + "\n";
    return ans;
}

// get coreNum_, ubSize_, sysWorkspaceSize_
ge::graphStatus FusedAdamTiling::GetPlatformInfo()
{
    auto platformInfo = tilingContext_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(tilingContext_, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0, OP_LOGE(tilingContext_, "ubSize is 0"), return ge::GRAPH_FAILED);
    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedAdamTiling::GetAttrInfo()
{
    auto* attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    const float* attrLr = attrs->GetAttrPointer<float>(ATTR_LR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrLr);
    lr_ = static_cast<float>(*attrLr);

    const float* attrBeta1 = attrs->GetAttrPointer<float>(ATTR_BETA1_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrBeta1);
    beta1_ = static_cast<float>(*attrBeta1);

    const float* attrBeta2 = attrs->GetAttrPointer<float>(ATTR_BETA2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrBeta2);
    beta2_ = static_cast<float>(*attrBeta2);

    const float* attrWeightDecay = attrs->GetAttrPointer<float>(ATTR_WEIGHT_DECAY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrWeightDecay);
    weightDecay_ = static_cast<float>(*attrWeightDecay);

    const float* attrEps = attrs->GetAttrPointer<float>(ATTR_EPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrEps);
    eps_ = static_cast<float>(*attrEps);

    const bool* attrAmsgrad = attrs->GetAttrPointer<bool>(ATTR_AMSGRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrAmsgrad);
    amsgrad_ = static_cast<uint32_t>(*attrAmsgrad ? 1 : 0);

    const bool* attrMaximize = attrs->GetAttrPointer<bool>(ATTR_MAXIMIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrMaximize);
    maximize_ = static_cast<uint32_t>(*attrMaximize ? 1 : 0);

    return ge::GRAPH_SUCCESS;
}

uint32_t inline FusedAdamTiling::GetOptionalInput(uint32_t OPTIONAL_INPUT_IDX)
{
    auto tempInput = tilingContext_->GetOptionalInputTensor(OPTIONAL_INPUT_IDX);
    if (tempInput != nullptr) {
        const gert::Shape& inputShape = tempInput->GetStorageShape();
        uint32_t dims = inputShape.GetDimNum();
        bool flag = true;
        if (dims > 0) {
            for (uint32_t i = 0; i < dims; i++) {
                int64_t dimValue = inputShape.GetDim(i);
                if (dimValue == 0) {
                    flag = false;
                    break;
                }
            }
        }
        if (flag) {
            return 1;
        } else {
            return 0;
        }
    } else {
        return 0;
    }
}

ge::graphStatus FusedAdamTiling::CheckShapeAllPositive(const gert::Shape& shape, uint32_t idx)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        tilingContext_->GetNodeName(), "x", std::to_string(shape.GetDim(i)).c_str(),
                        "All axes of tensors in the tensor list must be 0 or positive numbers. Currently, the " +
                            std::to_string(i) + "th axis of the " + std::to_string(idx) +
                            "th tensor in the tensor list does not meet the condition"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// check InputTensorList[paramIdx][tensorIdx] if is same with given shape and dtype.
ge::graphStatus FusedAdamTiling::CheckShapeAndDType(uint32_t paramIdx, uint32_t tensorIdx,
                                                    const gert::Shape& paramsShape, ge::DataType paramsDtype,
                                                    const char* name)
{
    auto shapePtr = tilingContext_->GetDynamicInputShape(paramIdx, tensorIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, shapePtr);
    if (shapePtr->GetStorageShape() != paramsShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(tilingContext_->GetNodeName(), name,
                                               Ops::Base::ToString(shapePtr->GetStorageShape()).c_str(),
                                               "should have the same shape as params");
        return ge::GRAPH_FAILED;
    }
    auto dtypeInputInner = tilingContext_->GetDynamicInputDesc(paramIdx, tensorIdx);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, dtypeInputInner);
    auto dtype = dtypeInputInner->GetDataType();
    if (dtype != paramsDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(tilingContext_->GetNodeName(), name, Ops::Base::ToString(dtype).c_str(),
                                               "should have the same dtype as params");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus FusedAdamTiling::CheckStateSteps()
{
    auto computeNodeInfo = tilingContext_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, computeNodeInfo);

    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_STATE_STEPS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, anchorInstanceInfo);

    OP_CHECK_IF(tensorNum_ != anchorInstanceInfo->GetInstanceNum(),
                OP_LOGE(tilingContext_, "The stateSteps tensorList length is not match params."),
                return ge::GRAPH_FAILED);
    for (uint32_t i = 0; i < tensorNum_; i++) {
        auto tempDesc = tilingContext_->GetDynamicInputDesc(INPUT_STATE_STEPS_IDX, i);
        OP_CHECK_IF(tempDesc == nullptr, OP_LOGE(tilingContext_, "The stateSteps %u desc is null.", i),
                    return ge::GRAPH_FAILED);
        auto tempDtype = tempDesc->GetDataType();
        // Determine whether all data types are consistent.
        if (scalarType_ == ge::DT_UNDEFINED) {
            scalarType_ = tempDtype;
            if ((scalarType_ != ge::DT_FLOAT) && (scalarType_ != ge::DT_INT64)) {
                OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "stepStates",
                                          Ops::Base::ToString(scalarType_).c_str(), "float and int64");
                return ge::GRAPH_FAILED;
            }
        } else if (tempDtype != scalarType_) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                tilingContext_->GetNodeName(), "params", ge::TypeUtils::DataTypeToSerialString(scalarType_).c_str(),
                ("The dtypes of all tensors in the tensor list must be the same. "
                 "Currently, the dtype of the " +
                 std::to_string(i) + "th tensor is inconsistent with that (" +
                 ge::TypeUtils::DataTypeToSerialString(scalarType_) + ") of other tensors")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedAdamTiling::GetInputTensorInfo()
{
    auto computeNodeInfo = tilingContext_->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, computeNodeInfo);

    auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(INPUT_PARAMS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, anchorInstanceInfo);

    tensorNum_ = anchorInstanceInfo->GetInstanceNum();
    OP_CHECK_IF(
        tensorNum_ > MAX_TENSOR_CONT_950 || tensorNum_ <= 0,
        OP_LOGE_FOR_INVALID_TENSORNUM(tilingContext_->GetNodeName(), "params", static_cast<int64_t>(tensorNum_),
                                      ("within the range [1, " + std::to_string(MAX_TENSOR_CONT_950) + "]").c_str()),
        return ge::GRAPH_FAILED);

    useGradScale_ = GetOptionalInput(INPUT_GRAD_SCALE_IDX);
    useFoundInf_ = GetOptionalInput(INPUT_FOUND_INF_IDX);

    totalDataCount_ = 0;
    dataType_ = ge::DT_UNDEFINED;
    for (uint32_t i = 0; i < tensorNum_; i++) {
        auto tempDesc = tilingContext_->GetDynamicInputDesc(INPUT_PARAMS_IDX, i);
        OP_CHECK_IF(tempDesc == nullptr, OP_LOGE(tilingContext_, "The input %u desc is null.", i),
                    return ge::GRAPH_FAILED);
        auto paramsDtype = tempDesc->GetDataType();
        // Determine whether all data types are consistent.
        if (dataType_ == ge::DT_UNDEFINED) {
            dataType_ = paramsDtype;
            if ((dataType_ != ge::DT_FLOAT) && (dataType_ != ge::DT_BF16) && (dataType_ != ge::DT_FLOAT16)) {
                OP_LOGE_FOR_INVALID_DTYPE(tilingContext_->GetNodeName(), "params/grads/exp_avgs/exp_avg_sqs",
                                          Ops::Base::ToString(paramsDtype).c_str(), "float16, bfloat16 and float");
                return ge::GRAPH_FAILED;
            }
        } else if (paramsDtype != dataType_) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                tilingContext_->GetNodeName(), "params", ge::TypeUtils::DataTypeToSerialString(paramsDtype).c_str(),
                ("The dtypes of all tensors in the tensor list must be the same. "
                 "Currently, the dtype of the " +
                 std::to_string(i) + "th tensor is inconsistent with that (" +
                 ge::TypeUtils::DataTypeToSerialString(dataType_) + ") of other tensors")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        auto paramsShape = tilingContext_->GetDynamicInputShape(0, i);
        OP_CHECK_IF(paramsShape == nullptr, OP_LOGE(tilingContext_, "The input %u shape is null.", i),
                    return ge::GRAPH_FAILED);
        // check max dim
        OP_CHECK_IF(paramsShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIM_NUMS,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        tilingContext_->GetNodeName(), "params",
                        std::to_string(paramsShape->GetStorageShape().GetDimNum()).c_str(),
                        "The shape dim of the " + std::to_string(i) +
                            "th tensor in the tensor list should be less than or equal to 8"),
                    return ge::GRAPH_FAILED);
        if (CheckShapeAndDType(INPUT_GRADS_IDX, i, paramsShape->GetStorageShape(), paramsDtype, "grads") !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CheckShapeAndDType(INPUT_EXP_AVGS_IDX, i, paramsShape->GetStorageShape(), paramsDtype, "exp_avgs") !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (CheckShapeAndDType(INPUT_EXP_AVG_SQS_IDX, i, paramsShape->GetStorageShape(), paramsDtype, "exp_avg_sqs") !=
            ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (amsgrad_ != 0) {
            if (CheckShapeAndDType(INPUT_MAX_EXP_AVG_SQS_IDX, i, paramsShape->GetStorageShape(), paramsDtype,
                                   "max_exp_avg_sqs") != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
        if (CheckStateSteps() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        // Make a 32-byte alignment for each Tensor
        tensorDataCountList_[i] = paramsShape->GetStorageShape().GetShapeSize();
        if (CheckShapeAllPositive(paramsShape->GetStorageShape(), i) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        totalDataCount_ += tensorDataCountList_[i];
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedAdamTiling::CalcTilingData()
{
    int64_t sizePerElem = ge::GetSizeByDataType(dataType_);
    if (sizePerElem == 0) {
        OP_LOGE(tilingContext_, "sizePerElem must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    int64_t elementsPerBlock = BLOCK_SIZE / sizePerElem;
    int64_t blocksPerCore = SINGLE_CORE_PROCESS_BYTES_LOWER_BOUND / BLOCK_SIZE;
    usedCoreNum_ = std::min<uint64_t>(coreNum_, MAX_CORE_CONT_950); // just defense
    // 1. 计算每个张量的块数
    uint64_t tensorBlockCount[MAX_TENSOR_CONT_950];
    int64_t totalBlocks = 0;
    for (int64_t i = 0; i < tensorNum_; ++i) {
        tensorBlockCount[i] = (tensorDataCountList_[i] + elementsPerBlock - 1) / elementsPerBlock;
        (tensorDataCountList_[i] + elementsPerBlock - 1) / elementsPerBlock;
        totalBlocks += tensorBlockCount[i];
    }
    uint32_t tempBlockPerCore = (totalBlocks + usedCoreNum_ - 1) / usedCoreNum_;
    blocksPerCore = max(blocksPerCore, static_cast<int64_t>(tempBlockPerCore));
    if (blocksPerCore == 0) {
        OP_LOGE(tilingContext_, "blocksPerCore must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = (totalBlocks + blocksPerCore - 1) / blocksPerCore;
    if (totalDataCount_ == 0) {
        usedCoreNum_ = 1;
        tensorStartList_[0] = 0;
        tensorEndList_[0] = 0;
        tensorStartOffsetList_[0] = 0;
        tensorEndOffsetList_[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
    int64_t tensorIdx = 0;           // 当前张量索引
    int64_t blockOffsetInTensor = 0; // 当前张量内的元素偏移（以elementsPerBlock为单位）
    for (uint32_t coreIdx = 0; coreIdx < usedCoreNum_; coreIdx++) {
        int64_t remainingBlocks = (coreIdx == usedCoreNum_ - 1) ?
                                      (totalBlocks - coreIdx * blocksPerCore) // 最后一个核：剩余块
                                      :
                                      blocksPerCore;
        tensorStartList_[coreIdx] = tensorIdx;
        tensorStartOffsetList_[coreIdx] = blockOffsetInTensor * elementsPerBlock;
        // 消耗 coreBlocks 个块
        while ((tensorBlockCount[tensorIdx] - blockOffsetInTensor) < remainingBlocks) {
            remainingBlocks -= tensorBlockCount[tensorIdx] - blockOffsetInTensor;
            blockOffsetInTensor = 0;
            tensorIdx++;
        }
        // here tensorBlockCount[tensorIdx] >= remainingBlocks
        tensorEndList_[coreIdx] = tensorIdx;
        if ((blockOffsetInTensor + remainingBlocks) == tensorBlockCount[tensorIdx]) {
            // 可能存在尾块不对齐 覆写之
            tensorEndOffsetList_[coreIdx] = tensorDataCountList_[tensorIdx];
            // both tensor and core match end
            tensorIdx++;
            blockOffsetInTensor = 0;
        } else {
            blockOffsetInTensor += remainingBlocks;
            tensorEndOffsetList_[coreIdx] = blockOffsetInTensor * elementsPerBlock;
        }
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t FusedAdamTiling::CalcTilingKey()
{
    switch (dataType_) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_BF16:
            return TILING_KEY_BF16;
        default:
            return 0;
    }
}

void FusedAdamTiling::SetTilingData(FusedAdamTilingData* tilingData)
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
    tilingData->usedCoreNum = usedCoreNum_;
    for (uint32_t i = 0; i < tensorNum_; i++) {
        tilingData->tensorDataCountList_[i] = tensorDataCountList_[i];
    }
    for (uint32_t i = tensorNum_; i < MAX_TENSOR_CONT_950; i++) {
        tilingData->tensorDataCountList_[i] = 0;
    }
    for (uint32_t i = 0; i < usedCoreNum_; i++) {
        tilingData->tensorStartList_[i] = tensorStartList_[i];
        tilingData->tensorEndList_[i] = tensorEndList_[i];
        tilingData->tensorStartOffsetList_[i] = tensorStartOffsetList_[i];
        tilingData->tensorEndOffsetList_[i] = tensorEndOffsetList_[i];
    }
    for (uint32_t i = usedCoreNum_; i < MAX_CORE_CONT_950; i++) {
        tilingData->tensorStartList_[i] = 0;
        tilingData->tensorEndList_[i] = 0;
        tilingData->tensorStartOffsetList_[i] = 0;
        tilingData->tensorEndOffsetList_[i] = 0;
    }

    size_t* workspaceSize = tilingContext_->GetWorkspaceSizes(1);
    *workspaceSize = sysWorkspaceSize_;
    tilingContext_->SetTilingKey(CalcTilingKey());
    tilingContext_->SetBlockDim(usedCoreNum_);
}

ge::graphStatus FusedAdamTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "FusedAdamTilingFunc");
    FusedAdamTiling tilingGenerator(context);

    OP_CHECK_IF(tilingGenerator.GetPlatformInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingGenerator.GetAttrInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetAttrInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingGenerator.GetInputTensorInfo() != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetInputTensorInfo error"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingGenerator.CalcTilingData() != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalcTilingData error"),
                return ge::GRAPH_FAILED);

    FusedAdamTilingData* tilingData = context->GetTilingData<FusedAdamTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    OP_CHECK_IF(memset_s(tilingData, sizeof(FusedAdamTilingData), 0, sizeof(FusedAdamTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tilingGenerator.SetTilingData(tilingData);
    OP_LOGD(context, "tiling data: %s", tilingGenerator.TilingDataToString().c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4FusedAdam([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedAdam).Tiling(FusedAdamTilingFunc).TilingParse<FusedAdamCompileInfo>(TilingPrepare4FusedAdam);
} // namespace optiling
