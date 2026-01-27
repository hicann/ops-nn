/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_multi_scale_deformable_attention_grad.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "multi_scale_deformable_attention_grad.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const size_t FIRST_DIM = 0;
static const size_t SECOND_DIM = 1;
static const size_t THIRD_DIM = 2;
static const size_t FOURTH_DIM = 3;
static const size_t FIFTH_DIM = 4;
static const size_t SIXTH_DIM = 5;

static const size_t VALUE_SIZE = 4;
static const size_t SPATIAL_SHAPES_SIZE = 2;
static const size_t LEVEL_START_INDEX_SIZE = 1;
static const size_t LOCATION_SIZE = 6;
static const size_t ATTN_WEIGHT_SIZE = 5;
static const size_t GRAD_OUT_SIZE = 4;
static const size_t GRAD_VALUE_SIZE = 4;
static const size_t GRAD_LOCATION_SIZE = 6;
static const size_t GRAD_ATTN_SIZE = 5;

static const int64_t NUM_KEYS = 1;
static const int64_t NUM_HEADS = 2;
static const int64_t NUM_LEVELS = 4;
static const int64_t NUM_POINTS = 5;

static const int32_t INPUT_DIM_1 = 1;
static const int32_t INPUT_DIM_2 = 2;
static const int32_t INPUT_DIM_3 = 3;
static const int32_t INPUT_DIM_4 = 4;
static const int32_t INPUT_DIM_5 = 5;
static const int32_t INPUT_DIM_6 = 6;

static const std::initializer_list<DataType> VALUE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<DataType> INDEX_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};
static const std::initializer_list<DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static bool CheckNotNull(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                         const aclTensor *location, const aclTensor *attnWeight, const aclTensor *gradOutput,
                         const aclTensor *gradValue, const aclTensor *gradLocation, const aclTensor *gradAttnWeight) {
    OP_CHECK_NULL(value, return false);
    OP_CHECK_NULL(spatialShape, return false);
    OP_CHECK_NULL(levelStartIndex, return false);
    OP_CHECK_NULL(location, return false);
    OP_CHECK_NULL(attnWeight, return false);
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(gradValue, return false);
    OP_CHECK_NULL(gradLocation, return false);
    OP_CHECK_NULL(gradAttnWeight, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                            const aclTensor *location, const aclTensor *attnWeight, const aclTensor *gradOutput,
                            const aclTensor *gradValue, const aclTensor *gradLocation, const aclTensor *gradAttnWeight) {
    // 检查value的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(value, VALUE_DTYPE_SUPPORT_LIST, return false);

    // 检查spatialShape的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(spatialShape, INDEX_DTYPE_SUPPORT_LIST, return false);

    // 检查levelStartIndex的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(levelStartIndex, INDEX_DTYPE_SUPPORT_LIST, return false);

    // 检查location的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(location, VALUE_DTYPE_SUPPORT_LIST, return false);

    // 检查attnWeight的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(attnWeight, VALUE_DTYPE_SUPPORT_LIST, return false);

    // 检查gradOutput的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOutput, VALUE_DTYPE_SUPPORT_LIST, return false);

    // 检查gradValue的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradValue, OUT_DTYPE_SUPPORT_LIST, return false);

    // 检查gradLocation的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradLocation, OUT_DTYPE_SUPPORT_LIST, return false);

    // 检查gradAttnWeight的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(gradAttnWeight, OUT_DTYPE_SUPPORT_LIST, return false);

    return true;
}

static bool CheckFormat(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                        const aclTensor *location, const aclTensor *attnWeight, const aclTensor *gradOutput,
                        const aclTensor *gradValue, const aclTensor *gradLocation, const aclTensor *gradAttnWeight) {
    // 检查输入tensor格式，不支持NZ格式
    if (value->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "parameter 'value' format does not support NZ");
        return false;
    }
    if (spatialShape->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "parameter 'spatialShape' format does not support NZ");
        return false;
    }
    if (levelStartIndex->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "parameter 'levelStartIndex' format does not support NZ");
        return false;
    }
    if (location->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "parameter 'location' format does not support NZ");
        return false;
    }
    if (attnWeight->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "parameter 'attnWeight' format does not support NZ");
        return false;
    }
    if (gradOutput->GetStorageFormat() == Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "parameter 'gradOutput' format does not support NZ");
        return false;
    }
    return true;
}

static inline bool CheckShape(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                            const aclTensor *location, const aclTensor *attnWeight, const aclTensor *gradOutput,
                            const aclTensor *gradValue, const aclTensor *gradLocation, const aclTensor *gradAttnWeight) {
    if (value->GetViewShape().GetDimNum() != INPUT_DIM_4) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "parameter 'value' expects 4 dimensions, but got %lu",
            value->GetViewShape().GetDimNum());
        return false;
    }

    if (spatialShape->GetViewShape().GetDimNum() != INPUT_DIM_2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "parameter 'spatialShape' expects 2 dimensions, but got %lu",
            spatialShape->GetViewShape().GetDimNum());
        return false;
    }

    if (spatialShape->GetViewShape().GetDim(SECOND_DIM) != INPUT_DIM_2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "parameter 'spatialShape' expects the last dimension to be 2, but got %lu",
            spatialShape->GetViewShape().GetDim(SECOND_DIM));
        return false;    
    }

    if (levelStartIndex->GetViewShape().GetDimNum() != INPUT_DIM_1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "parameter 'levelStartIndex' expects 1 dimensions, but got %lu",
            levelStartIndex->GetViewShape().GetDimNum());
        return false;
    }

    if (location->GetViewShape().GetDimNum() != INPUT_DIM_6) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "parameter 'location' expects 6 dimensions, but got %lu",
            location->GetViewShape().GetDimNum());
        return false;
    }

    if (location->GetViewShape().GetDim(SIXTH_DIM) != INPUT_DIM_2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "parameter 'location' expects the last dimension to be 2, but got %lu",
            location->GetViewShape().GetDim(SIXTH_DIM));
        return false;    
    }

    if (attnWeight->GetViewShape().GetDimNum() != INPUT_DIM_5) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "parameter 'attnWeight' expects 5 dimensions, but got %lu",
            attnWeight->GetViewShape().GetDimNum());
        return false;
    }

    if (gradOutput->GetViewShape().GetDimNum() != INPUT_DIM_3) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "parameter 'gradOutput' expects 3 dimensions, but got %lu",
            gradOutput->GetViewShape().GetDimNum());
        return false;
    }

    int64_t num_heads = value->GetViewShape().GetDim(THIRD_DIM);
    int64_t channels = value->GetViewShape().GetDim(FOURTH_DIM);
    int64_t num_levels = spatialShape->GetViewShape().GetDim(FIRST_DIM);
    int64_t num_queries = location->GetViewShape().GetDim(SECOND_DIM);
    int64_t num_points = location->GetViewShape().GetDim(FIFTH_DIM);

    bool isInputValid = (channels % 8 == 0 && channels <= 256 && num_queries < 500000 && num_levels <= 16 && num_heads <= 16 && num_points <= 16);
    if (!isInputValid) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "invalid parameter: constraints are not satisfied");
        return false;
    }

    return true;
}
static aclnnStatus CheckParams(const aclTensor *value, const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                               const aclTensor *location, const aclTensor *attnWeight, const aclTensor *gradOutput,
                               const aclTensor *gradValue, const aclTensor *gradLocation, const aclTensor *gradAttnWeight) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(value, spatialShape, levelStartIndex, location, attnWeight,
                            gradOutput, gradValue, gradLocation, gradAttnWeight), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(value, spatialShape, levelStartIndex, location, attnWeight,
                              gradOutput, gradValue, gradLocation, gradAttnWeight), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的格式是否为ND，不支持NZ格式
    CHECK_RET(CheckFormat(value, spatialShape, levelStartIndex, location, attnWeight,
                          gradOutput, gradValue, gradLocation, gradAttnWeight), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的shape是否符合要求
    CHECK_RET(CheckShape(value, spatialShape, levelStartIndex, location, attnWeight,
                          gradOutput, gradValue, gradLocation, gradAttnWeight), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(const aclTensor *value,
                                                                   const aclTensor *spatialShape, const aclTensor *levelStartIndex,
                                                                   const aclTensor *location, const aclTensor *attnWeight,
                                                                   const aclTensor *gradOutput, aclTensor *gradValue,
                                                                   aclTensor *gradLocation, aclTensor *gradAttnWeight,
                                                                   uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnMultiScaleDeformableAttentionGrad,
        DFX_IN(value, spatialShape, levelStartIndex, location, attnWeight, gradOutput),
        DFX_OUT(gradValue, gradLocation, gradAttnWeight));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(value, spatialShape, levelStartIndex, location, attnWeight, gradOutput,
        gradValue, gradLocation, gradAttnWeight);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (value->IsEmpty() || spatialShape->IsEmpty() || levelStartIndex->IsEmpty() ||
        location->IsEmpty() || attnWeight->IsEmpty() || gradOutput->IsEmpty()) {
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
    }

    // 将输入转为连续tensor
    auto valueContiguous = l0op::Contiguous(value, uniqueExecutor.get());
    CHECK_RET(valueContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto spatialShapeContiguous = l0op::Contiguous(spatialShape, uniqueExecutor.get());
    CHECK_RET(spatialShapeContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto levelStartIndexContiguous = l0op::Contiguous(levelStartIndex, uniqueExecutor.get());
    CHECK_RET(levelStartIndexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto locationContiguous = l0op::Contiguous(location, uniqueExecutor.get());
    CHECK_RET(locationContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto attnWeightContiguous = l0op::Contiguous(attnWeight, uniqueExecutor.get());
    CHECK_RET(attnWeightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_RET(gradOutputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // loc transpose
    const int64_t permuteLocList[] = {0, 2, 3, 4, 5, 1};
    int64_t locDimNum = static_cast<int64_t>(locationContiguous->GetViewShape().GetDimNum());
    auto locAxes = uniqueExecutor.get()->AllocIntArray(permuteLocList, locDimNum);
    CHECK_RET(locAxes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto locationTrans = l0op::Transpose(locationContiguous, locAxes, uniqueExecutor.get());
    CHECK_RET(locationTrans != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // attnWeight transpose
    const int64_t permuteAttnList[] = {0, 2, 3, 4, 1};
    int64_t attnDimNum = static_cast<int64_t>(attnWeightContiguous->GetViewShape().GetDimNum());
    auto attnAxes = uniqueExecutor.get()->AllocIntArray(permuteAttnList, attnDimNum);
    CHECK_RET(attnAxes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto attnWeightTrans = l0op::Transpose(attnWeightContiguous, attnAxes, uniqueExecutor.get());
    CHECK_RET(attnWeightTrans != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输入如果是float16/bfloat16，需要cast为float32
    auto valueCasted = l0op::Cast(valueContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(valueCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto locationCasted = l0op::Cast(locationTrans, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(locationCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto attnWeightCasted = l0op::Cast(attnWeightTrans, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(attnWeightCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradOutputCasted = l0op::Cast(gradOutputContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(gradOutputCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 索引和spatialShape如果非int32类型，需要强转
    auto spatialShapeCasted = l0op::Cast(spatialShapeContiguous, DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(spatialShapeCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto levelStartIndexCasted = l0op::Cast(levelStartIndexContiguous, DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(levelStartIndexCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子MultiScaleDeformableAttentionGrad进行计算
    auto multiScaleDeformableAttentionGradResult = l0op::MultiScaleDeformableAttentionGrad(valueCasted,
        spatialShapeCasted, levelStartIndexCasted, locationCasted, attnWeightCasted,
        gradOutputCasted, uniqueExecutor.get());
    auto gradValueOut = std::get<0>(multiScaleDeformableAttentionGradResult);
    auto gradLocationNoTrans = std::get<1>(multiScaleDeformableAttentionGradResult);
    auto gradAttnWeightNoTrans = std::get<2>(multiScaleDeformableAttentionGradResult);
    CHECK_RET(gradValueOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gradLocationNoTrans != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gradAttnWeightNoTrans != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // locationGrad transpose
    const int64_t permuteLocResList[] = {0, 5, 1, 2, 3, 4};
    auto locResAxes = uniqueExecutor.get()->AllocIntArray(permuteLocResList, locDimNum);
    CHECK_RET(locResAxes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradLocationOut = l0op::Transpose(gradLocationNoTrans, locResAxes, uniqueExecutor.get());
    CHECK_RET(gradLocationOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // attnWeightGrad transpose
    const int64_t permuteAttnResList[] = {0, 4, 1, 2, 3};
    auto attnResAxes = uniqueExecutor.get()->AllocIntArray(permuteAttnResList, attnDimNum);
    CHECK_RET(attnResAxes != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradAttnWeightOut = l0op::Transpose(gradAttnWeightNoTrans, attnResAxes, uniqueExecutor.get());
    CHECK_RET(gradAttnWeightOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto gradValueCastOut = l0op::Cast(gradValueOut, gradValue->GetDataType(), uniqueExecutor.get());
    CHECK_RET(gradValueCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradLocationCastOut = l0op::Cast(gradLocationOut, gradLocation->GetDataType(), uniqueExecutor.get());
    CHECK_RET(gradLocationCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradAttnWeightCastOut = l0op::Cast(gradAttnWeightOut, gradAttnWeight->GetDataType(), uniqueExecutor.get());
    CHECK_RET(gradAttnWeightCastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopygradValue = l0op::ViewCopy(gradValueCastOut, gradValue, uniqueExecutor.get());
    CHECK_RET(viewCopygradValue != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopygradLocation = l0op::ViewCopy(gradLocationCastOut, gradLocation, uniqueExecutor.get());
    CHECK_RET(viewCopygradLocation != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopygradAttnWeight = l0op::ViewCopy(gradAttnWeightCastOut, gradAttnWeight, uniqueExecutor.get());
    CHECK_RET(viewCopygradAttnWeight != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMultiScaleDeformableAttentionGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnMultiScaleDeformableAttentionGrad);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif