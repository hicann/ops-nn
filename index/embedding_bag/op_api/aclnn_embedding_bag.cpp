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
 * \file aclnn_embedding_bag.cpp
 * \brief
 */
#include "log/log.h"
#include "aclnn_embedding_bag.h"
#include "embedding_bag.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr const char* kOpName = "aclnnEmbeddingBag";
static const int64_t MAX_SUPPORT_DIM = 2;
static const int64_t MIN_SUPPORT_DIM = 1;
static const int64_t BAG_SIZE_DIM = 2;
static const int64_t MAX_INDICES_DIM = 3;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> WEIGHT_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> WEIGHT_DTYPE_SUPPORT_LIST_910 = {op::DataType::DT_FLOAT,
                                                                                  op::DataType::DT_FLOAT16};

static inline const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == Ops::Base::DAV_2002 || curArch == Ops::Base::DAV_1001) {
        return WEIGHT_DTYPE_SUPPORT_LIST_910;
    } else {
        return WEIGHT_DTYPE_SUPPORT_LIST_910B;
    }
}

static const std::initializer_list<op::DataType> INT_DTYPE_LIST = {op::DataType::DT_INT64, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> INT_DTYPE_LIST_ALL = {op::DataType::DT_INT64, op::DataType::DT_INT32,
                                                                       op::DataType::DT_INT16, op::DataType::DT_INT8,
                                                                       op::DataType::DT_UINT8};

static bool ReportInvalidDtype(const aclTensor* tensor, const char* paramName, const char* reason)
{
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(kOpName, paramName, op::ToString(tensor->GetDataType()).GetString(), reason);
    return false;
}

static bool ReportInvalidShapedim(const char* paramName, int64_t dimNum, const char* reason)
{
    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, paramName, std::to_string(dimNum).c_str(), reason);
    return false;
}

static inline bool CheckNotNull(const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets,
                                aclTensor* output, aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices)
{
    if (weight == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "weight");
        return false;
    }
    if (indices == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "indices");
        return false;
    }
    if (offsets == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "offsets");
        return false;
    }
    if (output == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "y");
        return false;
    }
    if (offset2bag == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "offset2bag");
        return false;
    }
    if (bagSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "bag_size");
        return false;
    }
    if (maxIndices == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "max_indices");
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets,
                            const aclTensor* perSampleWeights, const aclTensor* output, const aclTensor* offset2bag,
                            const aclTensor* bagSize, const aclTensor* maxIndices)
{
    // 检查weight的数据类型是否在算子的支持列表内
    if (weight != nullptr && !CheckType(weight->GetDataType(), GetDtypeSupportList())) {
        return ReportInvalidDtype(weight, "weight", "weight dtype is not supported by EmbeddingBag on current SoC");
    }

    // 检查indices的数据类型是否在算子的支持列表内
    if (indices != nullptr && !CheckType(indices->GetDataType(), INT_DTYPE_LIST_ALL)) {
        return ReportInvalidDtype(indices, "indices",
                                  "indices dtype should be one of [INT64, INT32, INT16, INT8, UINT8]");
    }

    // 检查offsets的数据类型是否在算子的支持列表内
    if (offsets != nullptr && !CheckType(offsets->GetDataType(), INT_DTYPE_LIST_ALL)) {
        return ReportInvalidDtype(offsets, "offsets",
                                  "offsets dtype should be one of [INT64, INT32, INT16, INT8, UINT8]");
    }

    // 检查indices和offsets的数据类型是否有一个达到了INT32/INT64
    if (!CheckType(indices->GetDataType(), INT_DTYPE_LIST) && !CheckType(offsets->GetDataType(), INT_DTYPE_LIST)) {
        std::string dtypes = std::string(op::ToString(indices->GetDataType()).GetString()) + ", " +
                             std::string(op::ToString(offsets->GetDataType()).GetString());
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(kOpName, "indices, offsets", dtypes.c_str(),
                                               "indices or offsets must have one dtype in [INT32, INT64]");
        return false;
    }

    // 检查perSampleWeights的数据类型是否与weight相同
    if (perSampleWeights != nullptr && weight->GetDataType() != perSampleWeights->GetDataType()) {
        return ReportInvalidDtype(perSampleWeights, "per_sample_weight",
                                  "per_sample_weight dtype must be the same as weight dtype");
    }

    // 检查weight和output的数据类型一致
    if (weight->GetDataType() != output->GetDataType()) {
        return ReportInvalidDtype(output, "y", "y dtype must be the same as weight dtype");
    }

    // 检查offset2bag的数据类型是否在算子的支持列表内
    if (offset2bag != nullptr && !CheckType(offset2bag->GetDataType(), INT_DTYPE_LIST)) {
        return ReportInvalidDtype(offset2bag, "offset2bag", "offset2bag dtype should be one of [INT32, INT64]");
    }

    // 检查bagSize的数据类型是否在算子支持列表内
    if (bagSize != nullptr && !CheckType(bagSize->GetDataType(), INT_DTYPE_LIST)) {
        return ReportInvalidDtype(bagSize, "bag_size", "bag_size dtype should be one of [INT32, INT64]");
    }

    // 检查maxIndices的数据类型是否在算子支持列表内
    if (maxIndices != nullptr && !CheckType(maxIndices->GetDataType(), INT_DTYPE_LIST)) {
        return ReportInvalidDtype(maxIndices, "max_indices", "max_indices dtype should be one of [INT32, INT64]");
    }
    return true;
}

static bool CheckDims(const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets,
                      const aclTensor* perSampleWeights, const std::string& modeStr, const aclTensor* output,
                      const aclTensor* offset2bag, const aclTensor* bagSize, const aclTensor* maxIndices)
{
    if (weight != nullptr) {
        int64_t dimNum = static_cast<int64_t>(weight->GetViewShape().GetDimNum());
        if (dimNum != MAX_SUPPORT_DIM) {
            return ReportInvalidShapedim("weight", dimNum, "weight must be 2D");
        }
    }

    if (indices != nullptr) {
        int64_t dimNum = static_cast<int64_t>(indices->GetViewShape().GetDimNum());
        int64_t maxDim = Ops::NN::AclnnUtil::IsRegbase() ? MAX_SUPPORT_DIM : MIN_SUPPORT_DIM;
        if (dimNum > maxDim) {
            return ReportInvalidShapedim("indices", dimNum, "indices shape dim exceeds the supported maximum");
        }
    }

    if (offsets != nullptr) {
        int64_t dimNum = static_cast<int64_t>(offsets->GetViewShape().GetDimNum());
        if (dimNum > MIN_SUPPORT_DIM) {
            return ReportInvalidShapedim("offsets", dimNum, "offsets must be at most 1D");
        }
    }

    if (perSampleWeights != nullptr) {
        int64_t dimNum = static_cast<int64_t>(perSampleWeights->GetViewShape().GetDimNum());
        if (Ops::NN::AclnnUtil::IsRegbase()) {
            if (dimNum > MAX_SUPPORT_DIM || dimNum < MIN_SUPPORT_DIM) {
                return ReportInvalidShapedim("per_sample_weight", dimNum, "per_sample_weight must be 1D or 2D");
            }
        } else if (dimNum != MIN_SUPPORT_DIM) {
            return ReportInvalidShapedim("per_sample_weight", dimNum, "per_sample_weight must be 1D");
        }
    }

    if (output != nullptr) {
        int64_t dimNum = static_cast<int64_t>(output->GetViewShape().GetDimNum());
        if (dimNum != MAX_SUPPORT_DIM) {
            return ReportInvalidShapedim("y", dimNum, "y must be 2D");
        }
    }

    if (offset2bag != nullptr) {
        int64_t dimNum = static_cast<int64_t>(offset2bag->GetViewShape().GetDimNum());
        if (dimNum > MIN_SUPPORT_DIM) {
            return ReportInvalidShapedim("offset2bag", dimNum, "offset2bag must be at most 1D");
        }
    }

    if (bagSize != nullptr) {
        int64_t dimNum = static_cast<int64_t>(bagSize->GetViewShape().GetDimNum());
        if (dimNum > MIN_SUPPORT_DIM) {
            return ReportInvalidShapedim("bag_size", dimNum, "bag_size must be at most 1D");
        }
    }

    if (maxIndices != nullptr) {
        int64_t dimNum = static_cast<int64_t>(maxIndices->GetViewShape().GetDimNum());
        if (Ops::NN::AclnnUtil::IsRegbase()) {
            if (dimNum > MAX_SUPPORT_DIM) {
                return ReportInvalidShapedim("max_indices", dimNum, "max_indices must be at most 2D");
            }
        } else if (modeStr == "max") {
            if (dimNum != MAX_SUPPORT_DIM) {
                return ReportInvalidShapedim("max_indices", dimNum, "max_indices must be 2D when mode is 'max'");
            }
        } else if (dimNum > MIN_SUPPORT_DIM) {
            return ReportInvalidShapedim("max_indices", dimNum,
                                         "max_indices must be at most 1D when mode is not 'max'");
        }
    }
    return true;
}

static op::Shape GetOutPutShape(const aclTensor* weight, const aclTensor* offsets, bool includeLastOffset)
{
    op::Shape outputShape;
    int64_t offsetSize = offsets->GetViewShape().GetShapeSize();
    if (includeLastOffset) {
        offsetSize -= 1;
    }
    outputShape.AppendDim(offsetSize);
    outputShape.AppendDim((weight->GetViewShape())[1]);
    return outputShape;
}

static bool CheckShape(const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets,
                       const aclTensor* perSampleWeights, const std::string& modeStr, bool includeLastOffset,
                       const aclTensor* output, const aclTensor* offset2bag, const aclTensor* bagSize,
                       const aclTensor* maxIndices)
{
    if (!Ops::NN::AclnnUtil::IsRegbase()) {
        if (modeStr != "sum" && perSampleWeights != nullptr) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(kOpName, "mode, per_sample_weight",
                                                   "per_sample_weight is set while mode is not 'sum'",
                                                   "per_sample_weight is only supported with mode='sum'");
            return false;
        }
    }

    if (perSampleWeights != nullptr &&
        indices->GetViewShape().GetShapeSize() != perSampleWeights->GetViewShape().GetShapeSize()) {
        std::string sizes = std::to_string(indices->GetViewShape().GetShapeSize()) + ", " +
                            std::to_string(perSampleWeights->GetViewShape().GetShapeSize());
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            kOpName, "indices, per_sample_weight", sizes.c_str(),
            "indices and per_sample_weight must have the same number of elements");
        return false;
    }

    auto outputShape = GetOutPutShape(weight, offsets, includeLastOffset);
    if (output->GetViewShape() != outputShape) {
        OP_LOGE_FOR_INVALID_SHAPE(kOpName, "y", op::ToString(output->GetViewShape()).GetString(),
                                  op::ToString(outputShape).GetString());
        return false;
    }

    if (offset2bag->GetViewShape().GetShapeSize() != 0 &&
        offset2bag->GetViewShape().GetShapeSize() != indices->GetViewShape().GetShapeSize()) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(kOpName, "offset2bag",
                                                  std::to_string(offset2bag->GetViewShape().GetShapeSize()).c_str(),
                                                  "offset2bag shape size must be 0 or equal to the number of indices");
        return false;
    }

    if (Ops::NN::AclnnUtil::IsRegbase()) {
        if (bagSize->GetViewShape().GetShapeSize() != offsets->GetViewShape().GetDim(0)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(kOpName, "bag_size",
                                                      std::to_string(bagSize->GetViewShape().GetShapeSize()).c_str(),
                                                      "bag_size shape size must equal the number of offsets");
            return false;
        }
    } else {
        if (includeLastOffset) {
            if (bagSize->GetViewShape().GetShapeSize() != offsets->GetViewShape().GetDim(0) - 1) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    kOpName, "bag_size", std::to_string(bagSize->GetViewShape().GetShapeSize()).c_str(),
                    "bag_size shape size must equal the number of offsets minus one when include_last_offset is true");
                return false;
            }
        } else {
            if (bagSize->GetViewShape().GetShapeSize() != offsets->GetViewShape().GetDim(0)) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    kOpName, "bag_size", std::to_string(bagSize->GetViewShape().GetShapeSize()).c_str(),
                    "bag_size shape size must equal the number of offsets");
                return false;
            }
        }
    }

    if (modeStr == "max") {
        auto maxIndicesShape = GetOutPutShape(weight, offsets, includeLastOffset);
        if (maxIndices->GetViewShape() != maxIndicesShape) {
            OP_LOGE_FOR_INVALID_SHAPE(kOpName, "max_indices", op::ToString(maxIndices->GetViewShape()).GetString(),
                                      op::ToString(maxIndicesShape).GetString());
            return false;
        }
    } else {
        if (!Ops::NN::AclnnUtil::IsRegbase()) {
            int64_t expectedSize = includeLastOffset ? offsets->GetViewShape().GetDim(0) - 1 :
                                                       offsets->GetViewShape().GetDim(0);
            if (maxIndices->GetViewShape().GetShapeSize() != expectedSize) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    kOpName, "max_indices", std::to_string(maxIndices->GetViewShape().GetShapeSize()).c_str(),
                    "max_indices shape size must equal the expected number of bags");
                return false;
            }
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* weight, const aclTensor* indices, const aclTensor* offsets,
                               const std::string& modeStr, const aclTensor* perSampleWeights, bool includeLastOffset,
                               aclTensor* output, aclTensor* offset2bag, aclTensor* bagSize, aclTensor* maxIndices)
{
    // 1. 检查参数是否为空指针
    if (!CheckNotNull(weight, indices, offsets, output, offset2bag, bagSize, maxIndices)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    if (!CheckDtypeValid(weight, indices, offsets, perSampleWeights, output, offset2bag, bagSize, maxIndices)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 3. 检查输入的dim 是否满足
    if (!CheckDims(weight, indices, offsets, perSampleWeights, modeStr, output, offset2bag, bagSize, maxIndices)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 4. 检查输入形状是否满足
    if (!CheckShape(weight, indices, offsets, perSampleWeights, modeStr, includeLastOffset, output, offset2bag, bagSize,
                    maxIndices)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static std::string getModeStr(const int64_t mode)
{
    std::string modeStr = "max";
    if (mode == 0) {
        modeStr = "sum";
    } else if (mode == 1) {
        modeStr = "mean";
    }
    return modeStr;
}

aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(const aclTensor* weight, const aclTensor* indices,
                                              const aclTensor* offsets, bool scaleGradByFreq, int64_t mode, bool sparse,
                                              const aclTensor* perSampleWeights, bool includeLastOffset,
                                              int64_t paddingIdx, aclTensor* output, aclTensor* offset2bag,
                                              aclTensor* bagSize, aclTensor* maxIndices, uint64_t* workspaceSize,
                                              aclOpExecutor** executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "workspaceSize");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    L2_DFX_PHASE_1(aclnnEmbeddingBag,
                   DFX_IN(weight, indices, offsets, scaleGradByFreq, mode, sparse, perSampleWeights, includeLastOffset,
                          paddingIdx),
                   DFX_OUT(output, offset2bag, bagSize, maxIndices));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();

    // 固定写法，参数检查
    auto ret = CheckParams(weight, indices, offsets, getModeStr(mode), perSampleWeights, includeLastOffset, output,
                           offset2bag, bagSize, maxIndices);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    if (weight->IsEmpty() || indices->IsEmpty() || offsets->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 将输入weight,indices,offsets转换成连续的tensor
    auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    auto offsetsContiguous = l0op::Contiguous(offsets, uniqueExecutor.get());
    if (weightContiguous == nullptr || indicesContiguous == nullptr || offsetsContiguous == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "Contiguous of weight, indices or offsets failed.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    // 将输入per_sample_weights转换成连续的tensor
    auto perSampleWeightsContiguous = perSampleWeights;
    if (perSampleWeights != nullptr) {
        perSampleWeightsContiguous = l0op::Contiguous(perSampleWeights, uniqueExecutor.get());
        if (perSampleWeightsContiguous == nullptr) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "Contiguous of per_sample_weight failed.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    const aclTensor* indicesCast = nullptr;
    const aclTensor* offsetsCast = nullptr;
    // indices、offsets类型转换。
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        if (indices->GetDataType() == op::DataType::DT_INT32 || indices->GetDataType() == op::DataType::DT_INT64) {
            indicesCast = indicesContiguous;
        } else {
            indicesCast = l0op::Cast(indicesContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
        }

        if (offsets->GetDataType() == op::DataType::DT_INT32 || offsets->GetDataType() == op::DataType::DT_INT64) {
            offsetsCast = offsetsContiguous;
        } else {
            offsetsCast = l0op::Cast(offsetsContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
        }
    } else {
        indicesCast = l0op::Cast(indicesContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
        offsetsCast = l0op::Cast(offsetsContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
    }
    if (indicesCast == nullptr || offsetsCast == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "Cast of indices or offsets failed.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    static const std::string modeStr[] = {"sum", "mean"};
    static const std::string modeMax = "max";
    auto result = l0op::EmbeddingBag(weightContiguous, indicesCast, offsetsCast, scaleGradByFreq,
                                     (mode != 0 && mode != 1) ? modeMax : modeStr[mode], sparse,
                                     perSampleWeightsContiguous, includeLastOffset, paddingIdx, uniqueExecutor.get());
    if (std::get<0>(result) == nullptr || std::get<1>(result) == nullptr || std::get<BAG_SIZE_DIM>(result) == nullptr ||
        std::get<MAX_INDICES_DIM>(result) == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "EmbeddingBag l0op returned null result.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyOutputResult = l0op::ViewCopy(std::get<0>(result), output, uniqueExecutor.get());

    if (offset2bag->GetViewShape().GetShapeSize() != 0) {
        auto offset2bagL0Cast = l0op::Cast(std::get<1>(result), offset2bag->GetDataType(), uniqueExecutor.get());
        if (offset2bagL0Cast == nullptr) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "Cast of offset2bag result failed.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        auto viewCopyoffset2bagL0CastResult = l0op::ViewCopy(offset2bagL0Cast, offset2bag, uniqueExecutor.get());
        if (viewCopyoffset2bagL0CastResult == nullptr) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "ViewCopy of offset2bag failed.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    auto bagSizeL0Cast = l0op::Cast(std::get<2>(result), bagSize->GetDataType(), uniqueExecutor.get());
    if (bagSizeL0Cast == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "Cast of bag_size result failed.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    auto viewCopyBagSizeL0CastResult = l0op::ViewCopy(bagSizeL0Cast, bagSize, uniqueExecutor.get());
    auto maxIndicesShapeSize = std::get<3>(result)->GetViewShape().GetShapeSize();
    if (viewCopyOutputResult == nullptr || viewCopyBagSizeL0CastResult == nullptr) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "ViewCopy of output or bag_size failed.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (maxIndicesShapeSize != 0) {
        auto maxIndicesL0Cast = l0op::Cast(std::get<3>(result), maxIndices->GetDataType(), uniqueExecutor.get());
        if (maxIndicesL0Cast == nullptr) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "Cast of max_indices result failed.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        auto viewCopyMaxIndicesL0CastResult = l0op::ViewCopy(maxIndicesL0Cast, maxIndices, uniqueExecutor.get());
        if (viewCopyMaxIndicesL0CastResult == nullptr) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER_NULLPTR, "ViewCopy of max_indices failed.");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEmbeddingBag(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnEmbeddingBag);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
