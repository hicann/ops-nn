/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_segment_mean_grad_aicpu.h"
#include "aicpu/nn_aicpu_register.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char* const kSparseSegmentMeanGrad = "SparseSegmentMeanGrad";
constexpr uint32_t kInputNum = 4;
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kXInputIdx = 0;
constexpr uint32_t kIndicesInputIdx = 1;
constexpr uint32_t kSegmentIdsInputIdx = 2;
constexpr uint32_t kOutputDim0InputIdx = 3;
constexpr uint32_t kYOutputIdx = 0;
// x and y are viewed as 2-D [rows, column] tensors, where column is the product of all dims except dim 0.
constexpr int32_t kFlatRank = 2;
constexpr int32_t kScalarRank = 0;
constexpr int32_t kMinRank = 1;

// Element count of a single row, i.e. the product of all dims except dim 0.
// Returns false when dim 0 is 0, which leaves the row size undefined.
bool TryGetRowSize(const std::shared_ptr<aicpu::TensorShape>& shape, const char* name, int64_t& rowSize)
{
    const int64_t dim0 = shape->GetDimSize(0);
    if (dim0 == 0) {
        KERNEL_LOG_ERROR("Tensor [%s] dim(0) must not be 0.", name);
        return false;
    }
    rowSize = shape->NumElements() / dim0;
    return true;
}

// Counts how many times every segment id is referenced, then turns the counts into the mean scale 1 / count.
// Segments that are never referenced keep a scale of 1 so that the reciprocal never divides by 0.
template <typename TSeg>
uint32_t BuildSegmentScale(const TSeg* segmentIdsPtr, const int64_t segmentIdsNum, const int64_t numSegments,
                           std::vector<double>& segmentScale)
{
    segmentScale.assign(static_cast<size_t>(numSegments), 0.0);
    for (int64_t i = 0; i < segmentIdsNum; ++i) {
        const int64_t segmentId = static_cast<int64_t>(segmentIdsPtr[i]);
        if ((segmentId >= numSegments) || (segmentId < 0)) {
            KERNEL_LOG_ERROR("Segment id [%ld] out of range [0, %ld).", segmentId, numSegments);
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
        segmentScale[static_cast<size_t>(segmentId)] += 1.0;
    }
    for (int64_t i = 0; i < numSegments; ++i) {
        segmentScale[static_cast<size_t>(i)] = 1.0 / std::max(segmentScale[static_cast<size_t>(i)], 1.0);
    }
    return aicpu::KERNEL_STATUS_OK;
}

// Scatters every gradient row back to the position recorded in indices, scaled by the segment mean factor.
// The first write to a given output row overwrites it, the following ones accumulate.
template <typename T, typename TIdx, typename TSeg>
uint32_t AccumulateGrad(const TIdx* indicesPtr, const TSeg* segmentIdsPtr, const int64_t segmentIdsNum,
                        const std::vector<double>& segmentScale,
                        Eigen::TensorMap<Eigen::Tensor<T, kFlatRank, Eigen::RowMajor>>& inputFlat,
                        Eigen::TensorMap<Eigen::Tensor<T, kFlatRank, Eigen::RowMajor>>& outputFlat)
{
    const int64_t outputDim0 = outputFlat.dimension(0);
    std::vector<bool> isModified(static_cast<size_t>(outputDim0), false);
    outputFlat.setZero();
    for (int64_t i = 0; i < segmentIdsNum; ++i) {
        const int64_t outputIdx = static_cast<int64_t>(indicesPtr[i]);
        if ((outputIdx >= outputDim0) || (outputIdx < 0)) {
            KERNEL_LOG_ERROR("Index [%ld] out of range [0, %ld).", outputIdx, outputDim0);
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
        const int64_t segmentIdx = static_cast<int64_t>(segmentIdsPtr[i]);
        const T scale = static_cast<T>(segmentScale[static_cast<size_t>(segmentIdx)]);
        if (isModified[static_cast<size_t>(outputIdx)]) {
            outputFlat.template chip<0>(outputIdx) += inputFlat.template chip<0>(segmentIdx) * scale;
        } else {
            outputFlat.template chip<0>(outputIdx) = inputFlat.template chip<0>(segmentIdx) * scale;
        }
        isModified[static_cast<size_t>(outputIdx)] = true;
    }
    return aicpu::KERNEL_STATUS_OK;
}
} // namespace

namespace aicpu {
uint32_t SparseSegmentMeanGradCpuKernel::CheckDataPara(const CpuKernelContext& ctx) const
{
    for (uint32_t i = 0; i < kInputNum; ++i) {
        if (ctx.Input(i)->GetDataSize() == 0) {
            KERNEL_LOG_ERROR("[%s] Input[%u] is an empty tensor.", ctx.GetOpType().c_str(), i);
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    const DataType xDataType = ctx.Input(kXInputIdx)->GetDataType();
    const DataType yDataType = ctx.Output(kYOutputIdx)->GetDataType();
    if (xDataType != yDataType) {
        KERNEL_LOG_ERROR("[%s] Tensor data type mismatch, x is [%s] but y is [%s].", ctx.GetOpType().c_str(),
                         DTypeStr(xDataType).c_str(), DTypeStr(yDataType).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    const DataType outputDim0DataType = ctx.Input(kOutputDim0InputIdx)->GetDataType();
    if (outputDim0DataType != DT_INT32) {
        KERNEL_LOG_ERROR("[%s] Tensor output_dim0 data type [%s] must be DT_INT32.", ctx.GetOpType().c_str(),
                         DTypeStr(outputDim0DataType).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t SparseSegmentMeanGradCpuKernel::CheckRankPara(const CpuKernelContext& ctx) const
{
    if (ctx.Input(kXInputIdx)->GetTensorShape()->GetDims() < kMinRank) {
        KERNEL_LOG_ERROR("[%s] Tensor x's rank less than 1.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Input(kOutputDim0InputIdx)->GetTensorShape()->GetDims() != kScalarRank) {
        KERNEL_LOG_ERROR("[%s] Tensor output_dim0 should be a scalar.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ctx.Output(kYOutputIdx)->GetTensorShape()->GetDims() < kMinRank) {
        KERNEL_LOG_ERROR("[%s] Tensor y's rank less than 1.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

uint32_t SparseSegmentMeanGradCpuKernel::CheckShapePara(const CpuKernelContext& ctx) const
{
    const auto xShape = ctx.Input(kXInputIdx)->GetTensorShape();
    const auto indicesShape = ctx.Input(kIndicesInputIdx)->GetTensorShape();
    const auto segmentIdsShape = ctx.Input(kSegmentIdsInputIdx)->GetTensorShape();
    const auto yShape = ctx.Output(kYOutputIdx)->GetTensorShape();

    if (indicesShape->NumElements() != segmentIdsShape->NumElements()) {
        KERNEL_LOG_ERROR("[%s] segment_ids and indices should have same size.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    int64_t xRowSize = 0;
    int64_t yRowSize = 0;
    if (!TryGetRowSize(xShape, "x", xRowSize) || !TryGetRowSize(yShape, "y", yRowSize)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (xRowSize != yRowSize) {
        KERNEL_LOG_ERROR("[%s] Tensor x and y's row sizes mismatch, [%ld] vs [%ld].", ctx.GetOpType().c_str(), xRowSize,
                         yRowSize);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    const int32_t outputDim0 = *static_cast<const int32_t*>(ctx.Input(kOutputDim0InputIdx)->GetData());
    if (yShape->GetDimSize(0) != outputDim0) {
        KERNEL_LOG_ERROR("[%s] Tensor y's dim(0) [%ld] mismatch with output_dim0 [%d].", ctx.GetOpType().c_str(),
                         yShape->GetDimSize(0), outputDim0);
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

template <typename T, typename TIdx, typename TSeg>
uint32_t SparseSegmentMeanGradCpuKernel::ComputeKernelWithType(const CpuKernelContext& ctx) const
{
    const auto xShape = ctx.Input(kXInputIdx)->GetTensorShape();
    const int64_t segmentIdsNum = ctx.Input(kIndicesInputIdx)->GetTensorShape()->NumElements();
    const int64_t numSegments = xShape->GetDimSize(0);
    int64_t column = 0;
    if (!TryGetRowSize(xShape, "x", column)) {
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto* xPtr = static_cast<T*>(ctx.Input(kXInputIdx)->GetData());
    const auto* indicesPtr = static_cast<const TIdx*>(ctx.Input(kIndicesInputIdx)->GetData());
    const auto* segmentIdsPtr = static_cast<const TSeg*>(ctx.Input(kSegmentIdsInputIdx)->GetData());
    const int32_t outputDim0 = *static_cast<const int32_t*>(ctx.Input(kOutputDim0InputIdx)->GetData());
    auto* yPtr = static_cast<T*>(ctx.Output(kYOutputIdx)->GetData());

    std::vector<double> segmentScale;
    const uint32_t scaleRet = BuildSegmentScale<TSeg>(segmentIdsPtr, segmentIdsNum, numSegments, segmentScale);
    if (scaleRet != KERNEL_STATUS_OK) {
        return scaleRet;
    }

    Eigen::TensorMap<Eigen::Tensor<T, kFlatRank, Eigen::RowMajor>> inputFlat(xPtr, numSegments, column);
    Eigen::TensorMap<Eigen::Tensor<T, kFlatRank, Eigen::RowMajor>> outputFlat(yPtr, outputDim0, column);
    return AccumulateGrad<T, TIdx, TSeg>(indicesPtr, segmentIdsPtr, segmentIdsNum, segmentScale, inputFlat, outputFlat);
}

template <typename T>
uint32_t SparseSegmentMeanGradCpuKernel::ComputeKernel(const CpuKernelContext& ctx) const
{
    const DataType indicesDataType = ctx.Input(kIndicesInputIdx)->GetDataType();
    const DataType segmentIdsDataType = ctx.Input(kSegmentIdsInputIdx)->GetDataType();
    if (indicesDataType != DT_INT32 && indicesDataType != DT_INT64) {
        KERNEL_LOG_ERROR("SparseSegmentMeanGrad indices data type [%s] not support.",
                         DTypeStr(indicesDataType).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    if (segmentIdsDataType != DT_INT32 && segmentIdsDataType != DT_INT64) {
        KERNEL_LOG_ERROR("SparseSegmentMeanGrad segment_ids data type [%s] not support.",
                         DTypeStr(segmentIdsDataType).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (indicesDataType == DT_INT32) {
        return (segmentIdsDataType == DT_INT32) ? ComputeKernelWithType<T, int32_t, int32_t>(ctx) :
                                                  ComputeKernelWithType<T, int32_t, int64_t>(ctx);
    }
    return (segmentIdsDataType == DT_INT32) ? ComputeKernelWithType<T, int64_t, int32_t>(ctx) :
                                              ComputeKernelWithType<T, int64_t, int64_t>(ctx);
}

uint32_t SparseSegmentMeanGradCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Check SparseSegmentMeanGrad params failed.");
    KERNEL_HANDLE_ERROR(CheckDataPara(ctx), "Check SparseSegmentMeanGrad data params failed.");
    KERNEL_HANDLE_ERROR(CheckRankPara(ctx), "Check SparseSegmentMeanGrad rank params failed.");
    KERNEL_HANDLE_ERROR(CheckShapePara(ctx), "Check SparseSegmentMeanGrad shape params failed.");

    const DataType xDataType = ctx.Input(kXInputIdx)->GetDataType();
    switch (xDataType) {
        case DT_FLOAT:
            return ComputeKernel<float>(ctx);
        case DT_DOUBLE:
            return ComputeKernel<double>(ctx);
        case DT_FLOAT16:
            return ComputeKernel<Eigen::half>(ctx);
        default:
            KERNEL_LOG_ERROR("SparseSegmentMeanGrad kernel data type [%s] not support.", DTypeStr(xDataType).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

OPS_NN_REGISTER_CPU_KERNELV2(kSparseSegmentMeanGrad, SparseSegmentMeanGradCpuKernel);
} // namespace aicpu
