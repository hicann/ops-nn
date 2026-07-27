/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sparse_segment_mean_aicpu.h"

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char* const kSparseSegmentMean = "SparseSegmentMean";
const uint32_t kDim2 = 2;
} // namespace

namespace aicpu {
KernelStatus SparseSegmentMeanCpuKernel::SparseSegmentCheck(const CpuKernelContext& ctx) const
{
    Tensor* x = ctx.Input(0);
    Tensor* indices = ctx.Input(1);
    Tensor* segmentIds = ctx.Input(2);
    Tensor* y = ctx.Output(0);

    if (x->GetDataSize() == 0 || indices->GetDataSize() == 0 || segmentIds->GetDataSize() == 0) {
        KERNEL_LOG_ERROR("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto xShape = x->GetTensorShape();
    auto indicesShape = indices->GetTensorShape();
    auto segmentIdsShape = segmentIds->GetTensorShape();
    if (xShape->GetDims() < 1) {
        KERNEL_LOG_ERROR("[%s] Tensor x's rank less than 1.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (indicesShape->NumElements() != segmentIdsShape->NumElements()) {
        KERNEL_LOG_ERROR("[%s] Tensor indices and segment_ids size mismatch.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (x->GetDataType() != y->GetDataType()) {
        KERNEL_LOG_ERROR("[%s] Tensor data type mismatch.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
}

uint32_t SparseSegmentMeanCpuKernel::Compute(CpuKernelContext& ctx)
{
    if ((NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) ||
        (SparseSegmentCheck(ctx) != KERNEL_STATUS_OK)) {
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }

    KernelStatus result = KERNEL_STATUS_OK;
    auto xDataType = ctx.Input(0)->GetDataType();
    switch (xDataType) {
        case DT_FLOAT:
            result = ComputeKernel<float>(ctx);
            break;
        case DT_DOUBLE:
            result = ComputeKernel<double>(ctx);
            break;
        case DT_FLOAT16:
            result = ComputeKernel<Eigen::half>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("SparseSegmentMean kernel data type [%s] not support.", DTypeStr(xDataType).c_str());
            result = KERNEL_STATUS_PARAM_INVALID;
            break;
    }
    return static_cast<uint32_t>(result);
}

template <typename T, typename T1, typename T2>
KernelStatus SparseSegmentMeanCpuKernel::ComputeKernelWithType(const CpuKernelContext& ctx)
{
    auto xShape = ctx.Input(0)->GetTensorShape();
    T1 xDim0 = static_cast<T1>(xShape->GetDimSize(0));
    int64_t innerSize = 1;
    for (int32_t i = 1; i < xShape->GetDims(); i++) {
        innerSize *= xShape->GetDimSize(i);
    }

    size_t numIndices = ctx.Input(2)->GetTensorShape()->NumElements();
    auto xPtr = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto indicesPtr = reinterpret_cast<T1*>(ctx.Input(1)->GetData());
    auto segmentIdsPtr = reinterpret_cast<T2*>(ctx.Input(2)->GetData());
    auto yPtr = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    if (numIndices == 0) {
        return KERNEL_STATUS_OK;
    }

    T2 outputRows = segmentIdsPtr[numIndices - 1];
    Eigen::TensorMap<Eigen::Tensor<T, kDim2, Eigen::RowMajor>> inputFlat(xPtr, xShape->GetDimSize(0), innerSize);
    Eigen::TensorMap<Eigen::Tensor<T, kDim2, Eigen::RowMajor>> outputFlat(yPtr, outputRows + 1, innerSize);
    outputFlat.setConstant(static_cast<T>(0));

    size_t start = 0;
    size_t end = 1;
    T2 uninitializedIndex = 0;
    T2 outIndex = segmentIdsPtr[start];
    if (outIndex < 0) {
        KERNEL_LOG_ERROR("segment ids must be >= 0");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    while (true) {
        T2 nextIndex = 0;
        if (end < numIndices) {
            nextIndex = segmentIdsPtr[end];
            if (outIndex == nextIndex) {
                ++end;
                continue;
            }
            if (outIndex >= nextIndex) {
                KERNEL_LOG_ERROR("segment ids are not increasing, out_index is %ld, next_index is %ld.",
                                 static_cast<int64_t>(outIndex), static_cast<int64_t>(nextIndex));
                return KERNEL_STATUS_PARAM_INVALID;
            }
        }

        if (outIndex > outputRows) {
            KERNEL_LOG_ERROR("segment id %ld out of range [0, %ld], possibly because segment_ids input is not sorted.",
                             static_cast<int64_t>(outIndex), static_cast<int64_t>(outputRows));
            return KERNEL_STATUS_PARAM_INVALID;
        }

        if (outIndex > uninitializedIndex) {
            Eigen::DSizes<Eigen::DenseIndex, kDim2> gapSliceShape(outIndex - uninitializedIndex, innerSize);
            Eigen::TensorMap<Eigen::Tensor<T, kDim2, Eigen::RowMajor>, Eigen::Unaligned> gapSlice(
                &outputFlat(uninitializedIndex, 0), gapSliceShape);
            gapSlice.setConstant(static_cast<T>(0));
        }

        auto out = outputFlat.template chip<0>(outIndex);
        for (size_t r = start; r < end; r++) {
            T1 index = indicesPtr[r];
            if (index < 0 || index >= xDim0) {
                KERNEL_LOG_ERROR("indices out of range.");
                return KERNEL_STATUS_PARAM_INVALID;
            }
            out = out + inputFlat.template chip<0>(index);
        }
        out = out / static_cast<T>(end - start);
        start = end;
        ++end;
        uninitializedIndex = outIndex + 1;
        outIndex = nextIndex;
        if (end > numIndices) {
            break;
        }
    }

    if (uninitializedIndex < outputRows) {
        Eigen::DSizes<Eigen::DenseIndex, kDim2> gapSliceShape(outputRows - uninitializedIndex, innerSize);
        Eigen::TensorMap<Eigen::Tensor<T, kDim2, Eigen::RowMajor>, Eigen::Unaligned> gapSlice(
            &outputFlat(uninitializedIndex, 0), gapSliceShape);
        gapSlice.setConstant(static_cast<T>(0));
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
KernelStatus SparseSegmentMeanCpuKernel::ComputeKernel(const CpuKernelContext& ctx)
{
    auto indicesDataType = ctx.Input(1)->GetDataType();
    auto segmentIdsDtype = ctx.Input(2)->GetDataType();
    if (indicesDataType == DT_INT32) {
        if (segmentIdsDtype == DT_INT32) {
            return ComputeKernelWithType<T, int32_t, int32_t>(ctx);
        } else if (segmentIdsDtype == DT_INT64) {
            return ComputeKernelWithType<T, int32_t, int64_t>(ctx);
        }
    } else if (indicesDataType == DT_INT64) {
        if (segmentIdsDtype == DT_INT32) {
            return ComputeKernelWithType<T, int64_t, int32_t>(ctx);
        } else if (segmentIdsDtype == DT_INT64) {
            return ComputeKernelWithType<T, int64_t, int64_t>(ctx);
        }
    }
    return KERNEL_STATUS_PARAM_INVALID;
}

REGISTER_CPU_KERNEL(kSparseSegmentMean, SparseSegmentMeanCpuKernel);
} // namespace aicpu
