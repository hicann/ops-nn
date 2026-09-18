/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "sparse_segment_sum_aicpu.h"
#include "aicpu/nn_aicpu_register.h"

namespace {
const uint32_t kInputNum = 3;
const uint32_t kOutputNum = 1;
const char* const SparseSegmentSum = "SparseSegmentSum";
const uint32_t dim_2 = 2;
} // namespace

namespace aicpu {
KernelStatus SparseSegmentSumCpuKernel::SparseSegmentCheck(const CpuKernelContext& ctx) const
{
    Tensor* x = ctx.Input(0);
    Tensor* indices = ctx.Input(1);
    Tensor* segment_ids = ctx.Input(2);
    Tensor* y = ctx.Output(0);

    auto x_shape = x->GetTensorShape();
    auto indices_shape = indices->GetTensorShape();
    auto segment_ids_shape = segment_ids->GetTensorShape();

    if (x_shape->GetDims() < 1) {
        KERNEL_LOG_ERROR("[%s] Tensor x's rank less than 1.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (indices_shape->NumElements() != segment_ids_shape->NumElements()) {
        KERNEL_LOG_ERROR("[%s] Tensor indices&segment_ids's ranks mismatch.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    auto x_data_type = x->GetDataType();
    auto y_data_type = y->GetDataType();
    if (x_data_type != y_data_type) {
        KERNEL_LOG_ERROR("[%s] Tensor data type mismatch.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    return KERNEL_STATUS_OK;
}

template <typename T1, typename T2>
KernelStatus SparseSegmentSumCpuKernel::SparseSegmentDataCheckWithType(const CpuKernelContext& ctx) const
{
    auto indices_ptr = reinterpret_cast<T1*>(ctx.Input(1)->GetData());
    auto segment_ids_ptr = reinterpret_cast<T2*>(ctx.Input(2)->GetData());
    size_t m = ctx.Input(2)->GetTensorShape()->NumElements();
    auto x_dim0 = ctx.Input(0)->GetTensorShape()->GetDimSize(0);

    if (m >= 1) {
        if (segment_ids_ptr[0] < 0) {
            KERNEL_LOG_ERROR("segment ids must be >= 0.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (indices_ptr[0] >= x_dim0) {
            KERNEL_LOG_ERROR("indices out of range.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    for (size_t i = 1; i < m; i++) {
        if (segment_ids_ptr[i] < segment_ids_ptr[i - 1]) {
            KERNEL_LOG_ERROR("segment ids are not increasing.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (segment_ids_ptr[i] < 0) {
            KERNEL_LOG_ERROR("segment ids must be >= 0.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (indices_ptr[i] >= x_dim0) {
            KERNEL_LOG_ERROR("indices out of range.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    return KERNEL_STATUS_OK;
}

KernelStatus SparseSegmentSumCpuKernel::SparseSegmentDataCheck(const CpuKernelContext& ctx) const
{
    auto indices_data_type = ctx.Input(1)->GetDataType();
    auto segment_ids_dtype = ctx.Input(2)->GetDataType();
    if (indices_data_type == DT_INT32) {
        if (segment_ids_dtype == DT_INT32) {
            return SparseSegmentDataCheckWithType<int32_t, int32_t>(ctx);
        } else if (segment_ids_dtype == DT_INT64) {
            return SparseSegmentDataCheckWithType<int32_t, int64_t>(ctx);
        } else {
            KERNEL_LOG_ERROR("SparseSegmentSum kernel data type [%s] not support.",
                             DTypeStr(segment_ids_dtype).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else if (indices_data_type == DT_INT64) {
        if (segment_ids_dtype == DT_INT32) {
            return SparseSegmentDataCheckWithType<int64_t, int32_t>(ctx);
        } else if (segment_ids_dtype == DT_INT64) {
            return SparseSegmentDataCheckWithType<int64_t, int64_t>(ctx);
        } else {
            KERNEL_LOG_ERROR("SparseSegmentSum kernel data type [%s] not support.",
                             DTypeStr(segment_ids_dtype).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else {
        KERNEL_LOG_ERROR("SparseSegmentSum kernel data type [%s] not support.", DTypeStr(indices_data_type).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
}

KernelStatus SparseSegmentSumCpuKernel::ComputeWithType(const CpuKernelContext& ctx)
{
    KernelStatus result = KERNEL_STATUS_OK;
    auto x_data_type = ctx.Input(0)->GetDataType();
    switch (x_data_type) {
        case (DT_INT8):
            result = ComputeKernel<int8_t>(ctx);
            break;
        case (DT_INT16):
            result = ComputeKernel<int16_t>(ctx);
            break;
        case (DT_INT32):
            result = ComputeKernel<int32_t>(ctx);
            break;
        case (DT_INT64):
            result = ComputeKernel<int64_t>(ctx);
            break;
        case (DT_UINT8):
            result = ComputeKernel<uint8_t>(ctx);
            break;
        case (DT_UINT16):
            result = ComputeKernel<uint16_t>(ctx);
            break;
        case (DT_UINT32):
            result = ComputeKernel<uint32_t>(ctx);
            break;
        case (DT_UINT64):
            result = ComputeKernel<uint64_t>(ctx);
            break;
        case (DT_FLOAT16):
            result = ComputeKernel<Eigen::half>(ctx);
            break;
        case (DT_FLOAT):
            result = ComputeKernel<float>(ctx);
            break;
        case (DT_DOUBLE):
            result = ComputeKernel<double>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("SparseSegmentSum kernel data type [%s] not support.", DTypeStr(x_data_type).c_str());
            result = KERNEL_STATUS_PARAM_INVALID;
    }
    return result;
}

uint32_t SparseSegmentSumCpuKernel::Compute(CpuKernelContext& ctx)
{
    if ((NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) ||
        (SparseSegmentCheck(ctx) != KERNEL_STATUS_OK) || (SparseSegmentDataCheck(ctx) != KERNEL_STATUS_OK)) {
        return static_cast<uint32_t>(KERNEL_STATUS_PARAM_INVALID);
    }

    return static_cast<uint32_t>(ComputeWithType(ctx));
}

template <typename T, typename T1, typename T2>
KernelStatus SparseSegmentSumCpuKernel::ComputeKernelWithType(const CpuKernelContext& ctx)
{
    size_t n = ctx.Input(0)->GetTensorShape()->NumElements() / ctx.Input(0)->GetTensorShape()->GetDimSize(0);
    size_t num_indices = ctx.Input(2)->GetTensorShape()->NumElements();
    auto x_ptr = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto indices_ptr = reinterpret_cast<T1*>(ctx.Input(1)->GetData());
    auto segment_ids_ptr = reinterpret_cast<T2*>(ctx.Input(2)->GetData());
    auto y_ptr = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    if (num_indices == 0) {
        return KERNEL_STATUS_OK;
    }
    int64_t output_rows = segment_ids_ptr[num_indices - 1];
    Eigen::TensorMap<Eigen::Tensor<T, dim_2, Eigen::RowMajor>> input_flat(
        x_ptr, ctx.Input(0)->GetTensorShape()->GetDimSize(0), n);
    Eigen::TensorMap<Eigen::Tensor<T, dim_2, Eigen::RowMajor>> output_flat(y_ptr, output_rows + 1, n);
    output_flat.setConstant(static_cast<T>(0));

    size_t start = 0;
    size_t end = 1;
    int64_t uninitialized_index = 0;
    int64_t out_index = segment_ids_ptr[start];

    while (true) {
        int64_t next_index = 0;
        if (end < num_indices) {
            next_index = segment_ids_ptr[end];
            if (out_index == next_index) {
                ++end;
                continue;
            }
        }
        if (out_index > uninitialized_index) {
            Eigen::DSizes<Eigen::DenseIndex, dim_2> gap_slice_shape(out_index - uninitialized_index, n);
            Eigen::TensorMap<Eigen::Tensor<T, dim_2, Eigen::RowMajor>, Eigen::Unaligned> gap_slice(
                &output_flat(uninitialized_index, 0), gap_slice_shape);
            gap_slice.setConstant(static_cast<T>(0));
        }

        auto out = output_flat.template chip<0>(out_index);
        for (size_t r = start; r < end; r++) {
            int64_t index = indices_ptr[r];
            out = out + input_flat.template chip<0>(index);
        }
        start = end;
        ++end;
        uninitialized_index = out_index + 1;
        out_index = next_index;
        if (end > num_indices)
            break;
    }
    if (uninitialized_index < output_rows) {
        Eigen::DSizes<Eigen::DenseIndex, dim_2> gap_slice_shape(output_rows - uninitialized_index, n);
        Eigen::TensorMap<Eigen::Tensor<T, dim_2, Eigen::RowMajor>, Eigen::Unaligned> gap_slice(
            &output_flat(uninitialized_index, 0), gap_slice_shape);
        gap_slice.setConstant(static_cast<T>(0));
    }
    return KERNEL_STATUS_OK;
};
template <typename T>
KernelStatus SparseSegmentSumCpuKernel::ComputeKernel(const CpuKernelContext& ctx)
{
    auto indices_data_type = ctx.Input(1)->GetDataType();
    auto segment_ids_dtype = ctx.Input(2)->GetDataType();
    if (indices_data_type == DT_INT32) {
        if (segment_ids_dtype == DT_INT32) {
            return ComputeKernelWithType<T, int32_t, int32_t>(ctx);
        } else if (segment_ids_dtype == DT_INT64) {
            return ComputeKernelWithType<T, int32_t, int64_t>(ctx);
        } else {
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else if (indices_data_type == DT_INT64) {
        if (segment_ids_dtype == DT_INT32) {
            return ComputeKernelWithType<T, int64_t, int32_t>(ctx);
        } else if (segment_ids_dtype == DT_INT64) {
            return ComputeKernelWithType<T, int64_t, int64_t>(ctx);
        } else {
            return KERNEL_STATUS_PARAM_INVALID;
        }
    } else {
        return KERNEL_STATUS_PARAM_INVALID;
    }
}

OPS_NN_REGISTER_CPU_KERNELV2(SparseSegmentSum, SparseSegmentSumCpuKernel);
} // namespace aicpu
