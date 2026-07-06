/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_nd_min_aicpu.h"

#include <atomic>
#include <iostream>

#include "Eigen/Dense"
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const int32_t kScatterNdMinInputNum = 3;
const int32_t kScatterNdMinOutputNum = 1;
const char* const kScatterNdMin = "ScatterNdMin";
const int64_t kParallelDataNum = 32 * 1024;
const int32_t kSplitSize = 64 * 1024;

#define DO_SCATTER_ND_MIN_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX) \
    case (DTYPE):                                                \
    do {                                                         \
        if ((ITYPE) == DT_INT32) {                               \
            return DoScatterNdMinCompute<TYPE, int32_t>(CTX);    \
        } else {                                                 \
            return DoScatterNdMinCompute<TYPE, int64_t>(CTX);    \
        }                                                        \
    } while (0)
} // namespace

namespace aicpu {
uint32_t ScatterNdMinCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kScatterNdMinInputNum, kScatterNdMinOutputNum),
                        "ScatterNdMin check input or output is failed.");
    Tensor* input_data = ctx.Input(0);
    Tensor* input_indices = ctx.Input(1);
    auto data_type = input_data->GetDataType();
    auto data_type_indices = input_indices->GetDataType();
    KERNEL_CHECK_FALSE((data_type_indices == DT_INT32 || data_type_indices == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                       "Input[1] data type[%s] is unsupported", DTypeStr(data_type_indices).c_str());
    switch (data_type) {
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_DOUBLE, double, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_FLOAT, float, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_FLOAT16, Eigen::half, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_INT16, int16_t, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_INT32, int32_t, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_INT64, int64_t, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_INT8, int8_t, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_UINT16, uint16_t, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_UINT32, uint32_t, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_UINT64, uint64_t, data_type_indices, ctx);
        DO_SCATTER_ND_MIN_COMPUTE_CASE(DT_UINT8, uint8_t, data_type_indices, ctx);
        default: {
            std::string unsupported_dtype_msg = ConcatString(
                "Input[0] data type[", DTypeStr(data_type), "] is unsupported.",
                "It should be double|float|float16|int16|int32|int64|int8|uint16|uint32|uint64|uint8");
            KERNEL_LOG_ERROR("%s", unsupported_dtype_msg.c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
}

template <typename T>
uint32_t ScatterNdMinCpuKernel::InitScatterNdMinOutput(const CpuKernelContext& ctx)
{
    auto src_ref = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto dst_ref = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    std::atomic<uint32_t> copy_status(KERNEL_STATUS_OK);
    int64_t output_num = ctx.Input(0)->NumElements();
    if (output_num >= kParallelDataNum) {
        int64_t total_size = output_num * static_cast<int64_t>(sizeof(T));
        int64_t thread_num = total_size / kSplitSize;
        if (thread_num > output_num || thread_num == 0) {
            thread_num = output_num;
        }
        size_t block_size = static_cast<size_t>(output_num / thread_num) * sizeof(T);
        size_t tail_block_size = static_cast<size_t>(output_num % thread_num) * sizeof(T) + block_size;
        auto copy_by_shard = [&](size_t start, size_t end) {
            for (size_t shard_idx = start; shard_idx < end; ++shard_idx) {
                size_t copy_size =
                    (shard_idx == static_cast<size_t>(thread_num - 1)) ? tail_block_size : block_size;
                int64_t data_offset = static_cast<int64_t>(shard_idx) * (output_num / thread_num);
                if (data_offset >= output_num) {
                    copy_status = KERNEL_STATUS_PARAM_INVALID;
                    KERNEL_LOG_ERROR("Pointer offset %ld more than %ld which is overflow", data_offset, output_num);
                    return;
                }
                auto dst = dst_ref + data_offset;
                auto src = src_ref + data_offset;
                auto mem_ret = memcpy_s(dst, copy_size, src, copy_size);
                if (mem_ret != EOK) {
                    KERNEL_LOG_ERROR("Initial memory copy failed[%d].Offerst is %ld, copy size is %lu", mem_ret,
                                     data_offset, copy_size);
                    copy_status = KERNEL_STATUS_INNER_ERROR;
                    return;
                }
            }
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, thread_num, 1, copy_by_shard),
                            "Initial output data failed!")
        if (copy_status != KERNEL_STATUS_OK) {
            return copy_status;
        }
    } else {
        for (int64_t data_idx = 0; data_idx < output_num; data_idx++) {
            dst_ref[data_idx] = src_ref[data_idx];
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename TI>
uint32_t ScatterNdMinCpuKernel::CheckScatterNdMinShapeAndData(const CpuKernelContext& ctx)
{
    auto x_shape = ctx.Input(0)->GetTensorShape();
    auto y_shape = ctx.Output(0)->GetTensorShape();
    auto indices_data = reinterpret_cast<TI*>(ctx.Input(1)->GetData());
    size_t x_rank = static_cast<size_t>(x_shape->GetDims());
    std::vector<int64_t> x_dims = x_shape->GetDimSizes();
    std::vector<int64_t> y_dims = y_shape->GetDimSizes();
    int64_t indices_num = ctx.Input(1)->NumElements();

    for (size_t dim_idx = 0; dim_idx < x_rank; dim_idx++) {
        if (x_dims[dim_idx] != y_dims[dim_idx]) {
            KERNEL_LOG_ERROR("The data shape of the input need be the same as the output");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    for (int64_t index_idx = 0; index_idx < indices_num; index_idx++) {
        if (indices_data[index_idx] < 0) {
            KERNEL_LOG_ERROR("Indices exist element which less than 0.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename T, typename TI>
uint32_t ScatterNdMinCpuKernel::DoScatterNdMinCompute(CpuKernelContext& ctx)
{
    auto ref_data = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto indices_ptr = reinterpret_cast<TI*>(ctx.Input(1)->GetData());
    auto updates_ptr = reinterpret_cast<T*>(ctx.Input(2)->GetData());
    auto ref_shape = ctx.Input(0)->GetTensorShape();
    auto indices_shape = ctx.Input(1)->GetTensorShape();
    auto updates_shape = ctx.Input(2)->GetTensorShape();

    KERNEL_CHECK_FALSE((CheckScatterNdMinShapeAndData<TI>(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "CheckShapeAndData failed.");
    KERNEL_CHECK_FALSE((indices_shape->GetDims() >= 1), KERNEL_STATUS_PARAM_INVALID,
                       "Indices tensor dims must be greater than or equal to 1.");

    int64_t ref_element_num = ctx.Input(0)->NumElements();
    std::vector<int64_t> indices_dims = indices_shape->GetDimSizes();
    std::vector<int64_t> ref_dims = ref_shape->GetDimSizes();
    const int64_t indices_outer_rank = indices_shape->GetDims() - 1;
    const int64_t updates_rank = updates_shape->GetDims();
    int64_t index_depth = static_cast<int64_t>(indices_dims.back());

    int64_t unit_num = 1;
    for (int64_t outer_idx = 0; outer_idx < indices_outer_rank; ++outer_idx) {
        unit_num *= indices_shape->GetDimSize(outer_idx);
    }
    int64_t update_unit_size = 1;
    for (int dim_pos = indices_outer_rank; dim_pos < updates_rank; ++dim_pos) {
        update_unit_size *= updates_shape->GetDimSize(dim_pos);
    }
    int64_t stride_count = ref_element_num;
    std::vector<int64_t> ref_strides(index_depth, 0);
    for (int64_t depth_idx = 0; depth_idx < index_depth; ++depth_idx) {
        ref_strides[depth_idx] = stride_count / ref_dims[depth_idx];
        stride_count = ref_strides[depth_idx];
    }

    for (int64_t unit_idx = 0; unit_idx < unit_num; ++unit_idx) {
        int64_t ref_offset = 0;
        for (int64_t depth_idx = 0; depth_idx < index_depth; ++depth_idx) {
            int64_t index_value = indices_ptr[unit_idx * index_depth + depth_idx];
            if (index_value < 0 || index_value >= ref_dims[depth_idx]) {
                KERNEL_LOG_ERROR("The indices[%ld] is so big or small", index_value);
                return KERNEL_STATUS_PARAM_INVALID;
            }
            ref_offset += index_value * ref_strides[depth_idx];
        }
        for (int64_t update_idx = 0; update_idx < update_unit_size; update_idx++) {
            if (ref_data[ref_offset + update_idx] > updates_ptr[unit_idx * update_unit_size + update_idx]) {
                ref_data[ref_offset + update_idx] = updates_ptr[unit_idx * update_unit_size + update_idx];
            }
        }
    }
    KERNEL_CHECK_FALSE((InitScatterNdMinOutput<T>(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID, "InitOutput failed.");
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kScatterNdMin, ScatterNdMinCpuKernel);
} // namespace aicpu
