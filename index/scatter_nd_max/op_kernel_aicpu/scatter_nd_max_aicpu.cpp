/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_nd_max_aicpu.h"

#include <atomic>
#include <iostream>

#include "Eigen/Dense"
#include "cpu_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_util.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const int32_t kScatterNdMaxInputNum = 3;
const int32_t kScatterNdMaxOutputNum = 1;
const char* const kScatterNdMax = "ScatterNdMax";
const int64_t kParallelDataNum = 32 * 1024;
const int32_t kSplitSize = 64 * 1024;

#define DO_SCATTER_ND_MAX_COMPUTE_CASE(DTYPE, TYPE, ITYPE, CTX) \
    case (DTYPE):                                                \
    do {                                                         \
        if ((ITYPE) == DT_INT32) {                               \
            return DoScatterNdMaxCompute<TYPE, int32_t>(CTX);    \
        } else {                                                 \
            return DoScatterNdMaxCompute<TYPE, int64_t>(CTX);    \
        }                                                        \
    } while (0)
} // namespace

namespace aicpu {
uint32_t ScatterNdMaxCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kScatterNdMaxInputNum, kScatterNdMaxOutputNum),
                        "ScatterNdMax check input or output is failed.");
    Tensor* input_data = ctx.Input(0);
    Tensor* input_indices = ctx.Input(1);
    auto data_type = input_data->GetDataType();
    auto data_type_indices = input_indices->GetDataType();
    KERNEL_CHECK_FALSE((data_type_indices == DT_INT32 || data_type_indices == DT_INT64), KERNEL_STATUS_PARAM_INVALID,
                       "Input[1] data type[%s] is unsupported", DTypeStr(data_type_indices).c_str());
    switch (data_type) {
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_DOUBLE, double, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_FLOAT, float, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_FLOAT16, Eigen::half, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_INT16, int16_t, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_INT32, int32_t, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_INT64, int64_t, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_INT8, int8_t, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_UINT16, uint16_t, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_UINT32, uint32_t, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_UINT64, uint64_t, data_type_indices, ctx);
        DO_SCATTER_ND_MAX_COMPUTE_CASE(DT_UINT8, uint8_t, data_type_indices, ctx);
        default: {
            std::string err_msg = ConcatString(
                "Input[0] data type[", DTypeStr(data_type), "] is unsupported.",
                "It should be double|float|float16|int16|int32|int64|int8|uint16|uint32|uint64|uint8");
            KERNEL_LOG_ERROR("%s", err_msg.c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
}

template <typename T>
uint32_t ScatterNdMaxCpuKernel::InitScatterNdMaxOutput(const CpuKernelContext& ctx)
{
    auto input_ref = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto output_ref = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    std::atomic<uint32_t> work_ret(KERNEL_STATUS_OK);
    int64_t total_value_num = ctx.Input(0)->NumElements();
    if (total_value_num >= kParallelDataNum) {
        int64_t initial_size = total_value_num * static_cast<int64_t>(sizeof(T));
        int64_t max_thread_num = initial_size / kSplitSize;
        if (max_thread_num > total_value_num || max_thread_num == 0) {
            max_thread_num = total_value_num;
        }
        KERNEL_CHECK_FALSE((max_thread_num != 0), KERNEL_STATUS_PARAM_INVALID,
                           "max_thread_num must not be equal to zero.");
        size_t chunk_bytes = static_cast<size_t>(total_value_num / max_thread_num) * sizeof(T);
        size_t final_chunk_bytes = static_cast<size_t>(total_value_num % max_thread_num) * sizeof(T) + chunk_bytes;
        auto copy_output_shard = [&](size_t begin_idx, size_t end_idx) {
            for (size_t shard_id = begin_idx; shard_id < end_idx; ++shard_id) {
                size_t copy_bytes =
                    (shard_id == static_cast<size_t>(max_thread_num - 1)) ? final_chunk_bytes : chunk_bytes;
                int64_t data_base = static_cast<int64_t>(shard_id) * (total_value_num / max_thread_num);
                if (data_base >= total_value_num) {
                    work_ret = KERNEL_STATUS_PARAM_INVALID;
                    KERNEL_LOG_ERROR("Pointer offset %ld more than %ld which is overflow", data_base, total_value_num);
                    return;
                }
                auto output_ptr = output_ref + data_base;
                auto input_ptr = input_ref + data_base;
                auto mem_ret = memcpy_s(output_ptr, copy_bytes, input_ptr, copy_bytes);
                if (mem_ret != EOK) {
                    KERNEL_LOG_ERROR("Initial memory copy failed[%d].Offerst is %ld, copy size is %lu", mem_ret,
                                     data_base, copy_bytes);
                    work_ret = KERNEL_STATUS_INNER_ERROR;
                    return;
                }
            }
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, max_thread_num, 1, copy_output_shard),
                            "Initial output data failed!")
        if (work_ret != KERNEL_STATUS_OK) {
            return work_ret;
        }
    } else {
        for (int64_t value_idx = 0; value_idx < total_value_num; value_idx++) {
            output_ref[value_idx] = input_ref[value_idx];
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename TI>
uint32_t ScatterNdMaxCpuKernel::CheckScatterNdMaxShapeAndData(const CpuKernelContext& ctx)
{
    auto shape_ref = ctx.Input(0)->GetTensorShape();
    auto shape_output = ctx.Output(0)->GetTensorShape();
    auto input_indices = reinterpret_cast<TI*>(ctx.Input(1)->GetData());
    size_t value_dim_num_ref = static_cast<size_t>(shape_ref->GetDims());
    std::vector<int64_t> value_dim_ref = shape_ref->GetDimSizes();
    std::vector<int64_t> value_dim_output = shape_output->GetDimSizes();
    int64_t indices_value_num = ctx.Input(1)->NumElements();

    for (size_t i = 0; i < value_dim_num_ref; i++) {
        if (value_dim_ref[i] != value_dim_output[i]) {
            KERNEL_LOG_ERROR("The data shape of the input need be the same as the output");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    for (int64_t i = 0; i < indices_value_num; i++) {
        if (input_indices[i] < 0) {
            KERNEL_LOG_ERROR("Indices exist element which less than 0.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename T, typename TI>
uint32_t ScatterNdMaxCpuKernel::DoScatterNdMaxCompute(CpuKernelContext& ctx)
{
    auto input_ref = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto input_indices = reinterpret_cast<TI*>(ctx.Input(1)->GetData());
    auto input_updates = reinterpret_cast<T*>(ctx.Input(2)->GetData());
    auto shape_ref = ctx.Input(0)->GetTensorShape();
    auto shape_indices = ctx.Input(1)->GetTensorShape();
    auto shape_updates = ctx.Input(2)->GetTensorShape();

    KERNEL_CHECK_FALSE((CheckScatterNdMaxShapeAndData<TI>(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID,
                       "CheckShapeAndData failed.");
    KERNEL_CHECK_FALSE((shape_indices->GetDims() >= 1), KERNEL_STATUS_PARAM_INVALID,
                       "Indices tensor dims must be greater than or equal to 1.");

    int64_t total_value_num = ctx.Input(0)->NumElements();
    std::vector<int64_t> value_dim_indices = shape_indices->GetDimSizes();
    std::vector<int64_t> value_dim_ref = shape_ref->GetDimSizes();
    const int64_t value_dim_num_x2 = shape_indices->GetDims() - 1;
    const int64_t value_dim_num_x3 = shape_updates->GetDims();
    int64_t indices_unit_rank = static_cast<int64_t>(value_dim_indices.back());

    int64_t num_units = 1;
    for (int64_t i = 0; i < value_dim_num_x2; ++i) {
        num_units *= shape_indices->GetDimSize(i);
    }
    int64_t unit_size = 1;
    for (int i = value_dim_num_x2; i < value_dim_num_x3; ++i) {
        unit_size *= shape_updates->GetDimSize(i);
    }
    int64_t count = total_value_num;
    std::vector<int64_t> dims_to_count(indices_unit_rank, 0);
    for (int64_t i = 0; i < indices_unit_rank; ++i) {
        dims_to_count[i] = count / value_dim_ref[i];
        count = dims_to_count[i];
    }

    for (int64_t i = 0; i < num_units; ++i) {
        int64_t offset = 0;
        for (int64_t j = 0; j < indices_unit_rank; ++j) {
            int64_t index = input_indices[i * indices_unit_rank + j];
            if (index < 0 || index >= value_dim_ref[j]) {
                KERNEL_LOG_ERROR("The indices[%ld] is so big or small", index);
                return KERNEL_STATUS_PARAM_INVALID;
            }
            offset += index * dims_to_count[j];
        }
        for (int64_t j = 0; j < unit_size; j++) {
            if (input_ref[offset + j] < input_updates[i * unit_size + j]) {
                input_ref[offset + j] = input_updates[i * unit_size + j];
            }
        }
    }
    KERNEL_CHECK_FALSE((InitScatterNdMaxOutput<T>(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID, "InitOutput failed.");
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kScatterNdMax, ScatterNdMaxCpuKernel);
} // namespace aicpu
