/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log_softmax_v2_aicpu.h"

#include <securec.h>
#include <atomic>
#include <functional>
#include "cpu_types.h"
#include "log.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kInputNum = 1;
constexpr uint32_t kOutputNum = 1;
constexpr uint32_t kDimType1 = 1;
constexpr uint32_t kDimType2 = 2;
constexpr uint32_t kDimType3 = 3;
constexpr int64_t kParalleledDataSize = 4 * 1024;
const char* const kLogSoftmaxV2 = "LogSoftmaxV2";

struct SoftmaxShapeInfo {
    size_t pivot;
    int64_t inner_size{1};
    int64_t outer_size{1};
    int64_t length{1};
    int64_t total;
    std::vector<int64_t> dims;
};

SoftmaxShapeInfo ComputeShapeInfo(const aicpu::CpuKernelContext& ctx)
{
    SoftmaxShapeInfo info;
    info.total = ctx.Input(0)->NumElements();
    info.dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    std::vector<std::int64_t> axes{-1};
    if (ctx.GetAttr("axes") != nullptr) {
        axes = ctx.GetAttr("axes")->GetListInt();
    }
    if (axes[0] >= 0) {
        info.pivot = static_cast<size_t>(axes[0]);
    } else {
        info.pivot = info.dims.size() + static_cast<size_t>(axes[0]);
    }
    for (size_t index = 0; index < info.dims.size(); index++) {
        if (index > info.pivot) {
            info.inner_size *= info.dims[index];
        }
        if (index < info.pivot) {
            info.outer_size *= info.dims[index];
        }
    }
    info.length = info.inner_size * info.outer_size;
    return info;
}

template <typename Op>
void IterateBatches(int64_t total, int64_t inner_size, int64_t dim_pivot, Op op)
{
    for (int64_t index = 0, index_dst = 0, index_batch = 0, step = 0; index < total; index++) {
        if (index % inner_size == 0 && index != 0) {
            step++;
            if (step == dim_pivot) {
                step = 0;
                index_batch += inner_size;
            }
            index_dst = index_batch;
        }
        op(index, index_dst);
        index_dst++;
    }
}

template <typename T>
aicpu::KernelStatus ComputeSequential(T* input, T* output, T* dims_exp_sum, T* dims_maximum,
                                      const SoftmaxShapeInfo& info)
{
    Eigen::TensorMap<Eigen::Tensor<T, kDimType3>, Eigen::Aligned> logits(
        input, info.inner_size, static_cast<int>(info.dims[info.pivot]), info.outer_size);
    Eigen::TensorMap<Eigen::Tensor<T, kDimType1>, Eigen::Aligned> dims_sum(dims_exp_sum, info.length);
    Eigen::TensorMap<Eigen::Tensor<T, kDimType2>, Eigen::Aligned> dims_max(dims_maximum, info.inner_size,
                                                                           info.outer_size);
    Eigen::array<int, 1> softmax_axes{{1}};
    dims_max = logits.maximum(softmax_axes);

    IterateBatches(info.total, info.inner_size, info.dims[info.pivot], [&](int64_t index, int64_t index_dst) {
        *(output + index) = Eigen::numext::exp(*(input + index) - dims_maximum[index_dst]);
        dims_exp_sum[index_dst] += (*(output + index));
    });

    dims_sum = dims_sum.inverse();

    IterateBatches(info.total, info.inner_size, info.dims[info.pivot], [&](int64_t index, int64_t index_dst) {
        *(output + index) = (*(output + index)) * (dims_exp_sum[index_dst]);
        *(output + index) = Eigen::numext::log(*(output + index));
    });

    return aicpu::KERNEL_STATUS_OK;
}

template <typename T>
void ComputeOneBatch(T* input, T* output, T* dims_exp_sum, T* dims_maximum, int64_t index, int64_t inner_size,
                     int64_t dim_length, std::atomic<bool>& failed)
{
    int64_t outer_index = index / inner_size;
    int64_t index_base = outer_index * dim_length * inner_size + index % inner_size;

    dims_maximum[index] = *(input + index_base);
    for (int64_t inner_index = 0, index_dst = index_base; inner_index < dim_length; ++inner_index) {
        if (*(input + index_dst) > dims_maximum[index]) {
            dims_maximum[index] = *(input + index_dst);
        }
        index_dst += inner_size;
    }
    for (int64_t inner_index = 0, index_dst = index_base; inner_index < dim_length; ++inner_index) {
        *(output + index_dst) = Eigen::numext::exp(*(input + index_dst) - dims_maximum[index]);
        dims_exp_sum[index] += (*(output + index_dst));
        index_dst += inner_size;
    }
    if (dims_exp_sum[index] == static_cast<T>(0)) {
        KERNEL_LOG_ERROR("LogSoftmaxV2 sum of exp is zero, division by zero.");
        failed.store(true, std::memory_order_relaxed);
        return;
    }
    dims_exp_sum[index] = static_cast<T>(1.0) / dims_exp_sum[index];
    for (int64_t inner_index = 0, index_dst = index_base; inner_index < dim_length; ++inner_index) {
        *(output + index_dst) = *(output + index_dst) * dims_exp_sum[index];
        *(output + index_dst) = Eigen::numext::log(*(output + index_dst));
        index_dst += inner_size;
    }
}

template <typename T>
aicpu::KernelStatus ComputeParallel(const aicpu::CpuKernelContext& ctx, T* input, T* output, T* dims_exp_sum,
                                    T* dims_maximum, const SoftmaxShapeInfo& info, uint32_t cores)
{
    std::int64_t per_unit_size{info.length / std::min(std::max(1L, static_cast<int64_t>(cores) - 2L), info.length)};
    std::atomic<bool> failed{false};
    auto sharder = [&](std::int64_t begin, std::int64_t end) {
        for (int64_t index = begin; index < end; ++index) {
            ComputeOneBatch<T>(input, output, dims_exp_sum, dims_maximum, index, info.inner_size, info.dims[info.pivot],
                               failed);
        }
    };
    aicpu::CpuKernelUtils::ParallelFor(ctx, info.length, per_unit_size, sharder);
    if (failed.load(std::memory_order_relaxed)) {
        return aicpu::KERNEL_STATUS_PARAM_INVALID;
    }
    return aicpu::KERNEL_STATUS_OK;
}
} // namespace

namespace aicpu {
uint32_t LogSoftmaxV2CpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kLogSoftmaxV2);
    KERNEL_HANDLE_ERROR(LogSoftmaxV2Check(ctx), "[%s] check params failed.", kLogSoftmaxV2);

    auto data_type = ctx.Input(0)->GetDataType();
    uint32_t result = KERNEL_STATUS_OK;
    switch (data_type) {
        case (DT_FLOAT16):
            result = LogSoftmaxV2Compute<Eigen::half>(ctx);
            break;
        case (DT_FLOAT):
            result = LogSoftmaxV2Compute<float>(ctx);
            break;
        case (DT_DOUBLE):
            result = LogSoftmaxV2Compute<double>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("LogSoftmaxV2 kernel data type [%s] not support.", DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    if (result != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("LogSoftmaxV2 kernel compute failed.");
    }
    return result;
}

KernelStatus LogSoftmaxV2CpuKernel::LogSoftmaxV2Check(const CpuKernelContext& ctx)
{
    KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "get input failed.");
    KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetTensorShape(), KERNEL_STATUS_PARAM_INVALID, "Get input tensor shape failed.");
    KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "get output failed.");
    KERNEL_CHECK_NULLPTR(ctx.GetAttr("axes"), KERNEL_STATUS_PARAM_INVALID, "get axes failed.");
    std::vector<std::int64_t> axes_data = ctx.GetAttr("axes")->GetListInt();
    KERNEL_CHECK_FALSE((axes_data.size() == 1), KERNEL_STATUS_PARAM_INVALID, "axes is out of shape");
    int64_t axes = axes_data[0];
    if (ctx.Input(0)->GetTensorShape()->GetDims() > 0) {
        KERNEL_CHECK_FALSE((axes < ctx.Input(0)->GetTensorShape()->GetDims()), KERNEL_STATUS_PARAM_INVALID,
                           "axes is larger than input dims - 1");
        KERNEL_CHECK_FALSE((axes >= -ctx.Input(0)->GetTensorShape()->GetDims()), KERNEL_STATUS_PARAM_INVALID,
                           "axes is lower than -input dims");
    }
    std::vector<int64_t> shape_input = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> shape_output = ctx.Output(0)->GetTensorShape()->GetDimSizes();
    KERNEL_CHECK_FALSE((shape_input.size() == shape_output.size()), KERNEL_STATUS_PARAM_INVALID,
                       "The output shape size should be same as the input shape size");
    return KERNEL_STATUS_OK;
}

template <typename T>
KernelStatus LogSoftmaxV2CpuKernel::LogSoftmaxV2Compute(const CpuKernelContext& ctx)
{
    auto input = static_cast<T*>(ctx.Input(0)->GetData());
    auto output = static_cast<T*>(ctx.Output(0)->GetData());
    std::int64_t total = ctx.Input(0)->NumElements();

    if (ctx.Input(0)->GetTensorShape()->GetDims() == 0 && total == 1) {
        output[0] = static_cast<T>(std::log(std::exp(input[0]) / std::exp(input[0])));
        KERNEL_LOG_DEBUG("LogSoftmaxV2 handling scalar scenarios.");
        return KERNEL_STATUS_OK;
    }

    auto info = ComputeShapeInfo(ctx);
    if (info.inner_size == 0) {
        KERNEL_LOG_INFO("LogSoftmaxV2 inner_size is 0, skip compute.");
        return KERNEL_STATUS_OK;
    }
    uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    if (cores < 1) {
        return KERNEL_STATUS_INNER_ERROR;
    }

    std::unique_ptr<T[]> dims_exp_sum(new (std::nothrow) T[info.length]);
    KERNEL_CHECK_NULLPTR(dims_exp_sum, KERNEL_STATUS_INNER_ERROR, "alloc dims exp sum failed.");
    std::unique_ptr<T[]> dims_maximum(new (std::nothrow) T[info.length]);
    KERNEL_CHECK_NULLPTR(dims_maximum, KERNEL_STATUS_INNER_ERROR, "alloc dims max sum failed.");

    KERNEL_LOG_DEBUG("inner_size is %ld, outer_size is %ld", info.inner_size, info.outer_size);
    auto ret = BiggerMemSet(dims_exp_sum.get(), static_cast<size_t>(info.length) * sizeof(T), 0,
                            static_cast<size_t>(info.length) * sizeof(T));
    if (!ret) {
        KERNEL_LOG_ERROR("LogSoftmaxV2 BiggerMemSet failed.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    int64_t data_size = info.total * static_cast<int64_t>(sizeof(T));
    if (data_size <= kParalleledDataSize) {
        return ComputeSequential<T>(input, output, dims_exp_sum.get(), dims_maximum.get(), info);
    }
    return ComputeParallel<T>(ctx, input, output, dims_exp_sum.get(), dims_maximum.get(), info, cores);
}

REGISTER_CPU_KERNEL(kLogSoftmaxV2, LogSoftmaxV2CpuKernel);
} // namespace aicpu
