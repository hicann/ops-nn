/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include "softmax_v2_aicpu.h"

#include <algorithm>
#include <complex>
#include <cstring>
#include "cpu_types.h"
#include "log.h"
#include "securec.h"
#include "utils/eigen_tensor.h"
#include <unsupported/Eigen/CXX11/Tensor>

namespace {
constexpr uint32_t kSoftmaxV2InputNum = 1;
constexpr uint32_t kSoftmaxV2OutputNum = 1;
constexpr uint32_t kDimType1 = 1U;
constexpr uint32_t kDimType2 = 2U;
constexpr uint32_t kDimType3 = 3U;
constexpr int64_t kDefaultAxis = static_cast<int64_t>(-1);
constexpr int64_t kParalledDataNum = 2048;
const char* const kSoftmaxV2 = "SoftmaxV2";
const std::vector<std::string> kSoftmaxV2Attr = {"axes"};

uint32_t NormalizeMultiAxes(std::vector<int64_t>& axes, int64_t dim_size)
{
    for (auto& axis : axes) {
        if (axis < -dim_size || axis >= dim_size) {
            KERNEL_LOG_ERROR("Invalid axis value %ld for input with %ld dimensions.", axis, dim_size);
            return aicpu::KERNEL_STATUS_PARAM_INVALID;
        }
        if (axis < 0) {
            axis += dim_size;
        }
    }
    std::sort(axes.begin(), axes.end());
    return aicpu::KERNEL_STATUS_OK;
}
} // namespace

namespace aicpu {
namespace detail {

template <typename T>
uint32_t MultiAxesComputeResult(T* input, T* output, T* dims_maximum, T* dims_exp_sum, int64_t total,
                                int64_t inner_size, int64_t outer_size)
{
    Eigen::TensorMap<Eigen::Tensor<T, kDimType2>> logits(input, outer_size, inner_size);
    Eigen::TensorMap<Eigen::Tensor<T, kDimType2>> dims_max(dims_maximum, outer_size, inner_size);

    // compute max num only on inner size direction
    Eigen::array<int, 1> reduction_axis = {1};
    Eigen::Tensor<T, 1> max_values = logits.maximum(reduction_axis); // shape=(outer_size,)

    for (int64_t index = 0; index < outer_size; ++index) {
        std::fill_n(dims_maximum + index * inner_size, inner_size, max_values(index));
    }

    for (int64_t index = 0; index < total; ++index) {
        int64_t outer_idx = index / inner_size;
        int64_t inner_idx = index % inner_size;
        T max_val = dims_max(outer_idx, inner_idx);
        T exp_val = Eigen::numext::exp(input[index] - max_val);
        dims_exp_sum[outer_idx] += exp_val;
    }

    for (int64_t index = 0; index < total; ++index) {
        int64_t outer_idx = index / inner_size;
        T sum = dims_exp_sum[outer_idx];
        if (sum == static_cast<T>(0)) {
            KERNEL_LOG_ERROR("SoftmaxV2 multi-axes sum is zero, division by zero.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        int64_t inner_idx = index % inner_size;
        T exp_val = Eigen::numext::exp(input[index] - dims_max(outer_idx, inner_idx));
        output[index] = exp_val / sum;
    }

    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MultiAxesComputeSoftmaxV2Kernel(const CpuKernelContext& ctx)
{
    auto input = static_cast<T*>(ctx.Input(0)->GetData());
    auto output = static_cast<T*>(ctx.Output(0)->GetData());

    std::vector<int64_t> axes = ctx.GetAttr("axes")->GetListInt();
    int64_t dim_size = static_cast<int64_t>(ctx.Input(0)->GetTensorShape()->GetDimSizes().size());

    uint32_t ret = NormalizeMultiAxes(axes, dim_size);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    std::vector<int64_t> dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    int64_t total = static_cast<int64_t>(ctx.Input(0)->NumElements());
    int64_t inner_size = 1;
    for (auto axis : axes) {
        inner_size *= dims[axis];
    }

    KERNEL_CHECK_FALSE((inner_size > 0), KERNEL_STATUS_PARAM_INVALID, "inner size must be greater than 0 but is [%ld].",
                       inner_size);
    KERNEL_CHECK_FALSE((total % inner_size == 0), KERNEL_STATUS_PARAM_INVALID,
                       "total nums[%ld] must be an integer multiple of inner size[%ld].", total, inner_size);
    int64_t outer_size = total / inner_size;

    KERNEL_LOG_INFO("multi axis total=%ld, inner_size=%ld, outer_size=%ld.", total, inner_size, outer_size);
    KERNEL_CHECK_FALSE(CheckInt64MulOverflow(inner_size, outer_size), KERNEL_STATUS_INNER_ERROR,
                       "the product of inner_size %ld and outer_size %ld exceeds INT64_MAX", inner_size, outer_size);
    std::unique_ptr<T[]> dims_exp_sum(new (std::nothrow) T[static_cast<size_t>(total)]);
    KERNEL_CHECK_NULLPTR(dims_exp_sum, KERNEL_STATUS_INNER_ERROR, "Fail to allocate dims_exp_sum.");
    std::unique_ptr<T[]> dims_maximum(new (std::nothrow) T[static_cast<size_t>(total)]);
    KERNEL_CHECK_NULLPTR(dims_maximum, KERNEL_STATUS_INNER_ERROR, "Fail to allocate dims_maximum.");

    auto result = BiggerMemSet(dims_exp_sum.get(), static_cast<size_t>(total) * sizeof(T), 0,
                               static_cast<size_t>(total) * sizeof(T));
    if (!result) {
        KERNEL_LOG_ERROR("softmaxv2 op multi axes process bigger memset failed.");
        return KERNEL_STATUS_PARAM_INVALID;
    }

    ret = MultiAxesComputeResult<T>(input, output, dims_maximum.get(), dims_exp_sum.get(), total, inner_size,
                                    outer_size);
    if (ret != KERNEL_STATUS_OK) {
        return ret;
    }

    KERNEL_LOG_INFO("multi axis compute end.");
    return KERNEL_STATUS_OK;
}

template <typename T>
void ComputeSoftmaxV2Serial(T* input, T* output, T* dims_exp_sum, T* dims_maximum, int64_t total, int64_t inner_size,
                            int64_t outer_size, int64_t length, int64_t pivot_len)
{
    // Note: the shape of Eigen::Tensor logits and softmax is reverse of input Tensor
    Eigen::TensorMap<Eigen::Tensor<T, kDimType3>, Eigen::Aligned> logits(input, inner_size, static_cast<int>(pivot_len),
                                                                         outer_size);
    Eigen::TensorMap<Eigen::Tensor<T, kDimType1>, Eigen::Aligned> dims_sum(dims_exp_sum, length);
    Eigen::TensorMap<Eigen::Tensor<T, kDimType2>, Eigen::Aligned> dims_max(dims_maximum, inner_size, outer_size);
    Eigen::array<int, 1> softmax_axes{{1}};
    dims_max = logits.maximum(softmax_axes);
    for (int64_t index = 0, index_dst = 0, index_batch = 0, count_step = 0; index < total; index++) {
        if (index % inner_size == 0 && index != 0) {
            count_step++;
            if (count_step == pivot_len) {
                count_step = 0;
                index_batch += inner_size;
            }
            index_dst = index_batch;
        }
        *(output + index) = Eigen::numext::exp(*(input + index) - dims_maximum[index_dst]);
        dims_exp_sum[index_dst] += (*(output + index));
        index_dst++;
    }
    dims_sum = dims_sum.inverse();
    for (int64_t index = 0, index_dst = 0, index_batch = 0, count_step = 0; index < total; index++) {
        if (index % inner_size == 0 && index != 0) {
            count_step++;
            if (count_step == pivot_len) {
                count_step = 0;
                index_batch += inner_size;
            }
            index_dst = index_batch;
        }
        *(output + index) = (*(output + index)) * (dims_exp_sum[index_dst]);
        index_dst++;
    }
}

template <typename T>
uint32_t ComputeSoftmaxV2Parallel(const CpuKernelContext& ctx, T* input, T* output, T* dims_exp_sum, T* dims_maximum,
                                  int64_t length, int64_t inner_size, int64_t pivot_len, uint32_t cores)
{
    int64_t per_unit_size{length / std::min(std::max(1L, static_cast<int64_t>(cores) - 2L), length)};
    const T constant_one(1.0);
    KERNEL_HANDLE_ERROR(
        aicpu::CpuKernelUtils::ParallelFor(
            ctx, length, per_unit_size,
            [&](int64_t begin, int64_t end) {
                for (int64_t index = begin, outer_index, index_base; index < end; ++index) {
                    outer_index = index / inner_size;
                    index_base = outer_index * pivot_len * inner_size + index % inner_size;
                    dims_maximum[index] = *(input + index_base);
                    for (int64_t inner_index = 0, index_dst = index_base; inner_index < pivot_len; ++inner_index) {
                        if (*(input + index_dst) > dims_maximum[index]) {
                            dims_maximum[index] = *(input + index_dst);
                        }
                        index_dst += inner_size;
                    }
                    for (int64_t inner_index = 0, index_dst = index_base; inner_index < pivot_len; ++inner_index) {
                        *(output + index_dst) = Eigen::numext::exp(*(input + index_dst) - dims_maximum[index]);
                        dims_exp_sum[index] += (*(output + index_dst));
                        index_dst += inner_size;
                    }
                    dims_exp_sum[index] = constant_one / dims_exp_sum[index];
                    for (int64_t inner_index = 0, index_dst = index_base; inner_index < pivot_len; ++inner_index) {
                        *(output + index_dst) = *(output + index_dst) * dims_exp_sum[index];
                        index_dst += inner_size;
                    }
                }
            }),
        "CpuKernelUtils::ParallelFor failed.");
    return KERNEL_STATUS_OK;
}

struct SoftmaxV2Sizes {
    int64_t pivot;
    int64_t inner_size;
    int64_t outer_size;
    int64_t length;
};

SoftmaxV2Sizes ComputeSoftmaxV2Sizes(const std::vector<int64_t>& dims, int64_t axis)
{
    int64_t dim_size = static_cast<int64_t>(dims.size());
    int64_t pivot = (axis >= 0 ? axis : dim_size + axis);
    int64_t inner_size = 1;
    int64_t outer_size = 1;
    for (int64_t index = 0; index < dim_size; index++) {
        if (index > pivot) {
            inner_size *= static_cast<int64_t>(dims[index]);
        }
        if (index < pivot) {
            outer_size *= static_cast<int64_t>(dims[index]);
        }
    }
    return {pivot, inner_size, outer_size, inner_size * outer_size};
}

template <typename T>
uint32_t ComputeSoftmaxV2Kernel(const CpuKernelContext& ctx)
{
    auto input = static_cast<T*>(ctx.Input(0)->GetData());
    auto output = static_cast<T*>(ctx.Output(0)->GetData());
    // axes default values = [-1]
    std::vector<int64_t> axes{-1};
    if (ctx.GetAttr("axes") != nullptr) {
        axes = ctx.GetAttr("axes")->GetListInt();
    }

    KERNEL_LOG_INFO("attr axes size is %zu.", axes.size());
    if (axes.size() > 1) {
        return MultiAxesComputeSoftmaxV2Kernel<T>(ctx);
    }

    int64_t total = static_cast<int64_t>(ctx.Input(0)->NumElements());
    std::vector<int64_t> dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    if (cores < 1) {
        return static_cast<uint32_t>(KERNEL_STATUS_INNER_ERROR);
    }

    // scalar: output = e^i / sum(e^i) = 1
    if (dims.empty()) {
        output[0] = static_cast<T>(1);
        return KERNEL_STATUS_OK;
    }

    auto sizes = ComputeSoftmaxV2Sizes(dims, axes[0]);
    std::unique_ptr<T[]> dims_exp_sum(new (std::nothrow) T[sizes.length]);
    KERNEL_CHECK_NULLPTR(dims_exp_sum, KERNEL_STATUS_INNER_ERROR, "Fail to allocate dims_exp_sum.");
    std::unique_ptr<T[]> dims_maximum(new (std::nothrow) T[sizes.length]);
    KERNEL_CHECK_NULLPTR(dims_maximum, KERNEL_STATUS_INNER_ERROR, "Fail to allocate dims_maximum.");

    KERNEL_LOG_INFO("inner_size is %ld, outer_size is %ld.", sizes.inner_size, sizes.outer_size);
    auto result = BiggerMemSet(dims_exp_sum.get(), static_cast<size_t>(sizes.length) * sizeof(T), 0,
                               static_cast<size_t>(sizes.length) * sizeof(T));
    if (!result) {
        KERNEL_LOG_ERROR("softmaxv2 op BiggerMemSet failed, length = %ld, memsetlen = %ld.", sizes.length,
                         static_cast<size_t>(sizes.length) * sizeof(T));
        return KERNEL_STATUS_PARAM_INVALID;
    }

    int64_t pivot_len = static_cast<int64_t>(dims[sizes.pivot]);
    if (total > kParalledDataNum) {
        return ComputeSoftmaxV2Parallel<T>(ctx, input, output, dims_exp_sum.get(), dims_maximum.get(), sizes.length,
                                           sizes.inner_size, pivot_len, cores);
    }
    ComputeSoftmaxV2Serial<T>(input, output, dims_exp_sum.get(), dims_maximum.get(), total, sizes.inner_size,
                              sizes.outer_size, sizes.length, pivot_len);
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t ComputeSoftmaxV2(const CpuKernelContext& ctx)
{
    uint32_t result = ComputeSoftmaxV2Kernel<T>(ctx);
    if (result != 0) {
        KERNEL_LOG_ERROR("SoftmaxV2 compute failed.");
    }
    return result;
}

KernelStatus SoftmaxV2ExtraCheck(const CpuKernelContext& ctx)
{
    if (ctx.Input(0)->GetDataType() != ctx.Output(0)->GetDataType()) {
        KERNEL_LOG_ERROR("The data type of the input [%s] need be the same as the ouput [%s].",
                         DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    std::vector<int64_t> input_dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> output_dims = ctx.Output(0)->GetTensorShape()->GetDimSizes();
    if (input_dims.size() != output_dims.size()) {
        KERNEL_LOG_ERROR("The data dim of the input size [%lu] need be the same as the output size [%lu].",
                         input_dims.size(), output_dims.size());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    for (size_t index = 0; index < input_dims.size(); index++) {
        if (input_dims[index] != output_dims[index]) {
            KERNEL_LOG_ERROR("The input data dim[%lu]=%ld need be the same as the output dim[%lu]=%ld.", index,
                             input_dims[index], index, output_dims[index]);
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    std::vector<int64_t> axes = ctx.GetAttr("axes")->GetListInt();
    if (axes.size() < 1) {
        KERNEL_LOG_ERROR("The Attributes axes size is %ld, but size must grater than 0.", axes.size());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    int64_t size = static_cast<int64_t>(input_dims.size());
    for (int64_t ax : axes) {
        int64_t target_axis = 0;
        if (ax >= 0) {
            target_axis = ax;
        } else if (ax != kDefaultAxis) {
            target_axis = ax + size;
        }
        if (((axes[0] != kDefaultAxis) && (axes[0] != 0)) && ((target_axis < 0) || (target_axis >= size))) {
            KERNEL_LOG_ERROR("The Attributes axis:%ld is out of range of input size %ld.", target_axis, size);
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }
    return KERNEL_STATUS_OK;
}

uint32_t SoftmaxV2Check(CpuKernelContext& ctx)
{
    return NormalCheck(ctx, kSoftmaxV2InputNum, kSoftmaxV2OutputNum, kSoftmaxV2Attr) ? KERNEL_STATUS_PARAM_INVALID :
                                                                                       SoftmaxV2ExtraCheck(ctx);
}

// DT_FLOAT16, DT_FLOAT, DT_DOUBLE
uint32_t SoftmaxV2Compute(const CpuKernelContext& ctx)
{
    DataType input_type{ctx.Input(0)->GetDataType()};
    auto input_elements = ctx.Input(0)->NumElements();
    auto output_elements = ctx.Output(0)->NumElements();
    if ((input_elements == 0) || (output_elements == 0)) {
        KERNEL_LOG_INFO("element number is zero.");
        return KERNEL_STATUS_OK;
    }

    switch (input_type) {
        case DT_FLOAT16:
            return ComputeSoftmaxV2<Eigen::half>(ctx);
        case DT_FLOAT:
            return ComputeSoftmaxV2<std::float_t>(ctx);
        case DT_DOUBLE:
            return ComputeSoftmaxV2<std::double_t>(ctx);
        default:
            KERNEL_LOG_ERROR("Unsupported input data type [%s].", DTypeStr(input_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}
} // namespace detail

uint32_t SoftmaxV2CpuKernel::Compute(CpuKernelContext& ctx)
{
    return detail::SoftmaxV2Check(ctx) ? KERNEL_STATUS_PARAM_INVALID : detail::SoftmaxV2Compute(ctx);
}

REGISTER_CPU_KERNEL(kSoftmaxV2, SoftmaxV2CpuKernel);
} // namespace aicpu
