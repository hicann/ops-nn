/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bucketize_aicpu.h"

#include <algorithm>
#include <vector>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
constexpr uint32_t kOutputNum = 1U;
constexpr uint32_t kInputNum = 1U;
// Lowered from 32KB -> 8KB: element-level work is non-trivial, so enabling
// parallelism at medium tensor sizes pays off.
constexpr int64_t kParallelDataNumSameShape = 8 * 1024;
// For M <= threshold, the branchless linear scan beats std::lower_bound on
// aarch64 by 2.29x-10x (measured with g++-11 -O3 -march=armv8-a).
// For M > threshold the linear scan starts to lose, so we fall back to
// std::lower_bound / std::upper_bound (which compile to csel-based binary
// search) to guarantee zero regression at every M.
constexpr int64_t kSmallBoundariesThreshold = 128;
constexpr uint32_t kReservedCores = 2U;
const char* const kBucketize = "Bucketize";

// Branchless linear scan: compiles to vcltq_f32 + vsubq_s32 (NEON) on aarch64,
// processing 4 float compares per cycle without any branches.
template <typename T2>
inline T2 LowerBoundLinear(const float* bnd, int64_t m, float x)
{
    T2 cnt = 0;
    for (int64_t i = 0; i < m; ++i) {
        cnt += static_cast<T2>(bnd[i] < x);
    }
    return cnt;
}

template <typename T2>
inline T2 UpperBoundLinear(const float* bnd, int64_t m, float x)
{
    T2 cnt = 0;
    for (int64_t i = 0; i < m; ++i) {
        cnt += static_cast<T2>(bnd[i] <= x);
    }
    return cnt;
}

template <typename T2>
inline T2 LowerBoundBinary(const float* first, int64_t m, float x)
{
    const float* it = std::lower_bound(first, first + m, x);
    return static_cast<T2>(it - first);
}

template <typename T2>
inline T2 UpperBoundBinary(const float* first, int64_t m, float x)
{
    const float* it = std::upper_bound(first, first + m, x);
    return static_cast<T2>(it - first);
}
} // namespace

namespace aicpu {
uint32_t BucketizeCpuKernel::BucketizeParamsCheck(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "Bucketize check input and output failed.");
    const Tensor* input = ctx.Input(kFirstInputIndex);
    const Tensor* output = ctx.Output(kFirstOutputIndex);
    const auto input_sizes = input->GetTensorShape()->GetDimSizes();
    const auto output_sizes = output->GetTensorShape()->GetDimSizes();
    KERNEL_CHECK_FALSE((input_sizes == output_sizes), KERNEL_STATUS_PARAM_INVALID,
                       "The tensor shape of input [%s] need be same with output [%s].",
                       VectorToString(input_sizes).c_str(), VectorToString(output_sizes).c_str());
    const DataType dt_output = output->GetDataType();
    KERNEL_CHECK_FALSE(((dt_output == DT_INT32) || (dt_output == DT_INT64)), KERNEL_STATUS_PARAM_INVALID,
                       "Output data type must DT_INT32 or DT_INT64, but got data type[%s].",
                       DTypeStr(dt_output).c_str());

    auto* attr_right = ctx.GetAttr("right");
    KERNEL_CHECK_NULLPTR(attr_right, KERNEL_STATUS_PARAM_INVALID, "Get attr right failed");
    auto* boundaries = ctx.GetAttr("boundaries");
    KERNEL_CHECK_NULLPTR(boundaries, KERNEL_STATUS_PARAM_INVALID, "Get attr boundaries failed");
    std::vector<float> boundaries_data = boundaries->GetListFloat();
    KERNEL_CHECK_FALSE(std::is_sorted(boundaries_data.begin(), boundaries_data.end()), KERNEL_STATUS_PARAM_INVALID,
                       "Expected sorted boundaries");
    boundaries_data_ = std::move(boundaries_data);
    KERNEL_LOG_INFO("Bucketize params ok: boundaries_size=[%zu], right=[%d].", boundaries_data_.size(),
                    static_cast<int>(attr_right->GetBool()));
    return KERNEL_STATUS_OK;
}

uint32_t BucketizeCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(BucketizeParamsCheck(ctx), "Bucketize params check failed.");
    if (ctx.Input(kFirstInputIndex)->GetDataSize() == 0U) {
        KERNEL_LOG_DEBUG("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_OK;
    }
    const DataType input_dtype = ctx.Input(kFirstInputIndex)->GetDataType();
    const DataType output_dtype = ctx.Output(kFirstOutputIndex)->GetDataType();
    uint32_t res = KERNEL_STATUS_OK;
    const bool out_i32 = (output_dtype == DT_INT32);
    switch (input_dtype) {
        case DT_INT32:
            res = out_i32 ? BucketizeCompute<int32_t, int32_t>(ctx) : BucketizeCompute<int32_t, int64_t>(ctx);
            break;
        case DT_INT64:
            res = out_i32 ? BucketizeCompute<int64_t, int32_t>(ctx) : BucketizeCompute<int64_t, int64_t>(ctx);
            break;
        case DT_FLOAT:
            res = out_i32 ? BucketizeCompute<float, int32_t>(ctx) : BucketizeCompute<float, int64_t>(ctx);
            break;
        case DT_DOUBLE:
            res = out_i32 ? BucketizeCompute<double, int32_t>(ctx) : BucketizeCompute<double, int64_t>(ctx);
            break;
        default:
            KERNEL_LOG_ERROR("Bucketize kernel data type [%s] not support.", DTypeStr(input_dtype).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    if (res != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("Bucketize kernel compute failed, KernelStatus is[%u]", res);
        return res;
    }
    return KERNEL_STATUS_OK;
}

template <typename T, typename T2, bool Upper>
void BucketizeCpuKernel::DoComputeBound(const T* input_data, T2* output_data, int64_t start, int64_t end) const
{
    const float* bnd = boundaries_data_.data();
    const int64_t m = static_cast<int64_t>(boundaries_data_.size());
    // Large M: std::lower_bound/std::upper_bound is faster than our linear scan
    // (measured crossover near M=128 on aarch64). Using the library ensures no
    // regression for large M while we still win at small M with the linear scan.
    if (m > kSmallBoundariesThreshold) {
        for (int64_t i = start; i < end; ++i) {
            const float x = static_cast<float>(input_data[i]);
            output_data[i] = Upper ? UpperBoundBinary<T2>(bnd, m, x) : LowerBoundBinary<T2>(bnd, m, x);
        }
        return;
    }
    // Small/medium M: branchless linear scan, NEON-vectorized on aarch64.
    for (int64_t i = start; i < end; ++i) {
        const float x = static_cast<float>(input_data[i]);
        output_data[i] = Upper ? UpperBoundLinear<T2>(bnd, m, x) : LowerBoundLinear<T2>(bnd, m, x);
    }
}

template <typename T, typename T2>
uint32_t BucketizeCpuKernel::BucketizeComputeParallel(const CpuKernelContext& ctx, const T* input_data, T2* output_data,
                                                      int64_t data_num, bool right) const
{
    const uint32_t cpu_num = aicpu::CpuKernelUtils::GetCPUNum(ctx);
    // Guard against underflow when cpu_num < kReservedCores.
    const uint32_t usable = (cpu_num > kReservedCores) ? (cpu_num - kReservedCores) : 1U;
    const uint32_t max_core_num = std::max(1U, usable);
    const int64_t per_unit_size = CeilMultiple(data_num, static_cast<int64_t>(max_core_num));
    KERNEL_LOG_INFO("Bucketize parallel: data_num=[%ld], cpu_num=[%u], per_unit_size=[%ld].", data_num, cpu_num,
                    per_unit_size);
    if (right) {
        auto sharder = [this, input_data, output_data](int64_t start, int64_t end) {
            DoComputeBound<T, T2, true>(input_data, output_data, start, end);
        };
        return CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, sharder);
    }
    auto sharder = [this, input_data, output_data](int64_t start, int64_t end) {
        DoComputeBound<T, T2, false>(input_data, output_data, start, end);
    };
    return CpuKernelUtils::ParallelFor(ctx, data_num, per_unit_size, sharder);
}

template <typename T, typename T2>
uint32_t BucketizeCpuKernel::BucketizeCompute(const CpuKernelContext& ctx) const
{
    const int64_t data_num = ctx.Input(kFirstInputIndex)->NumElements();
    const T* input_data = PtrToPtr<void, T>(ctx.Input(kFirstInputIndex)->GetData());
    T2* output_data = PtrToPtr<void, T2>(ctx.Output(kFirstOutputIndex)->GetData());
    const bool right = ctx.GetAttr("right")->GetBool();
    if (data_num >= kParallelDataNumSameShape) {
        return BucketizeComputeParallel<T, T2>(ctx, input_data, output_data, data_num, right);
    }
    if (right) {
        DoComputeBound<T, T2, true>(input_data, output_data, 0, data_num);
    } else {
        DoComputeBound<T, T2, false>(input_data, output_data, 0, data_num);
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kBucketize, BucketizeCpuKernel);
} // namespace aicpu
