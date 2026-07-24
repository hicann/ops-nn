/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sigmoid_grad_aicpu.h"

#include <complex>
#include "cpu_types.h"
#include "log.h"
#include "utils/eigen_tensor.h"

namespace {
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr int64_t kParallelDataNums = 280 * 1024;
const char* const kSigmoidGrad = "SigmoidGrad";
} // namespace

namespace aicpu {
uint32_t SigmoidGradCpuKernel::Compute(CpuKernelContext& ctx)
{
    if ((ctx.Output(0) != nullptr) && (ctx.Output(0)->NumElements() == 0)) {
        KERNEL_LOG_DEBUG("shape element number is zero.");
        return KERNEL_STATUS_OK;
    }

    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "[%s] check input and output failed.", kSigmoidGrad);
    KERNEL_HANDLE_ERROR(SigmoidGradCheck(ctx), "[%s] check params failed.", kSigmoidGrad);

    AttrValue* complex_conj_attr = ctx.GetAttr("complex_conj");
    KERNEL_CHECK_NULLPTR(complex_conj_attr, KERNEL_STATUS_PARAM_INVALID,
                         "Get ATTR complex_conj error, complex_conj is nullptr.")
    bool complex_conj = complex_conj_attr->GetBool();

    auto data_type = ctx.Input(0)->GetDataType();
    switch (data_type) {
        case DT_COMPLEX64:
            if (complex_conj) {
                return SigmoidGradComputeConj<std::complex<float>>(ctx);
            }
            return SigmoidGradCompute<std::complex<float>>(ctx);
        case DT_COMPLEX128:
            if (complex_conj) {
                return SigmoidGradComputeConj<std::complex<double>>(ctx);
            }
            return SigmoidGradCompute<std::complex<double>>(ctx);
        case DT_FLOAT:
            return SigmoidGradCompute<float>(ctx);
        case DT_FLOAT16:
            return SigmoidGradCompute<Eigen::half>(ctx);
        case DT_DOUBLE:
            return SigmoidGradCompute<double>(ctx);
        default:
            KERNEL_LOG_ERROR("SigmoidGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

uint32_t SigmoidGradCpuKernel::SigmoidGradCheck(const CpuKernelContext& ctx)
{
    auto input_0 = ctx.Input(0);
    auto input_1 = ctx.Input(1);
    auto output = ctx.Output(0);
    DataType input0_type = input_0->GetDataType();
    DataType input1_type = input_1->GetDataType();
    KERNEL_CHECK_FALSE((input0_type == input1_type), KERNEL_STATUS_PARAM_INVALID,
                       "The data type of input0 [%s] need be same with input1 [%s].", DTypeStr(input0_type).c_str(),
                       DTypeStr(input1_type).c_str())

    auto input0_size = input_0->GetDataSize();
    auto input1_size = input_1->GetDataSize();
    auto output_size = output->GetDataSize();
    KERNEL_CHECK_FALSE((input0_size == output_size), KERNEL_STATUS_PARAM_INVALID,
                       "The data size of output [%lu] need be same with input0 [%lu].", output_size, input0_size)

    KERNEL_CHECK_FALSE((input0_size == input1_size), KERNEL_STATUS_PARAM_INVALID,
                       "The data size of input1 [%lu] need be same with input0 [%lu].", input1_size, input0_size)

    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SigmoidGradCpuKernel::SigmoidGradCompute(const CpuKernelContext& ctx)
{
    auto input_y = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto input_dy = reinterpret_cast<T*>(ctx.Input(1)->GetData());
    auto output_z = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    size_t data_num = ctx.Input(0)->NumElements();

    auto shard_sigmoid_grad = [&input_y, &input_dy, &output_z](size_t start, size_t end) {
        T one_trans = static_cast<T>(1.0);
        for (size_t i = start; i < end; i++) {
            auto y_idx = input_y + i;
            auto dy_idx = input_dy + i;
            *(output_z + i) = (*dy_idx) * (one_trans - (*y_idx)) * (*y_idx);
        }
    };

    if (data_num > static_cast<size_t>(kParallelDataNums)) {
        uint32_t min_core_num = 1;
        uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, data_num, CeilMultiple(data_num, max_core_num), shard_sigmoid_grad),
            "SigmoidGrad Compute failed.")
    } else {
        shard_sigmoid_grad(0, data_num);
    }

    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t SigmoidGradCpuKernel::SigmoidGradComputeConj(const CpuKernelContext& ctx)
{
    auto input_y = reinterpret_cast<T*>(ctx.Input(0)->GetData());
    auto input_dy = reinterpret_cast<T*>(ctx.Input(1)->GetData());
    auto output_z = reinterpret_cast<T*>(ctx.Output(0)->GetData());
    size_t data_num = ctx.Input(0)->NumElements();

    auto shard_sigmoid_grad = [&input_y, &input_dy, &output_z](size_t start, size_t end) {
        T one_trans = static_cast<T>(1.0);
        for (size_t i = start; i < end; i++) {
            auto y_idx = input_y + i;
            auto dy_idx = input_dy + i;
            *(output_z + i) = (std::conj((one_trans - (*y_idx)) * (*y_idx))) * (*dy_idx);
        }
    };

    if (data_num > static_cast<size_t>(kParallelDataNums)) {
        uint32_t min_core_num = 1;
        uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
        KERNEL_HANDLE_ERROR(
            CpuKernelUtils::ParallelFor(ctx, data_num, CeilMultiple(data_num, max_core_num), shard_sigmoid_grad),
            "SigmoidGrad Compute complex conj failed.")
    } else {
        shard_sigmoid_grad(0, data_num);
    }

    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kSigmoidGrad, SigmoidGradCpuKernel);
} // namespace aicpu
