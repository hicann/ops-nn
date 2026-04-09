/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "elu_aicpu.h"
#include <float.h>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"
#include "cpu_types.h"

namespace {
const char* const kElu = "Elu";
constexpr uint32_t kOutputNum = 1U;
constexpr uint32_t kInputNum = 1U;
constexpr int64_t kParallelDefaultDataNum = 60 * 1024;
constexpr int64_t kParallelBfloat16DataNum = 10 * 1024;
}

namespace aicpu {
template <typename T>
T TransType(float &val)
{
    return T(val);
}

template <>
double TransType(float &val)
{
    return atof(std::to_string(val).c_str());
}

template <typename T>
void EluDoCompute(const T *input, T *output, EluAttrInfo &eluInfo, int64_t start, int64_t end)
{
    const T zeroVal = static_cast<T>(0);
    const T oneVal = static_cast<T>(1);

    // tf api, these attr defalut val 1
    if (std::fabs(eluInfo.alpha - 1.0F) < FLT_EPSILON &&
        std::fabs(eluInfo.scale - 1.0F) < FLT_EPSILON &&
        std::fabs(eluInfo.inputScale - 1.0F) < FLT_EPSILON) {
        for (int64_t i = start; i < end; i++) {
            if (input[i] >= zeroVal) {
                output[i] = input[i];
                continue;
            }
            output[i] = Eigen::numext::exp(input[i]) - oneVal;
        }
        return;
    }

    const T alphaVal = TransType<T>(eluInfo.alpha);
    const T scaleVal = TransType<T>(eluInfo.scale);
    const T inputScaleVal = TransType<T>(eluInfo.inputScale);
    if (std::fabs(eluInfo.scale - 1.0F) < FLT_EPSILON &&
        std::fabs(eluInfo.inputScale - 1.0F) < FLT_EPSILON) { // pytorch api, scale and input scale attr default var 1
        for (int64_t i = start; i < end; i++) {
            if (input[i] >= zeroVal) {
                output[i] = input[i];
                continue;
            }
            output[i] = alphaVal * (Eigen::numext::exp(input[i]) - oneVal);
        }
        return;
    }

    for (int64_t i = start; i < end; i++) {
        if (input[i] >= zeroVal) {
            output[i] = input[i] * scaleVal;
            continue;
        }
        output[i] = alphaVal * scaleVal * (Eigen::numext::exp(input[i] * inputScaleVal) - oneVal);
    }
}

template <typename T>
uint32_t EluCompute(CpuKernelContext &ctx)
{
    auto dataNum = ctx.Output(0)->NumElements();
    if (dataNum == 0) {
        KERNEL_LOG_DEBUG("element number is zero.");
        return KERNEL_STATUS_OK;
    }

    auto dataType = ctx.Input(0)->GetDataType();
    AttrValue *attrAlpha = ctx.GetAttr("alpha");
    const float alpha = attrAlpha != nullptr ? attrAlpha->GetFloat() : 1.0F;
    AttrValue *attrScale = ctx.GetAttr("scale");
    const float scale = attrScale != nullptr ? attrScale->GetFloat() : 1.0F;
    AttrValue *attrInputScale = ctx.GetAttr("input_scale");
    const float inputScale = attrInputScale != nullptr ? attrInputScale->GetFloat() : 1.0F;
    EluAttrInfo eluAttInfo = {alpha, scale, inputScale};

    auto in = reinterpret_cast<T *>(ctx.Input(0)->GetData());
    auto out = reinterpret_cast<T *>(ctx.Output(0)->GetData());
    if ((dataNum <= kParallelDefaultDataNum && dataType != DT_BFLOAT16) ||
        (dataNum <= kParallelBfloat16DataNum && dataType == DT_BFLOAT16)) {
        EluDoCompute(in, out, eluAttInfo, 0, dataNum);
        return KERNEL_STATUS_OK;
    }

    const uint32_t minCoreNum = 1;
    const int64_t maxCoreNum = std::max(minCoreNum, aicpu::CpuKernelUtils::GetCPUNum(ctx) - kResvCpuNum);
    auto sharderElu = [&](int64_t start, int64_t end) {
        EluDoCompute<T>(in, out, eluAttInfo, start, end);
    };
    KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, dataNum, dataNum / maxCoreNum, sharderElu),
                        "Elu Compute failed.")
    return KERNEL_STATUS_OK;
}

static std::unordered_map<uint32_t, std::function<uint32_t(CpuKernelContext &)>>
    kCalls = {
        {DT_FLOAT16, EluCompute<Eigen::half>},
        {DT_FLOAT, EluCompute<float_t>},
        {DT_DOUBLE, EluCompute<double_t>},
        {DT_BFLOAT16, EluCompute<Eigen::bfloat16>}
};

uint32_t EluCpuKernel::Compute(CpuKernelContext &ctx) {
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                        "Check Elu op params failed.");
    Tensor *inputX = ctx.Input(0);
    auto inputDtype = inputX->GetDataType();
    const auto &computeFunc = kCalls.find(inputDtype);
    if (computeFunc != kCalls.end()) {
        return computeFunc->second(ctx);
    }
    KERNEL_LOG_ERROR("Elu kernel data type [%u] not support", inputDtype);
    return KERNEL_STATUS_PARAM_INVALID;
}

REGISTER_CPU_KERNEL(kElu, EluCpuKernel);
}  // namespace aicpu