/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "avg_pool1d_avg_matrix_aicpu.h"

#include <algorithm>
#include <vector>

#include "Eigen/Core"
#include "cpu_types.h"
#include "log.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/kernel_util.h"

namespace aicpu {
namespace {
const char* kAvgPool1DAvgMatrix = "AvgPool1DAvgMatrix";
constexpr int64_t kInputN = 1;
constexpr int64_t kInputC1 = 1;
constexpr int64_t kInputH = 1;
constexpr int64_t kInputC0 = 16;
constexpr int64_t kDimSize = 4;
constexpr int64_t kPadSize = 2;

struct AvgPool1DAvgMatrixParam {
    int64_t wInInput = 1;
    int64_t kSize = 0;
    int64_t strides = 0;
    int64_t padL = 0;
    int64_t padR = 0;
    int64_t wOutput = 0;
    int64_t outputSize = 0;
    bool countIncludePad = false;
};

#define AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DTYPE, TYPE, CTX)                    \
    case (DTYPE): {                                                          \
        uint32_t result = DoCompute<TYPE>(CTX);                              \
        if (result != KERNEL_STATUS_OK) {                                    \
            KERNEL_LOG_ERROR("AvgPool1DAvgMatrix kernel doCompute failed."); \
            return result;                                                   \
        }                                                                    \
        break;                                                               \
    }

static int64_t CalculateWoutput(const int64_t w_in_input, const int64_t pad_l, const int64_t pad_r,
                                const int64_t k_size, const int64_t strides, const bool ceil_mode)
{
    int64_t res = 0;
    if (ceil_mode) {
        res = (w_in_input + pad_l + pad_r - k_size + strides - 1) / strides + 1;
    } else {
        res = ((w_in_input + pad_l + pad_r) - k_size) / strides + 1;
    }
    if (pad_l > 0) {
        if (((res - 1) * strides) >= (w_in_input + pad_l)) {
            res--;
        }
    }
    return res;
}

static uint32_t GetInputWidth(const Format inputFormat, const std::vector<int64_t>& dims, int64_t& wInInput)
{
    if ((inputFormat == FORMAT_NCHW) || (inputFormat == FORMAT_NC1HWC0)) {
        wInInput = dims[3];
        return KERNEL_STATUS_OK;
    }
    if (inputFormat == FORMAT_NHWC) {
        wInInput = dims[2];
        return KERNEL_STATUS_OK;
    }
    KERNEL_LOG_ERROR("Format is not in [FORMAT_NHWC or FORMAT_NCHW or FORMAT_NC1HWC0], current input format is [%d].",
                     inputFormat);
    return KERNEL_STATUS_PARAM_INVALID;
}

static uint32_t InitAvgPool1DAvgMatrixParam(CpuKernelContext& ctx, AvgPool1DAvgMatrixParam& param)
{
    auto input_shape = ctx.Input(0)->GetTensorShape();
    auto input_format = input_shape->GetFormat();
    std::vector<int64_t> dims = input_shape->GetDimSizes();
    AttrValue* k_size_ptr = ctx.GetAttr("ksize");
    AttrValue* strides_ptr = ctx.GetAttr("strides");
    AttrValue* ceil_mode_ptr = ctx.GetAttr("ceil_mode");
    AttrValue* pads_ptr = ctx.GetAttr("pads");
    AttrValue* count_include_pad_ptr = ctx.GetAttr("count_include_pad");

    param.kSize = k_size_ptr->GetInt();
    param.strides = strides_ptr->GetInt();
    KERNEL_CHECK_FALSE((param.strides != 0), KERNEL_STATUS_PARAM_INVALID, "%s strides [%ld] must not be equal to zero.",
                       kAvgPool1DAvgMatrix, param.strides);
    uint32_t ret = GetInputWidth(input_format, dims, param.wInInput);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "Get input width failed.");

    std::vector<int64_t> pads = pads_ptr->GetListInt();
    param.padL = pads[0];
    param.padR = pads[1];
    param.countIncludePad = count_include_pad_ptr->GetBool();
    param.outputSize = ctx.Output(0)->NumElements();
    param.wOutput = CalculateWoutput(param.wInInput, param.padL, param.padR, param.kSize, param.strides,
                                     ceil_mode_ptr->GetBool());
    return KERNEL_STATUS_OK;
}

template <typename T>
static uint32_t FillAvgPool1DAvgMatrixWindow(T* outputData, const AvgPool1DAvgMatrixParam& param, const int64_t n,
                                             const int64_t c1, const int64_t h, const int64_t w)
{
    int64_t start = param.strides * w;
    int64_t end = param.strides * w + param.kSize;
    if (!param.countIncludePad) {
        start = std::max(start, param.padL);
        end = std::min(end, param.wInInput + param.padL);
    } else {
        end = std::min(end, param.wInInput + param.padL + param.padR);
    }
    int64_t data_num = end - start;
    KERNEL_CHECK_FALSE((data_num > 0), KERNEL_STATUS_PARAM_INVALID, "%s data_num [%ld] must be greater than zero.",
                       kAvgPool1DAvgMatrix, data_num);
    T tmp = static_cast<T>(1.0 / data_num);
    for (int64_t c0 = 0; c0 < kInputC0; c0++) {
        int64_t out_offset_point = n * kInputC1 * kInputH * param.wOutput * kInputC0 +
                                   c1 * kInputH * param.wOutput * kInputC0 + h * param.wOutput * kInputC0 +
                                   w * kInputC0 + c0;
        KERNEL_CHECK_FALSE((out_offset_point < param.outputSize), KERNEL_STATUS_PARAM_INVALID,
                           "%s out_offset_point [%ld] must < out_put_size [%ld].", kAvgPool1DAvgMatrix,
                           out_offset_point, param.outputSize);
        outputData[out_offset_point] = tmp;
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
static uint32_t FillAvgPool1DAvgMatrix(T* outputData, const AvgPool1DAvgMatrixParam& param)
{
    for (int64_t n = 0; n < kInputN; n++) {
        for (int64_t c1 = 0; c1 < kInputC1; c1++) {
            for (int64_t h = 0; h < kInputH; h++) {
                for (int64_t w = 0; w < param.wOutput; w++) {
                    uint32_t ret = FillAvgPool1DAvgMatrixWindow(outputData, param, n, c1, h, w);
                    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "Fill AvgPool1DAvgMatrix window failed.");
                }
            }
        }
    }
    return KERNEL_STATUS_OK;
}
} // namespace
uint32_t AvgPool1DAvgMatrixCpuKernel::CheckParam(CpuKernelContext& ctx)
{
    auto output_data_temp = ctx.Output(0)->GetData();
    Tensor* input_tensor = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(output_data_temp, KERNEL_STATUS_PARAM_INVALID, "[%s] get output data failed.",
                         kAvgPool1DAvgMatrix);
    auto input_shape = input_tensor->GetTensorShape();
    std::vector<int64_t> dims = input_shape->GetDimSizes();
    KERNEL_CHECK_FALSE((dims.size() >= kDimSize), KERNEL_STATUS_PARAM_INVALID, "%s dims size [%zu] must >= 4.",
                       kAvgPool1DAvgMatrix, dims.size());
    AttrValue* k_size_ptr = ctx.GetAttr("ksize");
    KERNEL_CHECK_NULLPTR(k_size_ptr, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr ksize fail.", kAvgPool1DAvgMatrix);
    AttrValue* strides_ptr = ctx.GetAttr("strides");
    KERNEL_CHECK_NULLPTR(strides_ptr, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr strides fail.", kAvgPool1DAvgMatrix);
    AttrValue* ceil_mode_ptr = ctx.GetAttr("ceil_mode");
    KERNEL_CHECK_NULLPTR(ceil_mode_ptr, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr ceil_mode fail.",
                         kAvgPool1DAvgMatrix);
    AttrValue* pads_ptr = ctx.GetAttr("pads");
    KERNEL_CHECK_NULLPTR(pads_ptr, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr pads fail.", kAvgPool1DAvgMatrix);
    AttrValue* count_include_pad_ptr = ctx.GetAttr("count_include_pad");
    KERNEL_CHECK_NULLPTR(count_include_pad_ptr, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr count_include_pad fail.",
                         kAvgPool1DAvgMatrix);
    std::vector<int64_t> pads = pads_ptr->GetListInt();
    KERNEL_CHECK_FALSE((pads.size() >= kPadSize), KERNEL_STATUS_PARAM_INVALID,
                       "%s pads [%ld] must have at least two elements.", kAvgPool1DAvgMatrix,
                       static_cast<int64_t>(pads.size()));
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t AvgPool1DAvgMatrixCpuKernel::DoCompute(CpuKernelContext& ctx)
{
    auto output_data_temp = ctx.Output(0)->GetData();
    auto output_data = reinterpret_cast<T*>(output_data_temp);
    AvgPool1DAvgMatrixParam param;
    uint32_t ret = InitAvgPool1DAvgMatrixParam(ctx, param);
    KERNEL_CHECK_FALSE((ret == KERNEL_STATUS_OK), ret, "Init AvgPool1DAvgMatrix param failed.");
    return FillAvgPool1DAvgMatrix(output_data, param);
}

uint32_t AvgPool1DAvgMatrixCpuKernel::Compute(CpuKernelContext& ctx)
{
    Tensor* input_tensor = ctx.Input(0);
    KERNEL_CHECK_NULLPTR(input_tensor, KERNEL_STATUS_PARAM_INVALID, "[%s] get input_tensor fail.", kAvgPool1DAvgMatrix);
    Tensor* output_tensor = ctx.Output(0);
    KERNEL_CHECK_NULLPTR(output_tensor, KERNEL_STATUS_PARAM_INVALID, "[%s] get output_tensor fail.",
                         kAvgPool1DAvgMatrix);
    KERNEL_CHECK_FALSE((CheckParam(ctx) == KERNEL_STATUS_OK), KERNEL_STATUS_PARAM_INVALID, "CheckParam failed.");
    DataType dt = static_cast<DataType>(input_tensor->GetDataType());
    switch (dt) {
        AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_INT8, int8_t, ctx)
        AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_INT16, int16_t, ctx)
        AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_INT32, int32_t, ctx)
        AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_INT64, int64_t, ctx)
        AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_UINT8, uint8_t, ctx)
        AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_FLOAT16, Eigen::half, ctx)
        AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_FLOAT, float, ctx)
        AVGPOOL1DAVGMATRIX_COMPUTE_CASE(DT_DOUBLE, double, ctx)
        default:
            KERNEL_LOG_WARN("AvgPool1DAvgMatrix kernels does not support this data type [%d].", dt);
            return KERNEL_STATUS_PARAM_INVALID;
    }
    return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kAvgPool1DAvgMatrix, AvgPool1DAvgMatrixCpuKernel);
} // namespace aicpu
