/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adaptive_max_pool2d_aicpu.h"
#include <cmath>
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel_utils.h"

namespace {
const char* const kAdaptiveMaxPool2d = "AdaptiveMaxPool2d";
template <typename SCALAR_T, typename INDICES_T>
struct AdaptiveCalcArgs {
    SCALAR_T* input_data = nullptr;
    SCALAR_T* output_data = nullptr;
    INDICES_T* indices_data = nullptr;

    int64_t in_size_b = 0;
    int64_t in_size_d = 0;
    int64_t in_size_h = 0;
    int64_t in_size_w = 0;
    int64_t out_size_h = 0;
    int64_t out_size_w = 0;

    int64_t in_stride_b = 0;
    int64_t in_stride_d = 0;
    int64_t in_stride_h = 0;
    int64_t in_stride_w = 0;
};

inline int64_t StartIndex(int64_t offset, int64_t out_size, int64_t in_size)
{
    return (int64_t)std::floor((double)(offset * in_size) / out_size);
}

inline int64_t EndIndex(int64_t offset, int64_t out_size, int64_t in_size)
{
    return (int64_t)std::ceil((double)((offset + 1) * in_size) / out_size);
}
} // namespace

namespace aicpu {
inline bool CheckMulIsOverflow(int64_t in_size_h, int64_t out_size_h, int64_t in_size_w, int64_t out_size_w)
{
    int64_t tmp = 0;
    return (!MulWithoutOverflow(in_size_h, out_size_h, tmp)) || (!MulWithoutOverflow(in_size_w, out_size_w, tmp));
}
template <typename SCALAR_T, typename INDICES_T>
void AdaptiveMaxPool2dSingleOutFrame(const CpuKernelContext& ctx, AdaptiveCalcArgs<SCALAR_T, INDICES_T> args)
{
    (void)CpuKernelUtils::ParallelFor(ctx, args.in_size_d, 1, [&](int64_t start, int64_t end) {
        for (auto d = start; d < end; d++) {
            for (int64_t offset_h = 0; offset_h < args.out_size_h; offset_h++) {
                int64_t in_start_h = StartIndex(offset_h, args.out_size_h, args.in_size_h);
                int64_t in_end_h = EndIndex(offset_h, args.out_size_h, args.in_size_h);
                int64_t step_h = in_end_h - in_start_h;

                for (int64_t offset_w = 0; offset_w < args.out_size_w; offset_w++) {
                    int64_t in_start_w = StartIndex(offset_w, args.out_size_w, args.in_size_w);
                    int64_t in_end_w = EndIndex(offset_w, args.out_size_w, args.in_size_w);
                    int64_t step_w = in_end_w - in_start_w;

                    SCALAR_T* in_point = args.input_data + d * args.in_stride_d + in_start_h * args.in_stride_h +
                                         in_start_w * args.in_stride_w;
                    SCALAR_T* out_point = args.output_data + d * args.out_size_h * args.out_size_w +
                                          offset_h * args.out_size_w + offset_w;
                    INDICES_T* indices_point = args.indices_data + d * args.out_size_h * args.out_size_w +
                                               offset_h * args.out_size_w + offset_w;

                    int64_t ih = 0;
                    int64_t iw = 0;
                    INDICES_T max_index = static_cast<INDICES_T>((ih + in_start_h) * args.in_size_w +
                                                                 (iw + in_start_w));
                    SCALAR_T max_val = -std::numeric_limits<SCALAR_T>::infinity();
                    for (ih = 0; ih < step_h; ih++) {
                        for (iw = 0; iw < step_w; iw++) {
                            SCALAR_T val = *(in_point + ih * args.in_stride_h + iw * args.in_stride_w);
                            if ((val > max_val) || std::isnan(static_cast<double>(val))) {
                                max_val = val;
                                max_index = static_cast<INDICES_T>((ih + in_start_h) * args.in_size_w +
                                                                   (iw + in_start_w));
                            }
                        }
                    }

                    *out_point = max_val;
                    *indices_point = max_index;
                }
            }
        }
    });
}

template <typename SCALAR_T, typename INDICES_T>
void AdaptiveMaxPool2dOutFrame(const CpuKernelContext& ctx, AdaptiveCalcArgs<SCALAR_T, INDICES_T> args)
{
    CpuKernelUtils::ParallelFor(ctx, args.in_size_b, 1, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
            AdaptiveCalcArgs<SCALAR_T, INDICES_T> sub_args = args;
            sub_args.input_data = args.input_data + b * args.in_stride_b;
            sub_args.output_data = args.output_data + b * args.in_size_d * args.out_size_h * args.out_size_w;
            sub_args.indices_data = args.indices_data + b * args.in_size_d * args.out_size_h * args.out_size_w;

            AdaptiveMaxPool2dSingleOutFrame<SCALAR_T, INDICES_T>(ctx, sub_args);
        }
    });
}

template <typename SCALAR_T, typename INDICES_T>
uint32_t AdaptiveMaxPool2dOutCpuTemplate(const CpuKernelContext& ctx)
{
    Tensor& input = *(ctx.Input(kFirstInputIndex));

    auto input_shape_ptr = input.GetTensorShape();
    KERNEL_CHECK_NULLPTR(input_shape_ptr, KERNEL_STATUS_PARAM_INVALID, "Get input x0 shape failed.");
    int32_t input_dims = input_shape_ptr->GetDims();

    for (int32_t i = 0; i < input_dims; i++) {
        KERNEL_CHECK_FALSE((input_shape_ptr->GetDimSize(i) > 0), KERNEL_STATUS_PARAM_INVALID,
                           "Adaptive_max_pool2d: expected input to have non-empty spatial dimensions, "
                           "but input has sizes [%d] with dimension [%d] being empty.",
                           input_dims, i);
    }

    KERNEL_CHECK_FALSE((input_dims == 3 || input_dims == 4), KERNEL_STATUS_PARAM_INVALID,
                       "Non-empty [3D] or [4D] (batch mode) tensor expected for input.");

    AdaptiveCalcArgs<SCALAR_T, INDICES_T> args;
    args.in_size_b = 1;
    args.in_size_d = 0;
    args.in_size_h = 0;
    args.in_size_w = 0;
    args.in_stride_d = 1;
    args.in_stride_h = 1;
    args.in_stride_w = 1;
    args.in_stride_b = 1;
    args.out_size_h = 0;
    args.out_size_w = 0;

    std::vector<int64_t> output_size = ctx.GetAttr("output_size")->GetListInt();
    if (output_size.size() == 2) {
        args.out_size_h = output_size[0];
        args.out_size_w = output_size[1];
    } else if (output_size.size() == 1) {
        args.out_size_h = output_size[0];
        args.out_size_w = output_size[0];
    } else {
        KERNEL_LOG_ERROR(
            "Adaptive_max_pool2d: internal error, output_size.size() must be [2] or [1], now size is [%zu].",
            output_size.size());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    uint64_t output_data_num = static_cast<uint64_t>(args.out_size_h * args.out_size_w);
    uint64_t output0_data_size = output_data_num * sizeof(SCALAR_T);
    if (output0_data_size > ctx.Output(kFirstOutputIndex)->GetDataSize()) {
        KERNEL_LOG_ERROR("Adaptive_max_pool2d: output 0 size must big then [%lu], now size is [%lu].",
                         output0_data_size, ctx.Output(kFirstOutputIndex)->GetDataSize());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    uint64_t output1_data_size = output_data_num * sizeof(INDICES_T);
    if (output1_data_size > ctx.Output(kSecondOutputIndex)->GetDataSize()) {
        KERNEL_LOG_ERROR("Adaptive_max_pool2d: output 1 size must big then [%lu], now size is [%lu].",
                         output1_data_size, ctx.Output(kSecondOutputIndex)->GetDataSize());
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (args.out_size_h <= 0 || args.out_size_w <= 0) {
        KERNEL_LOG_ERROR(
            "Adaptive_max_pool2d: internal error, output_size H or W must be positive, now H is [%ld], W is [%ld].",
            args.out_size_h, args.out_size_w);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    int dim_b = 0;
    int dim_d = 0;
    int dim_w = 0;
    int dim_h = 0;
    auto input_format = input_shape_ptr->GetFormat();
    if (input_format == FORMAT_NCHW) {
        if (input_dims == 4) {
            dim_b = kFormatNCHWIndexN;
            dim_d = kFormatNCHWIndexC;
            dim_h = kFormatNCHWIndexH;
            dim_w = kFormatNCHWIndexW;
        } else {
            dim_d = kFormatCHWIndexC;
            dim_h = kFormatCHWIndexH;
            dim_w = kFormatCHWIndexW;
        }
    } else if (input_format == FORMAT_NHWC) {
        if (input_dims == 4) {
            dim_b = kFormatNHWCIndexN;
            dim_h = kFormatNHWCIndexH;
            dim_w = kFormatNHWCIndexW;
            dim_d = kFormatNHWCIndexC;
        } else {
            dim_h = kFormatHWCIndexH;
            dim_w = kFormatHWCIndexW;
            dim_d = kFormatHWCIndexC;
        }
    } else {
        KERNEL_LOG_ERROR("Format is not in [FORMAT_NHWC or FORMAT_NCHW],"
                         "current input format is [%d].",
                         input_format);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (input_dims == 4) {
        args.in_size_b = input_shape_ptr->GetDimSize(dim_b);
    }
    args.in_size_d = input_shape_ptr->GetDimSize(dim_d);
    args.in_size_h = input_shape_ptr->GetDimSize(dim_h);
    args.in_size_w = input_shape_ptr->GetDimSize(dim_w);

    args.in_stride_b = args.in_size_d * args.in_size_h * args.in_size_w;
    args.in_stride_d = args.in_size_h * args.in_size_w;
    args.in_stride_h = args.in_size_w;
    args.in_stride_w = 1;

    args.input_data = static_cast<SCALAR_T*>(input.GetData());
    args.output_data = static_cast<SCALAR_T*>(ctx.Output(kFirstOutputIndex)->GetData());
    args.indices_data = static_cast<INDICES_T*>(ctx.Output(kSecondOutputIndex)->GetData());
    if (CheckMulIsOverflow(args.in_size_h, args.out_size_h, args.in_size_w, args.out_size_w)) {
        KERNEL_LOG_ERROR("Mul Overflow in_size_h[%ld], out_size_h[%ld], in_size_w[%ld], out_size_w[%ld]",
                         args.in_size_h, args.out_size_h, args.in_size_w, args.out_size_w);
        return KERNEL_STATUS_PARAM_INVALID;
    }

    if (input_dims == 3) {
        AdaptiveMaxPool2dSingleOutFrame<SCALAR_T>(ctx, args);
    } else {
        AdaptiveMaxPool2dOutFrame<SCALAR_T>(ctx, args);
    }
    return KERNEL_STATUS_OK;
}

template <typename SCALAR_T>
uint32_t DoCompute(CpuKernelContext& ctx, DataType indices_type)
{
    switch (indices_type) {
        case DT_INT32:
            return AdaptiveMaxPool2dOutCpuTemplate<SCALAR_T, int32_t>(ctx);
        case DT_INT64:
            return AdaptiveMaxPool2dOutCpuTemplate<SCALAR_T, int64_t>(ctx);
        default:
            KERNEL_LOG_ERROR("Output data_type [%s] must be in [{DT_INT32, DT_INT64}].",
                             DTypeStr(indices_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

uint32_t AdaptiveMaxPool2d::Compute(CpuKernelContext& ctx)
{
    Tensor* input_0 = ctx.Input(kFirstInputIndex);
    KERNEL_CHECK_NULLPTR(input_0, KERNEL_STATUS_PARAM_INVALID, "Get input tensor failed.");
    KERNEL_CHECK_NULLPTR(input_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input data failed.");
    Tensor* output_0 = ctx.Output(kFirstOutputIndex);
    KERNEL_CHECK_NULLPTR(output_0, KERNEL_STATUS_PARAM_INVALID, "Get first output tensor failed.");
    KERNEL_CHECK_NULLPTR(output_0->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get first output data failed.");
    Tensor* output_1 = ctx.Output(kSecondOutputIndex);
    KERNEL_CHECK_NULLPTR(output_1, KERNEL_STATUS_PARAM_INVALID, "Get second output tensor failed.");
    KERNEL_CHECK_NULLPTR(output_1->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get second output data failed.");

    AttrValue* attr_output_size = ctx.GetAttr("output_size");
    KERNEL_CHECK_NULLPTR(attr_output_size, KERNEL_STATUS_PARAM_INVALID, "[%s] get attr:output_size failed.",
                         kAdaptiveMaxPool2d);
    std::vector<int64_t> v_output_size = attr_output_size->GetListInt();

    KERNEL_LOG_INFO("AdaptiveMaxPool2d kernel,input[0]:size is [%lu];output_0:size is [%lu]; output_1:size is [%lu].",
                    input_0->GetDataSize(), output_0->GetDataSize(), output_1->GetDataSize());

    KERNEL_LOG_INFO("[%s] get attr:output_size [%s].", kAdaptiveMaxPool2d, VectorToString(v_output_size).c_str());

    auto data_type = static_cast<DataType>(input_0->GetDataType());
    auto indices_type = output_1->GetDataType();
    switch (data_type) {
        case DT_FLOAT:
            return DoCompute<float>(ctx, indices_type);
        case DT_DOUBLE:
            return DoCompute<double>(ctx, indices_type);
        case DT_FLOAT16:
            return DoCompute<Eigen::half>(ctx, indices_type);
        default:
            KERNEL_LOG_ERROR("AdaptiveMaxPool2d kernel data type [%s] not support.", DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

REGISTER_CPU_KERNEL(kAdaptiveMaxPool2d, AdaptiveMaxPool2d);
} // namespace aicpu
