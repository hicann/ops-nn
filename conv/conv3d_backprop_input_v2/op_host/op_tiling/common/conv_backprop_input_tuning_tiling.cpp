/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv_backprop_input_tuning_tiling.h"

namespace tuningtiling {
DECLARE_STRUCT_RELATE_WITH_OP_V2(Conv3DBackpropInputV2, Conv3DBackpropInputArgs, batch_n, groups, 
    dedx_d, dedx_cin, dedx_h, dedx_w, dedy_d, dedy_cout, dedy_h, dedy_w, kernel_d, kernel_h, kernel_w, 
    stride_d, stride_h, stride_w, pad_h, pad_t, pad_u, pad_d, pad_l, pad_r, 
    dilation_d, dilation_h, dilation_w, 
    backprop_pad_h, backprop_pad_t, backprop_pad_u, backprop_pad_d, backprop_pad_l, backprop_pad_r,
    hf32_flag, a_dtype_bytes, b_dtype_bytes, c_dtype_bytes, 
    outBackpropFormat, filterFormat, yFormat);

REGISTER_TUNING_TILING_CLASS(Conv3DBackpropInputV2, Conv3DBackpropInputTunerTiling);
}  // namespace tuningtiling
