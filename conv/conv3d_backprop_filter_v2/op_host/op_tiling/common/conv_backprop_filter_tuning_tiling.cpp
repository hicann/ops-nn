/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "conv_backprop_filter_tuning_tiling.h"

namespace tuningtiling {
DECLARE_STRUCT_RELATE_WITH_OP_V2(Conv3DBackpropFilterV2, Conv3DBackpropFilterArgs, 
    batch, groups, co, ci, dout, wo, ho, wi, hi, di, kw, kh, kd, 
    stride_w, stride_h, stride_d, pad_l, pad_r, pad_u, pad_d, pad_f, pad_b, 
    dilation_w, dilation_h, dilation_d, hf32Flag, a_dtype, b_dtype, c_dtype, 
    a_dtype_bytes, b_dtype_bytes, c_dtype_bytes, fmapFormat, dedyFormat, filterFormat);

REGISTER_TUNING_TILING_CLASS(Conv3DBackpropFilterV2, Conv3DBackpropFilterTunerTiling);
}  // namespace tuningtiling
