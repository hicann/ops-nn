/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file conv_backprop_filter_context_utils.h
 * \brief
 */

#ifndef CONV_BACKPROP_FILTER_CONTEXT_UTILS_H_
#define CONV_BACKPROP_FILTER_CONTEXT_UTILS_H_

#include <exe_graph/runtime/tiling_context.h>
#include "tbe_tiling_api.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace optiling;

struct Conv3dBpFilterV2RunInfo {
    int32_t batch;
    int32_t co;             // output channels
    int32_t ci;             // input channels
    int32_t cout1_g;        // output channels per group
    int32_t cin1_g;         // input channels per group
    int32_t dout;           // output depth
    int32_t wo;             // output width
    int32_t ho;             // output height
    int32_t wi;             // input width
    int32_t hi;             // input height
    int32_t di;             // input depth
    int32_t kw;             // kernel width
    int32_t kh;             // kernel height
    int32_t kd;             // kernel depth
    int32_t real_g;         // actual groups
    int32_t stride_w;
    int32_t stride_h;
    int32_t stride_d;
    int32_t pad_l;          // left padding
    int32_t pad_r;          // right padding
    int32_t pad_u;          // up padding
    int32_t pad_d;          // down padding
    int32_t pad_f;          // front padding
    int32_t pad_b;          // back padding
    int32_t dilation_w;
    int32_t dilation_h;
    int32_t dilation_d;
    int32_t ci1;            // another input channels parameter

    uint64_t bl1_bound;     // buffer limit 1 bound, tiling结果的衍生参数，不建议放在这里
    int32_t batch_dout_single_core;  // batch*dout per core, tiling结果的衍生参数，不建议放在这里
    int32_t groups;
    int32_t mag_factor;

    // Tiling parameters
    uint32_t k0;
    uint32_t m0;
    uint32_t n0;
    uint32_t hf32Flag;

    ge::DataType a_dtype = ge::DT_FLOAT16;
    ge::DataType b_dtype = ge::DT_FLOAT16;
    ge::DataType c_dtype = ge::DT_FLOAT16;
    int32_t a_dtype_bytes = 2;
    int32_t b_dtype_bytes = 2;
    int32_t c_dtype_bytes = 2;
    uint32_t core_num;
};

bool SetConv3dBpFilterV2RunInfo(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2);
}
}
}
#endif // CONV_BACKPROP_FILTER_CONTEXT_UTILS_H_