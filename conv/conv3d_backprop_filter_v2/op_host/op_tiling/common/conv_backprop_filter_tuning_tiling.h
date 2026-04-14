/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV_BACKPROP_INPUT_TUNING_TILING_H_
#define CONV_BACKPROP_INPUT_TUNING_TILING_H_

#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"

namespace tuningtiling {
#pragma pack(push, 1)
struct Conv3DBackpropFilterArgs {
    int32_t batch;
    int32_t groups;
    int32_t co;             // output channels
    int32_t ci;             // input channels
    int32_t dout;           // output depth
    int32_t wo;             // output width
    int32_t ho;             // output height
    int32_t wi;             // input width
    int32_t hi;             // input height
    int32_t di;             // input depth
    int32_t kw;             // kernel width
    int32_t kh;             // kernel height
    int32_t kd;             // kernel depth
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
    uint32_t hf32Flag;
    ge::DataType a_dtype = ge::DT_FLOAT16;
    ge::DataType b_dtype = ge::DT_FLOAT16;
    ge::DataType c_dtype = ge::DT_FLOAT16;
    int32_t a_dtype_bytes = 2;
    int32_t b_dtype_bytes = 2;
    int32_t c_dtype_bytes = 2;
    ge::Format fmapFormat;
    ge::Format dedyFormat;
    ge::Format filterFormat;
    };
#pragma pack(pop)

BEGIN_TUNING_TILING_DEF(Conv3DBackpropFilterTunerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, cin1G);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, cout1G);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, realGroup);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, channelSize);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, al0Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, bl0Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, cl0Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, al1Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, bl1Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseM);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseK);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, baseN);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepM);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepN);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepKa);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, stepKb);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, iterateOrder);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, bl1Bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, al1Bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreDk);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreHo);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, splitWo);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreCin);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatchDout);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, streamkType);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreM);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreN);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, singleCoreK);
TUNING_TILING_DATA_FIELD_DEF(bool, isSplitKernelHW);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, coreBindDirection);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Conv3DBackpropFilterTunerTiling,
  FIELD(Conv3DBackpropFilterTunerTiling, cin1G),
  FIELD(Conv3DBackpropFilterTunerTiling, cout1G),
  FIELD(Conv3DBackpropFilterTunerTiling, realGroup),
  FIELD(Conv3DBackpropFilterTunerTiling, channelSize),
  FIELD(Conv3DBackpropFilterTunerTiling, al0Pbuffer),
  FIELD(Conv3DBackpropFilterTunerTiling, bl0Pbuffer),
  FIELD(Conv3DBackpropFilterTunerTiling, cl0Pbuffer),
  FIELD(Conv3DBackpropFilterTunerTiling, al1Pbuffer),
  FIELD(Conv3DBackpropFilterTunerTiling, bl1Pbuffer),
  FIELD(Conv3DBackpropFilterTunerTiling, baseM),
  FIELD(Conv3DBackpropFilterTunerTiling, baseK),
  FIELD(Conv3DBackpropFilterTunerTiling, baseN),
  FIELD(Conv3DBackpropFilterTunerTiling, m0),
  FIELD(Conv3DBackpropFilterTunerTiling, k0),
  FIELD(Conv3DBackpropFilterTunerTiling, n0),
  FIELD(Conv3DBackpropFilterTunerTiling, stepM),
  FIELD(Conv3DBackpropFilterTunerTiling, stepN),
  FIELD(Conv3DBackpropFilterTunerTiling, stepKa),
  FIELD(Conv3DBackpropFilterTunerTiling, stepKb),
  FIELD(Conv3DBackpropFilterTunerTiling, iterateOrder),
  FIELD(Conv3DBackpropFilterTunerTiling, bl1Bound),
  FIELD(Conv3DBackpropFilterTunerTiling, al1Bound),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreDk),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreGroup),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreCout),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreHo),
  FIELD(Conv3DBackpropFilterTunerTiling, splitWo),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreBatch),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreCin),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreBatchDout),
  FIELD(Conv3DBackpropFilterTunerTiling, streamkType),
  FIELD(Conv3DBackpropFilterTunerTiling, usedCoreNum),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreM),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreN),
  FIELD(Conv3DBackpropFilterTunerTiling, singleCoreK),
  FIELD(Conv3DBackpropFilterTunerTiling, isSplitKernelHW),
  FIELD(Conv3DBackpropFilterTunerTiling, coreBindDirection));
}  // namespace tuningtiling
#endif
