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
struct Conv3DBackpropInputArgs {
    int32_t batch_n;
    int32_t groups;

    int32_t dedx_d;
    int32_t dedx_cin;
    int32_t dedx_h;
    int32_t dedx_w;

    int32_t dedy_d;
    int32_t dedy_cout;
    int32_t dedy_h;
    int32_t dedy_w;

    int32_t pad_h;
    int32_t pad_t;
    int32_t pad_l;
    int32_t pad_r;
    int32_t pad_u;
    int32_t pad_d;

    int32_t kernel_d;
    int32_t kernel_h;
    int32_t kernel_w;

    int32_t stride_d;
    int32_t stride_h;
    int32_t stride_w;

    int32_t dilation_d;
    int32_t dilation_h;
    int32_t dilation_w;

    int32_t backprop_pad_h;
    int32_t backprop_pad_t;
    int32_t backprop_pad_l;
    int32_t backprop_pad_r;
    int32_t backprop_pad_u;
    int32_t backprop_pad_d;

    int32_t hf32_flag;
    int32_t a_dtype_bytes;
    int32_t b_dtype_bytes;
    int32_t c_dtype_bytes;

    ge::Format filterFormat;
    ge::Format outBackpropFormat;
    ge::Format yFormat;
};
#pragma pack(pop)

BEGIN_TUNING_TILING_DEF(Conv3DBackpropInputTunerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint64_t, al0Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, bl0Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cl0Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, al1Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, bl1Pbuffer);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, iterateOrder);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, c0);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, c0BitsA);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, c0BitsB);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, enlarge);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, initOutputFlag);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, isBiasFullLoad);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, enableVecTrans);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, enableFullLoad);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, quantMode);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cinG);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, coutG);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cout1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cin1);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cout1G);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, cin1G);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, backpropPadTail);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, backpropPadUp);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, backpropPadDown);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, backpropPadLeft);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, backpropPadRight);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreGroup);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreCout);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreCin);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreDin);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, baseM);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, baseK);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, baseN);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, stepKa);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, stepKb);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleIterateDk);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, singleCoreM);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, enRelu);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, coreNum);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kSCoutFullLoad);
TUNING_TILING_DATA_FIELD_DEF(uint64_t, kSUseWorkSpace);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, loadB2Condition);
TUNING_TILING_DATA_FIELD_DEF(uint8_t, loadB1Condition);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Conv3DBackpropInputTunerTiling,
  FIELD(Conv3DBackpropInputTunerTiling, al0Pbuffer),
  FIELD(Conv3DBackpropInputTunerTiling, bl0Pbuffer), 
  FIELD(Conv3DBackpropInputTunerTiling, cl0Pbuffer),
  FIELD(Conv3DBackpropInputTunerTiling, al1Pbuffer),
  FIELD(Conv3DBackpropInputTunerTiling, bl1Pbuffer),
  FIELD(Conv3DBackpropInputTunerTiling, iterateOrder),
  FIELD(Conv3DBackpropInputTunerTiling, c0),
  FIELD(Conv3DBackpropInputTunerTiling, c0BitsA),
  FIELD(Conv3DBackpropInputTunerTiling, c0BitsB),
  FIELD(Conv3DBackpropInputTunerTiling, enlarge),
  FIELD(Conv3DBackpropInputTunerTiling, initOutputFlag),
  FIELD(Conv3DBackpropInputTunerTiling, isBiasFullLoad),
  FIELD(Conv3DBackpropInputTunerTiling, enableVecTrans),
  FIELD(Conv3DBackpropInputTunerTiling, enableFullLoad),
  FIELD(Conv3DBackpropInputTunerTiling, quantMode),
  FIELD(Conv3DBackpropInputTunerTiling, cinG),
  FIELD(Conv3DBackpropInputTunerTiling, coutG),
  FIELD(Conv3DBackpropInputTunerTiling, cout1),
  FIELD(Conv3DBackpropInputTunerTiling, cin1),
  FIELD(Conv3DBackpropInputTunerTiling, cout1G),
  FIELD(Conv3DBackpropInputTunerTiling, cin1G),
  FIELD(Conv3DBackpropInputTunerTiling, backpropPadTail),
  FIELD(Conv3DBackpropInputTunerTiling, backpropPadUp),
  FIELD(Conv3DBackpropInputTunerTiling, backpropPadDown),
  FIELD(Conv3DBackpropInputTunerTiling, backpropPadLeft),
  FIELD(Conv3DBackpropInputTunerTiling, backpropPadRight),
  FIELD(Conv3DBackpropInputTunerTiling, singleCoreGroup),
  FIELD(Conv3DBackpropInputTunerTiling, singleCoreCout),
  FIELD(Conv3DBackpropInputTunerTiling, singleCoreCin),
  FIELD(Conv3DBackpropInputTunerTiling, singleCoreDin),
  FIELD(Conv3DBackpropInputTunerTiling, baseM),
  FIELD(Conv3DBackpropInputTunerTiling, baseK),
  FIELD(Conv3DBackpropInputTunerTiling, baseN),
  FIELD(Conv3DBackpropInputTunerTiling, stepKa),
  FIELD(Conv3DBackpropInputTunerTiling, stepKb),
  FIELD(Conv3DBackpropInputTunerTiling, singleIterateDk),
  FIELD(Conv3DBackpropInputTunerTiling, singleCoreBatch),
  FIELD(Conv3DBackpropInputTunerTiling, singleCoreM),
  FIELD(Conv3DBackpropInputTunerTiling, enRelu),
  FIELD(Conv3DBackpropInputTunerTiling, coreNum),
  FIELD(Conv3DBackpropInputTunerTiling, kSCoutFullLoad),
  FIELD(Conv3DBackpropInputTunerTiling, kSUseWorkSpace),
  FIELD(Conv3DBackpropInputTunerTiling, loadB2Condition),
  FIELD(Conv3DBackpropInputTunerTiling, loadB1Condition));
}  // namespace tuningtiling
#endif
