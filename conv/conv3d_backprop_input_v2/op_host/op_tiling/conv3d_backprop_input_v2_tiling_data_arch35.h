/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_input_v2_tiling_data_arch35.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BACKPROP_INPUT_V2_TILING_DATA_ARCH35_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BACKPROP_INPUT_V2_TILING_DATA_ARCH35_H

#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3DBackpropInputArch35TilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchDim);
TILING_DATA_FIELD_DEF(uint32_t, groupDim);
TILING_DATA_FIELD_DEF(uint32_t, mDim);
TILING_DATA_FIELD_DEF(uint32_t, kDim);
TILING_DATA_FIELD_DEF(uint32_t, nDim);
TILING_DATA_FIELD_DEF(uint32_t, dDim);
TILING_DATA_FIELD_DEF(uint64_t, coreNum);
TILING_DATA_FIELD_DEF(uint8_t, al0Pbuffer);
TILING_DATA_FIELD_DEF(uint8_t, bl0Pbuffer);
TILING_DATA_FIELD_DEF(uint8_t, cl0Pbuffer);
TILING_DATA_FIELD_DEF(uint8_t, al1Pbuffer);
TILING_DATA_FIELD_DEF(uint8_t, bl1Pbuffer);
TILING_DATA_FIELD_DEF(uint8_t, iterateOrder);
TILING_DATA_FIELD_DEF(uint8_t, c0);
TILING_DATA_FIELD_DEF(uint8_t, c0BitsA);
TILING_DATA_FIELD_DEF(uint8_t, c0BitsB);
TILING_DATA_FIELD_DEF(uint8_t, enlarge);
TILING_DATA_FIELD_DEF(uint8_t, hf32Flag);
TILING_DATA_FIELD_DEF(uint8_t, initOutputFlag);
TILING_DATA_FIELD_DEF(uint8_t, isBiasFullLoad);
TILING_DATA_FIELD_DEF(uint8_t, enableVecTrans);
TILING_DATA_FIELD_DEF(uint8_t, enableFullLoad);
TILING_DATA_FIELD_DEF(uint8_t, quantMode);
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, cin);
TILING_DATA_FIELD_DEF(uint32_t, cout);
TILING_DATA_FIELD_DEF(uint32_t, cinG);
TILING_DATA_FIELD_DEF(uint32_t, coutG);
TILING_DATA_FIELD_DEF(uint32_t, cout1);
TILING_DATA_FIELD_DEF(uint32_t, cin1);
TILING_DATA_FIELD_DEF(uint32_t, cout1G);
TILING_DATA_FIELD_DEF(uint32_t, cin1G);
TILING_DATA_FIELD_DEF(uint32_t, dout);
TILING_DATA_FIELD_DEF(uint32_t, ho);
TILING_DATA_FIELD_DEF(uint32_t, wo);
TILING_DATA_FIELD_DEF(uint32_t, di);
TILING_DATA_FIELD_DEF(uint32_t, hi);
TILING_DATA_FIELD_DEF(uint32_t, wi);
TILING_DATA_FIELD_DEF(uint32_t, dk);
TILING_DATA_FIELD_DEF(uint32_t, hk);
TILING_DATA_FIELD_DEF(uint32_t, wk);
TILING_DATA_FIELD_DEF(uint32_t, group);
TILING_DATA_FIELD_DEF(uint32_t, oriGroup);
TILING_DATA_FIELD_DEF(uint32_t, strideD);
TILING_DATA_FIELD_DEF(uint32_t, strideH);
TILING_DATA_FIELD_DEF(uint32_t, strideW);
TILING_DATA_FIELD_DEF(uint32_t, padFront);
TILING_DATA_FIELD_DEF(uint32_t, padBack);
TILING_DATA_FIELD_DEF(uint32_t, padUp);
TILING_DATA_FIELD_DEF(uint32_t, padDown);
TILING_DATA_FIELD_DEF(uint32_t, padLeft);
TILING_DATA_FIELD_DEF(uint32_t, padRight);
TILING_DATA_FIELD_DEF(int32_t, backpropPadTail);
TILING_DATA_FIELD_DEF(int32_t, backpropPadUp);
TILING_DATA_FIELD_DEF(int32_t, backpropPadDown);
TILING_DATA_FIELD_DEF(int32_t, backpropPadLeft);
TILING_DATA_FIELD_DEF(int32_t, backpropPadRight);
TILING_DATA_FIELD_DEF(uint32_t, dilationD);
TILING_DATA_FIELD_DEF(uint32_t, dilationH);
TILING_DATA_FIELD_DEF(uint32_t, dilationW);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreCin);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreDin);
TILING_DATA_FIELD_DEF(uint32_t, baseM);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, baseN);
TILING_DATA_FIELD_DEF(uint32_t, stepKa);
TILING_DATA_FIELD_DEF(uint32_t, stepKb);
TILING_DATA_FIELD_DEF(uint32_t, singleIterateDk);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreM);
TILING_DATA_FIELD_DEF(uint64_t, enRelu);
TILING_DATA_FIELD_DEF(uint64_t, kSegment);
TILING_DATA_FIELD_DEF(uint64_t, kSegmentTail);
TILING_DATA_FIELD_DEF(uint64_t, kValueSegment);
TILING_DATA_FIELD_DEF(bool, enableSplitK);
TILING_DATA_FIELD_DEF(bool, useUbAccumForSplitK);
TILING_DATA_FIELD_DEF(int8_t, offsetX);
TILING_DATA_FIELD_DEF(uint32_t, kSCoutFullLoad);
TILING_DATA_FIELD_DEF(uint32_t, kSUseWorkSpace);
TILING_DATA_FIELD_DEF(uint32_t, khDilation);
TILING_DATA_FIELD_DEF(uint32_t, kwDilation);
TILING_DATA_FIELD_DEF(uint32_t, hoExpand);
TILING_DATA_FIELD_DEF(uint32_t, woExpand);
TILING_DATA_FIELD_DEF(uint64_t, dkHkWk);
TILING_DATA_FIELD_DEF(uint64_t, hkWk);
TILING_DATA_FIELD_DEF(uint8_t, fixedShiftVal);
TILING_DATA_FIELD_DEF_ARR(uint8_t, 8, reserved);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3DBackpropInputV2, Conv3DBackpropInputArch35TilingData);
REGISTER_TILING_DATA_CLASS(Conv3DTransposeV2, Conv3DBackpropInputArch35TilingData);
REGISTER_TILING_DATA_CLASS(ExtendConvTranspose, Conv3DBackpropInputArch35TilingData);
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CONV3D_BACKPROP_INPUT_V2_TILING_DATA_ARCH35_H
