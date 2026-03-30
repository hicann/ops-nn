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
 * \file rms_norm_dynamic_mx_quant_tiling_def.h
 * \brief Tiling data definition for rms_norm_dynamic_mx_quant kernel UT
 */

#ifndef RMS_NORM_DYNAMIC_MX_QUANT_TILING_DEF_H_
#define RMS_NORM_DYNAMIC_MX_QUANT_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#pragma pack(1)
struct RmsNormDynamicMxQuantFullLoadTilingData {
    int64_t usedCoreNum;
    int64_t mTailCores;
    int64_t numM;
    int64_t numN;
    int64_t numNUbAligned;
    int64_t binAddFoldPoint;
    int64_t mPerCore;
    int64_t mUbFactor;
    int64_t mxBlockSize;
    int64_t nMxblockAligned;
    int64_t nMxblockNumAlignedTwo;
    int64_t nMxblockNum;
    int64_t needPadN;
    int64_t needPadScale;
    int64_t scaleAlg;
    int64_t roundMode;
    int64_t hasInputBeta;
    int64_t hasOutputRstd;
    float epsilon;
    float avgFactor;
};
#pragma pack()

template <typename T>
inline void InitTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
};

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data)

#define DTYPE_X half
#define DTYPE_GAMMA half
#define DTYPE_Y int8_t

#endif
