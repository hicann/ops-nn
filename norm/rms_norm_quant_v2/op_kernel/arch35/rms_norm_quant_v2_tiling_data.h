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
 * \file rms_norm_quant_v2_tiling_data.h
 * \brief
 */
#ifndef _RMS_NORM_QUANT_V2_TILING_DATA_H_
#define _RMS_NORM_QUANT_V2_TILING_DATA_H_

struct RmsNormQuantV2RegbaseFullLoadTilingData {
    int64_t a;           // a
    int64_t r;           // r
    int64_t q;           // q, scales value (1 or R)
    int64_t blockFactor; // rows per core
    int64_t blockTail;   // rows for last core
    int64_t ubFactor;    // rows per UB iteration
    int64_t binaryAdd;   // binary add fold point on R axis
    uint64_t optionMask; // flags: scales2 | zero_points1 | zero_points2 | beta
    int64_t divMode;     // quant mode
    int64_t dstDtype;    // output dtype
    float epsilon;
    float avgFactor;     // 1.0 / R
    uint32_t rstdFlag;   // 0=no rstd output, 1=output rstd
};

struct RmsNormQuantV2RegbaseRecomputeTilingData {
    int64_t numM;
    int64_t numN;
    int64_t baseM;
    int64_t baseN;
    int64_t mPerCore;       // rows per core
    int64_t mLastCore;      // rows for last core
    int64_t nUbLoops;       // UB loops along R axis
    int64_t binAddQuotient; // intra-tile binary fold point
    int64_t powerSplit;     // inter-tile binary fold power
    int64_t mainFoldCount;  // main fold tile count after powerSplit
    int64_t foldTail;       // tail length after folding
    uint64_t optionMask;    // flags: needBrc | scales2 | zero_points1 | zero_points2 | beta
    uint64_t divMode;       // quant mode
    uint64_t dstDtype;      // output dtype
    float epsilon;
    float avgFactor;        // 1.0 / R
    uint32_t rstdFlag;      // 0=no rstd output, 1=output rstd
};

#endif
