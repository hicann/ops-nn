/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_clamp_tiling_data.h
 * \brief SwigluClamp tiling data struct (row semantics: totalLength=M rows, N=gate/up width)
 */
#ifndef SWIGLU_CLAMP_TILING_DATA_H_
#define SWIGLU_CLAMP_TILING_DATA_H_

struct SwigluClampTilingData {
    int64_t totalLength;  // = M (rows of x[...,2N] / out[...,N])
    int64_t N;            // gate/up width (x last dim / 2)
    int64_t formerNum;    // number of former cores
    int64_t formerLength; // former core rows
    int64_t tailNum;      // number of tail cores (=1)
    int64_t tailLength;   // tail core rows
    int64_t tileLength;   // UB tile rows (tileM)
    float limit;          // clamp limit (Step-3.7 = 7.0)
};

#endif // SWIGLU_CLAMP_TILING_DATA_H_
