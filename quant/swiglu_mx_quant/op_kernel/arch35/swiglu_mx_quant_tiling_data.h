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
 * \file swiglu_mx_quant_tiling_data.h
 * \brief Tiling data structure for SwiGLU + MX quantization
 */

#ifndef SWIGLU_MX_QUANT_TILING_DATA_H
#define SWIGLU_MX_QUANT_TILING_DATA_H

struct SwigluMxQuantTilingData {
    // Basic parameters
    int64_t usedCoreNum;

    // 3D data shape: [inputDim0, inputDim1, inputDim2]
    //   inputDim0 = batch dim (=1 for 2D, product of leading dims for >2D)
    //   inputDim1 = axis=-2 quant dim (M)
    //   inputDim2 = SwiGLU output dim = axis=-1 / 2 (N)
    int64_t inputDim0;
    int64_t inputDim1;
    int64_t inputDim2;

    // 3D block distribution (axis=-2 path)
    int64_t dimNBlockNum;       // CeilDiv(N, 256)
    int64_t dimNTail;           // N % 256 (0→256 tail)
    int64_t dimMBlockNum;       // CeilDiv(M, 64)
    int64_t dimMTail;           // M % 64 (0→64 tail)
    int64_t blockCountPerBatch; // dimMBlockNum * dimNBlockNum
    // 3D grid core distribution (act=-2, axis=-1 path)
    int64_t nCoreNum;  // cores in N direction
    int64_t bCoreNum;  // cores in B (batch) direction
    int64_t mCorePerB; // M-cores per B-core

    // Memory allocation parameters (axis=-1 path)
    // axis = -1, basicDim1 = 1, basicDim2 = 256
    // axis = -2, basicDim1 = 64, basicDim2 = 256
    int64_t maxBasicNumUbDim2;
    int64_t maxBasicNumUbDim1;

    // Inter-core split parameters (axis=-1 path)
    int64_t frontCoreNum;
    int64_t tailCoreBasicNumDim1;
    int64_t groupIndexNum;

    // attr
    int64_t activateLeft;
    int64_t swigluMode;
    int64_t scaleAlg;
    int64_t groupMode;
    float clampLimit;
    float gluAlpha;
    float gluBias;
};
#endif // SWIGLU_MX_QUANT_TILING_DATA_H