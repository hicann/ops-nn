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
    int64_t usedCoreNum;          // Actual number of cores used

    // Data shape information
    int64_t inputDim1;            // Input x -2 axis dimension (after merge)      inputDimY
    int64_t inputDim2;            // Input x -1 axis dimension (after merge)      inputDimX
    int64_t outputDim2;           // Output y -1 axis dimension                   outputDimX

    // Memory allocation parameters
    int64_t basicDim2;               // Basic block -1 axis element count         basicX
    int64_t basicDim1;               // Basic block -2 axis element count         basicY
    int64_t maxBasicNumUbDim2;    // Number of basic blocks in -1 axis per UB iteration   maxBasicNumUbDimX
    int64_t maxBasicNumUbDim1;    // Number of basic blocks in -2 axis per UB iteration   maxBasicNumUbDimY
    int64_t ubLoopPerRow;         // UB loops per row for non-full-load scenario
    int64_t ubTailPerRow;         // Tail elements per row for non-full-load scenario

    // Inter-core split parameters
    int64_t frontCoreNum;               // Front cores: number of cores that process one extra block
    int64_t frontCoreBasicNumDim1;      // Front cores: basic blocks in -2 axis           frontCoreBasicNumDimY
    int64_t frontCoreLoopTimes;         // Front cores: loop times
    int64_t frontCoreLastLoopBasicNum;  // Front cores: basic blocks in last loop
    int64_t tailCoreBasicNumDim1;       // Tail core: basic blocks in -2 axis             tailCoreBasicNumDimY
    int64_t tailCoreLoopTimes;          // Tail core: loop times
    int64_t tailCoreLastLoopBasicNum;   // Tail core: basic blocks in last loop

    // SwiGLU related parameters (reserved, using float type as per README)
    int64_t activateLeft;         // SwiGLU left/right activation (reserved)
    int64_t swigluMode;           // SwiGLU or variant mode (reserved)

    // Quantization related parameters
    int64_t roundMode;            // Rounding mode
    int64_t scaleAlg;             // Scale algorithm (0=OCP, 1=cuBLAS, 2=RNE)
    int64_t groupMode;            // Group index mode (0=count, 1=cumsum)
    int64_t groupIndexNum;        // Group index number, 0 if group_index not exists, else shape[0] of group_index
    float clampLimit;              // Clamp limit for SwiGLU variant (reserved, float)
    float gluAlpha;                // Alpha for SwiGLU variant (reserved, float)
    float gluBias;                 // Bias for SwiGLU variant (reserved, float)
    float maxDtypeValue;           // Maximum dtype value (reserved, used when scale_alg=2 and dst_type=FP4_E1M2)
};
#endif // SWIGLU_MX_QUANT_TILING_DATA_H