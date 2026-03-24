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
 * \file add_rms_norm_dynamic_mx_quant_tiling_data.h
 * \brief
 */
#ifndef _ADD_RMS_NORM_DYNAMIC_MX_QUANT_TILING_DATA_H_
#define _ADD_RMS_NORM_DYNAMIC_MX_QUANT_TILING_DATA_H_

struct AddRmsNormDynamicMxQuantTilingData {
    uint64_t numRow;            // A: total rows (batch)
    uint64_t numCol;            // R: norm dimension size
    uint64_t numColAlign;       // R aligned to block boundary
    uint64_t blockFactor;       // rows per core
    uint64_t rowFactor;         // rows per UB iteration
    uint64_t binAddQuotient;    // binary add quotient point
    float epsilon;             
    float avgFactor;            // 1.0 / R
    uint64_t roundMode;          // rounding mode (0=round, 1=floor, 4=rint)
    uint64_t mxBlockSize;        // MX block size (32)
    uint64_t scaleAlg;           // scale algorithm (0=standard, 1=cublas)
    uint64_t blockNumInColAxis;  // CeilDiv(R, 32)
    uint64_t dstStrideUbBlocks;  // whether R needs dstStrideUbBlocksding to mxBlockSize multiple
    uint64_t mxScaleSize;        // mxscale output size per row
    uint32_t betaFlag;          // whether beta input exists
    uint32_t rstdFlag;          // whether rstd output is needed
};

#endif
