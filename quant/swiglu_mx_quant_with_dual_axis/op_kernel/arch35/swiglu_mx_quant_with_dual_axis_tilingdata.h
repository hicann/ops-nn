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
 * \file swiglu_mx_quant_with_dual_axis_tilingdata.h
 * \brief Kernel-side TilingData plain struct for SwigluMxQuantWithDualAxis
 */

#ifndef OPS_NN_SWIGLU_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H
#define OPS_NN_SWIGLU_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H


struct SwigluMxQuantWithDualAxisTilingData {
    int64_t usedCoreNum;            // 实际使用核数（kernel 侧视为 totalCoreNum）
    int64_t dstType;                // 输出量化类型
    int64_t activateLeft;           // 1=左激活, 0=右激活
    int64_t dimM;                   // 总行数
    int64_t dimN;                   // N（SwiGLU 输出列数 = x 列数 / 2）
    int64_t numGroups;              // group 数量
    int64_t blockW;                 // 列方向基本块宽度=256
    int64_t splitBlockH;            // 行方向基本块高度=64
    int64_t dimNBlockNum;           // N 方向切分基本块数 = ceil(N/blockW)
    int64_t dimNTail;               // N 方向尾块列数
};

#endif // OPS_NN_SWIGLU_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H
