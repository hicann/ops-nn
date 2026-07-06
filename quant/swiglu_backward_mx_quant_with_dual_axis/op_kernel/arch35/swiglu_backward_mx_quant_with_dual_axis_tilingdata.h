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
 * \file swiglu_backward_mx_quant_with_dual_axis_tilingdata.h
 * \brief Kernel-side TilingData plain struct for SwigluBackwardMxQuantWithDualAxis
 */

#ifndef OPS_NN_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H
#define OPS_NN_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H

struct SwigluBackwardMxQuantWithDualAxisTilingData {
    int64_t usedCoreNum;
    int64_t dstType;
    int64_t activateLeft;
    int64_t dimM;
    int64_t dimN;
    int64_t numGroups;
    int64_t blockW;
    int64_t splitBlockH;
    int64_t dimNBlockNum;
    int64_t dimNTail;
    int64_t yGradRowStride;
    int64_t dimGradX;
    int64_t dimBatch;
};

#endif // OPS_NN_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_TILINGDATA_H
