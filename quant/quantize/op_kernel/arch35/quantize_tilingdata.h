/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file quantize_tilingdata.h
 * \brief
 */
#ifndef OPS_ACTIVATION_QUANTIZE_OP_KERNEL_V35_QUANTIZE_TILINGDATA_H_
#define OPS_ACTIVATION_QUANTIZE_OP_KERNEL_V35_QUANTIZE_TILINGDATA_H_
#include <cstdint>

class QuantizeTilingData {
public:
    int64_t numCore;
    int64_t blockAxis;
    int64_t ubAxis;
    int64_t dim0;
    int64_t dim1;
    int64_t dim2;
    int64_t blockUnion;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t baseN;
    int64_t baseLen;
    int64_t hasZeroPoint;
    int64_t axis;
    int64_t roundMode;
    int64_t sqrtMode;
};

#endif