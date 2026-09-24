/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN
 * Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not
 * use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY,
 * OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */
/*!
 * \file binary_cross_entropy_grad_v2_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __BINARY_CROSS_ENTROPY_GRAD_V2_TILLING_DATA_H__
#define __BINARY_CROSS_ENTROPY_GRAD_V2_TILLING_DATA_H__

struct BinaryCrossEntropyGradV2TilingData {
    int64_t totalLength;
    int64_t smallCoreDataNum;
    int64_t bigCoreDataNum;
    int64_t finalBigTileNum;
    int64_t finalSmallTileNum;
    int64_t tileDataNum;
    int64_t smallTailDataNum;
    int64_t bigTailDataNum;
    int64_t tailBlockNum;
    int64_t reduction;
};
#endif
