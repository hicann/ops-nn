/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hinge_loss_grad_tiling_data.h
 * \brief tiling data struct for multi-core hinge loss gradient
 */

#ifndef _HINGE_LOSS_GRAD_TILING_DATA_H_
#define _HINGE_LOSS_GRAD_TILING_DATA_H_

struct HingeLossGradTilingData {
    int64_t smallCoreDataNum = 0;
    int64_t bigCoreDataNum = 0;
    int64_t finalBigTileNum = 0;
    int64_t finalSmallTileNum = 0;
    int64_t tileDataNum = 0;
    int64_t smallTailDataNum = 0;
    int64_t bigTailDataNum = 0;
    int64_t tailBlockNum = 0;
};
#endif
