/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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
 * \file group_normalization_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef GROUP_NORMALIZATION_GRAD_TILING_DATA_H_
#define GROUP_NORMALIZATION_GRAD_TILING_DATA_H_

struct GroupNormalizationGradTilingData {
    uint64_t groupElemNum;
    uint64_t groupCount;
    uint64_t smallCoreGroupNum;
    uint64_t bigCoreGroupNum;
    uint64_t finalGroupTileNum;
    uint64_t tileDataNum;
    uint64_t alignedTileDataNum;
    uint64_t tailDataNum;
    uint64_t tailBlockNum;
    float groupElemNumFloat;
    float groupElemNumReciprocal;
};
#endif
