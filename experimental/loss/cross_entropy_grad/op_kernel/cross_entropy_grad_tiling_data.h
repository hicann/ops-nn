/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
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
 * \file cross_entropy_grad_tiling_data.h
 * \brief CrossEntropyGrad tiling data struct
 */

#ifndef CROSS_ENTROPY_GRAD_TILING_DATA_H_
#define CROSS_ENTROPY_GRAD_TILING_DATA_H_

struct CrossEntropyGradTilingData {
    uint32_t rowLen = 0;
    uint32_t totalRows = 0;
    uint32_t smallCoreRowNum = 0;
    uint32_t bigCoreRowNum = 0;
    uint32_t bigCoreNum = 0;
    uint32_t tileRowNum = 0;
    uint32_t smallCoreTileNum = 0;
    uint32_t bigCoreTileNum = 0;
    uint32_t smallTailRowNum = 0;
    uint32_t bigTailRowNum = 0;
    uint32_t colSplitMode = 0;
    uint32_t tileCols = 0;
    uint32_t numColPasses = 0;
    uint32_t dataTypeId = 0;
    uint32_t typeBytes = 0;
};

#endif // CROSS_ENTROPY_GRAD_TILING_DATA_H_
