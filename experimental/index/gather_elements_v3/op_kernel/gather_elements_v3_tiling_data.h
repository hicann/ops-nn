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
 * \file gather_elements_v3_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __GATHER_ELEMENTS_V3_TILLING_DATA_H__
#define __GATHER_ELEMENTS_V3_TILLING_DATA_H__

struct GatherElementsV3TilingData {
    uint32_t xPreDim;
    uint32_t xGatherDim;
    uint32_t xPostDim;
    uint32_t idxPreDim;
    uint32_t idxGatherDim;
    uint32_t idxPostDim;
    uint32_t usedCores;
    uint32_t tileSize;
};
#endif
