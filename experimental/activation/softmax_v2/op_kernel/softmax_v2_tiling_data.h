/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):

 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
/*!
 * \file softmax_v2_tiling_data.h
 * \brief tiling data struct
*/
#ifndef _SOFTMAXV2_TILING_DATA_H_
#define _SOFTMAXV2_TILING_DATA_H_

struct SoftmaxV2TilingData {
    uint32_t totalLengthx;
    uint32_t totalLengthz;
    uint32_t small_tile_times;
    uint32_t big_tile_times;
    uint32_t small_tail_num;
    uint32_t big_tail_num;
    uint32_t big_core_num;
    uint32_t small_core_num;
    uint32_t small_tile_length;
    uint32_t big_tile_length;
    uint32_t tileNum;
    uint32_t core_tile_x1;
    int32_t dim;
    int32_t dimNum;
    int32_t dimarr[4];
    uint32_t sumtimes;
    uint32_t partnum;
    uint32_t elenum;
    uint32_t sumspace;
    uint32_t sumspacerows;
    uint32_t rows;
};
#endif