/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file glu_tiling.hpp
 * \brief
 */
#ifndef OPP_GLU_TILING_HPP
#define OPP_GLU_TILING_HPP

template<typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}
template<typename T>
__aicore__ inline T ISMAX(T num, T rnd)
{
    return ((num) > (rnd)) ? (num) : (rnd);
}

const uint32_t MAX_CORE_NUMBER = 64;
struct GluTilingData {
    uint32_t usedCoreNum = 0; // number of vector core. Don't move, must be in the first

    uint32_t isDoubleBuffer = 1; // is double buffer on?
    uint64_t maxTileLen = 0; // number of calculations in one tile On each core. Unit: number of elements

    // vector dimension on each core
    // Apply for memory based on the maximum number of vector cores, but use the memory based on usedCoreNum
    uint64_t singleCoreBlockLens[MAX_CORE_NUMBER] = {0}; // BlockLen on each core may be different. Unit: number of elements
};

// tiling for single input tensor, the memory of the two input tenors
struct GluSingleTilingData {
    uint32_t is32BAligned = 1; // Is 32-byte aligned for split colLen?
    uint32_t isDoubleBuffer = 1; // is double buffer on?

    // eg: for shape:[8192,1,2*3904], totalRowLen is 8192
    uint64_t rowLen = 2; // row length for split vector, Unit:element
    // eg: for shape:[8192,1,2*3904], totalColLen is 3904
    uint64_t colLen = 32; // column length for split vector, Unit:element
    uint32_t baseRowLen = 2; // for one tile in one core, Unit:element
    uint32_t baseColLen = 16; // for one tile in one core, Unit:element
};
#endif  // OPP_GLU_TILING_HPP
