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
 * \file broadcast_gradient_args_base.h
 * \brief
 */

#ifndef BROADCAST_GRADIENT_ARGS_BASE_H
#define BROADCAST_GRADIENT_ARGS_BASE_H

#include "../inc/platform.h"

namespace BroadcastGradientArgs {
using namespace AscendC;

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B64 = { AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };

constexpr static AscendC::MicroAPI::CastTrait castTraitB642B32 = { AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };

constexpr int64_t FIRST_UINT64_SHAPE_DIM_ONE = 0x80000001;
constexpr int64_t UINT64_SHAPE_DIM_ONE = 1;
constexpr int32_t SHAPE_SIZE_IDX = 0;
constexpr int32_t SHAPE_DIM0_IDX = 1;
constexpr int32_t SHAPE1_GM_IDX = 9;
constexpr int64_t FIRST_OUTPUT_DIMS_IDX = 0;
constexpr int64_t SECOND_OUTPUT_DIMS_IDX = 9;
constexpr int64_t FIRST_OUTPUT_DIM0_IDX = 1;
constexpr int64_t SECOND_OUTPUT_DIM0_IDX = 10;

template <typename T>
class BroadcastGradientArgsBase {
public:
    __aicore__ inline BroadcastGradientArgsBase() {}
protected:
    /* global memory address */
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> y1Gm;
    GlobalTensor<T> y2Gm;
    GlobalTensor<uint64_t> outShapeGm;
    /*system virable*/
    int16_t blockSize = GetDataBlockSizeInBytes();
    uint32_t vlLen = GetVecLen();
    
};
}
#endif  // BROADCAST_GRADIENT_ARGS_BASE_H