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
 * \file layer_norm_v4_common.h
 * \brief
 */

#ifndef LAYER_NORM_V4_COMMON_H
#define LAYER_NORM_V4_COMMON_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace LayerNormV4 {
using namespace AscendC;

constexpr static int64_t BLOCK_SIZE = 32;
constexpr static uint32_t FLOAT_BYTES = 4;
constexpr static int64_t MAX_STRIDE = 65535;
constexpr static int64_t DOUBLE_BUFFER = 2;
constexpr static int64_t LOOP_HIGH_SHIFT = 21;
constexpr static int64_t LOOP_STRIDE_HIGH_SHIFT = 40;
constexpr static uint32_t NUM_TWO = 2;
constexpr static float POS_INF = 3.40282366920938E+38;
constexpr static float zero = 0.0f;
constexpr static uint32_t VL_B32 = GetVecLen() / sizeof(float);
constexpr static uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);

constexpr static AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitB322B16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

template <typename M>
__aicore__ inline void CastBatchMeanRstdToDtype(
    __local_mem__ float* batchMeanInAddr, __local_mem__ float* batchRstdInAddr,
    __local_mem__ M* batchMeanOutAddr, __local_mem__ M* batchRstdOutAddr, uint64_t currentANum)
{
    constexpr uint32_t VL_F32 = AscendC::VECTOR_REG_WIDTH / sizeof(float);
    constexpr uint32_t VL_MEAN = AscendC::VECTOR_REG_WIDTH / sizeof(M);
    uint32_t castCount = static_cast<uint32_t>(currentANum);
    uint16_t castLoops = static_cast<uint32_t>((castCount + VL_F32 - 1) / VL_F32);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> input_mean;
        AscendC::MicroAPI::RegTensor<float> input_rstd;
        AscendC::MicroAPI::RegTensor<M> output_mean;
        AscendC::MicroAPI::RegTensor<M> output_rstd;
        AscendC::MicroAPI::MaskReg pregLoop;
        for (uint16_t i = 0; i < castLoops; i++) {
            pregLoop = AscendC::MicroAPI::UpdateMask<float>(castCount);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                input_mean, batchMeanInAddr + VL_F32 * i);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(
                input_rstd, batchRstdInAddr + VL_F32 * i);
            Cast<M, float, castTraitB322B16>(output_mean, input_mean, pregLoop);
            Cast<M, float, castTraitB322B16>(output_rstd, input_rstd, pregLoop);
            DataCopy<M, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                batchMeanOutAddr + i * VL_MEAN, output_mean, pregLoop);
            DataCopy<M, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(
                batchRstdOutAddr + i * VL_MEAN, output_rstd, pregLoop);
        }
    }
}

} // namespace LayerNormV4

#endif  // LAYER_NORM_V4_COMMON_H
