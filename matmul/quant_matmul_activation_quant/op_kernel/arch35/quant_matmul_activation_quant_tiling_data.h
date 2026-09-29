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
 * \file quant_matmul_activation_quant_tiling_data.h
 * \brief Shared host/device tiling fields for activation and MX quantization.
 */
#pragma once
#include "kernel_tiling/kernel_tiling.h"
#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// QuantMatmulActivationQuant tiling_data
namespace QMMAQ {
enum class QuantAlg : uint8_t {
    OCP = 0,
    BLAS = 1,
    DYN_DTYPE_RANGE = 2,
};

enum class ActivationAlg : uint8_t {
    TANH = 0,
    ERF = 1,
    SWIGLU = 2,
};

enum class MX_QUANT_ROUND_MODE : uint8_t {
    RINT = 0,
    FLOOR = 1,
    ROUND = 2,
};

#pragma pack(push, 8)
struct alignas(8) QMMAQTilingData {
    uint32_t batchA1 = 1;
    uint32_t batchA2 = 1;
    uint32_t batchA3 = 1;
    uint32_t batchA4 = 1;
    uint32_t batchB1 = 1;
    uint32_t batchB2 = 1;
    uint32_t batchB3 = 1;
    uint32_t batchB4 = 1;
    uint32_t batchC1 = 1;
    uint32_t batchC2 = 1;
    uint32_t batchC3 = 1;
    uint32_t batchC4 = 1;
    uint32_t batchCount = 1;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t kL1 = 0;
    uint32_t scaleKL1 = 0;
    float dstTypeMax = 0.0;
    uint16_t baseM = 0;
    uint16_t baseN = 0;
    uint16_t baseK = 0;
    uint16_t mTailTile = 0;
    uint16_t nTailTile = 0;
    uint16_t mBaseTailSplitCnt = 1;
    uint16_t nBaseTailSplitCnt = 1;
    uint16_t mTailMain = 0;
    uint16_t nTailMain = 0;
    uint8_t nBufferNum = 0;
    uint8_t isBias = 0;
    uint8_t dbL0C = 0;
    uint8_t weightMustHitL2 = 1;
    uint8_t biasThreeDim = 0;
    ActivationAlg activationType = ActivationAlg::TANH;
    QuantAlg scaleAlg = QuantAlg::OCP;
    MX_QUANT_ROUND_MODE roundMode = MX_QUANT_ROUND_MODE::RINT;
};

struct alignas(8) QMMAQWithoutBatchTilingData {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t kL1 = 0;
    uint32_t scaleKL1 = 0;
    float dstTypeMax = 0.0;
    uint16_t baseM = 0;
    uint16_t baseN = 0;
    uint16_t baseK = 0;
    uint16_t mTailTile = 0;
    uint16_t nTailTile = 0;
    uint16_t mBaseTailSplitCnt = 1;
    uint16_t nBaseTailSplitCnt = 1;
    uint16_t mTailMain = 0;
    uint16_t nTailMain = 0;
    uint8_t nBufferNum = 0;
    uint8_t isBias = 0;
    uint8_t dbL0C = 0;
    uint8_t weightMustHitL2 = 1;
    ActivationAlg activationType = ActivationAlg::TANH;
    QuantAlg scaleAlg = QuantAlg::OCP;
    MX_QUANT_ROUND_MODE roundMode = MX_QUANT_ROUND_MODE::RINT;
};
#pragma pack(pop)

} // namespace QMMAQ
