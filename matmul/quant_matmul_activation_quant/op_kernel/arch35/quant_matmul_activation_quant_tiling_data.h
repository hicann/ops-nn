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
 * \brief
 */
#pragma once
#include "kernel_tiling/kernel_tiling.h"
#if defined(__CCE_AICORE__)
#include "../../quant_batch_matmul_v3/arch35/quant_batch_matmul_v3_tiling_data.h"
#else
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#endif
#ifndef __CCE_AICORE__
#include <cstdint>
#endif

// QuantMatmulActivationQuant tiling_data
namespace QMMAQ {
enum class BasicQuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
    PERGROUP_MODE = 0x1U << 5,
};

enum class QuantAlg : uint8_t {
    OCP = 0,
    BLAS = 1,
    DYN_DTYPE_RANGE = 2,
};

enum class GeluAlg : uint8_t {
    TANH = 0,
    ERF = 1,
};

enum class MX_QUANT_ROUND_MODE : uint8_t {
    RINT = 0,
    FLOOR = 1,
    ROUND = 2,
};

#pragma pack(push, 8)
struct QuantMatmulActivationQuantTilingData {
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData mmTilingData;
    GeluAlg activationType = GeluAlg::TANH;
    QuantAlg scaleAlg = QuantAlg::OCP;
    MX_QUANT_ROUND_MODE roundMode = MX_QUANT_ROUND_MODE::RINT;
    float dstTypeMax = 0.0;
};
#pragma pack(pop)
} // namespace QMMAQ
