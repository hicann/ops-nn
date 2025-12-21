/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_matmul_common.h
 * \brief
 */
#ifndef __OP_HOST_FUSED_MATMUL_COMMON_H__
#define __OP_HOST_FUSED_MATMUL_COMMON_H__
#include <map>
#include "matmul/fused_mat_mul/op_kernel/arch35/fused_mat_mul_tilingkey.h"

namespace optiling {

constexpr size_t INPUT_X1_IDX = 0UL;
constexpr size_t INPUT_X2_IDX = 1UL;
constexpr size_t INPUT_BIAS_IDX = 2UL;
constexpr size_t INPUT_X3_IDX = 3UL;
constexpr size_t OUTPUT_Y_IDX = 4UL;

constexpr size_t ATTR_TRANS_X1_IDX = 0UL;
constexpr size_t ATTR_TRANS_X2_IDX = 1UL;
constexpr size_t ATTR_ENABLE_HF32_IDX = 2UL;
constexpr size_t ATTR_OP_TYPE_IDX = 3UL;

constexpr size_t TRANS_MODE_BIT_WIDTH = 4UL;

enum class FusedMatmulTrans : std::uint8_t
{
    NO_TRANS = F_NO_TRANS,
    A_TRANS = F_A_TRANS,
    B_TRANS = F_B_TRANS,
    AB_TRANS = F_AB_TRANS
};

enum class FusedOpType : std::uint8_t
{
    NONE = F_OPTYPE_NONE,
    ADD = F_OPTYPE_ADD,
    MUL = F_OPTYPE_MUL,
    GELU_ERF = F_OPTYPE_GELU_ERF,
    GELU_TANH = F_OPTYPE_GELU_TANH,
    RELU = F_OPTYPE_RELU,
};

const std::map<std::string, FusedOpType> FUSED_OP_TYPE_MAP = {
    {"", FusedOpType::NONE},
    {"add", FusedOpType::ADD},
    {"mul", FusedOpType::MUL},
    {"gelu_erf", FusedOpType::GELU_ERF},
    {"gelu_tanh", FusedOpType::GELU_TANH},
    {"relu", FusedOpType::RELU}};

} // namespace optiling
#endif // __OP_HOST_FUSED_MATMUL_COMMON_H__