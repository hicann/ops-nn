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
 * \file ada_layer_norm_grad_common_constants.h
 * \brief
 */

#ifndef ADA_LAYER_NORM_GRAD_COMMON_CONSTANTS_H
#define ADA_LAYER_NORM_GRAD_COMMON_CONSTANTS_H

#include <cstdint>

namespace AdaLayerNormGrad {

constexpr int32_t COMMON_B32_BLOCK_SIZE = 8;
constexpr int32_t COMMON_B16_BLOCK_SIZE = 16;
constexpr int32_t COMMON_B32_REPEAT_SIZE = 64;
constexpr int32_t COMMON_CONSTANT_TWO = 2;
constexpr int32_t COMMON_CONSTANT_EIGHT = 8;
constexpr int32_t COMMON_CONSTANT_SIXTEEN = 16;
constexpr int32_t COMMON_MAX_REPEAT = 255;
constexpr int32_t COMMON_VC_MAX_REPEAT = 248;

} // namespace AdaLayerNormGrad

#endif