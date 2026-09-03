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
 * \file fused_adam_tiling_data.h
 * \brief tiling data struct for fused_adam
 */

#ifndef _FUSED_ADAM_TILING_DATA_H_
#define _FUSED_ADAM_TILING_DATA_H_

#include <cstdint>

constexpr uint16_t MAX_TENSOR_CONT = 512;
constexpr uint16_t MAX_CORE_CONT = 50;
constexpr uint16_t MAX_TENSOR_CONT_950 = 512;
constexpr uint16_t MAX_CORE_CONT_950 = 80;

struct FusedAdamTilingData {
    float lr;
    float beta1;
    float beta2;
    float weightDecay;
    float eps;
    uint32_t amsgrad;
    uint32_t maximize;
    uint32_t useGradScale;
    uint32_t useFoundInf;
    uint64_t tensorNum;
    uint64_t usedCoreNum;
    uint32_t tensorStartList_[MAX_CORE_CONT_950] = {0};
    uint32_t tensorEndList_[MAX_CORE_CONT_950] = {0};
    uint64_t tensorDataCountList_[MAX_TENSOR_CONT_950] = {0};
    uint64_t tensorStartOffsetList_[MAX_CORE_CONT_950] = {0};
    uint64_t tensorEndOffsetList_[MAX_CORE_CONT_950] = {0};
};
#endif // _FUSED_ADAM_TILING_DATA_H_
