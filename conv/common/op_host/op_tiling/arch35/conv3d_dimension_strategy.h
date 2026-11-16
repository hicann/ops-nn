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
 * \file conv3d_dimension_strategy.h
 * \brief
 */

#ifndef CONV3D_DIMENSION_STRATEGY_H
#define CONV3D_DIMENSION_STRATEGY_H

#include "conv_api_strategy.h"
#include "conv_api_tiling_algorithm_BBmode.h"

namespace conv_tiling {

class Conv3DAPIStrategy : public ConvAPIStrategy {
public:
    uint64_t ValidateAndAdjustKL1(uint64_t kValue, uint64_t scaleFactor) const override;
    uint64_t CalcDinFromCinSlice(uint64_t cinSlice, uint64_t cin) const override;
};

} // namespace conv_tiling

#endif
