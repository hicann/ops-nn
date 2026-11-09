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
 * \file conv_api_strategy.h
 * \brief
 */

#ifndef CONV_API_STRATEGY_H
#define CONV_API_STRATEGY_H

#include <cstdint>

namespace conv_tiling_algo_bb {
class ConvTilingAlgorithmBBmode;
}

namespace conv_tiling {

class ConvAPIStrategy {
public:
    virtual ~ConvAPIStrategy() = default;
    void Bind(conv_tiling_algo_bb::ConvTilingAlgorithmBBmode* bb) { bbMode_ = bb; }

    virtual uint64_t ValidateAndAdjustKL1(uint64_t kValue, uint64_t /* scaleFactor */) const { return kValue; }
    virtual uint64_t CalcDinFromCinSlice(uint64_t, uint64_t) const { return 1; }

protected:
    conv_tiling_algo_bb::ConvTilingAlgorithmBBmode* bbMode_ = nullptr;
};

} // namespace conv_tiling

#endif
