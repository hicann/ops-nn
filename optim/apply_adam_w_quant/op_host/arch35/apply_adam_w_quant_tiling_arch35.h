/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_adam_w_quant_tiling_arch35.h
 * \brief ApplyAdamWQuant regbase (arch35 / Ascend950) tiling class declaration.
 */
#ifndef APPLY_ADAM_W_QUANT_ARCH35_TILING_H
#define APPLY_ADAM_W_QUANT_ARCH35_TILING_H

#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/apply_adam_w_quant_tiling_data.h"

namespace optiling {
struct ApplyAdamWQuantRegbaseCompileInfo {};

class ApplyAdamWQuantRegbaseTiling {
public:
    explicit ApplyAdamWQuantRegbaseTiling(gert::TilingContext* context) : tilingContext_(context) {}

    ge::graphStatus RunTiling();

protected:
    ge::graphStatus GetAttributes();
    ge::graphStatus CheckInputShape();
    ge::graphStatus DetermineTilingKey();
    ge::graphStatus DoTiling();
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext_ = nullptr;

    // attrs
    float lr_ = 0.0f;
    float beta1_ = 0.0f;
    float beta2_ = 0.0f;
    float weightDecay_ = 0.0f;
    float eps_ = 0.0f;
    float gnormScale_ = 0.0f;
    int64_t blockSize_ = 0;

    // tiling result
    uint64_t tilingKey_ = 0;
    uint64_t useNumCore_ = 0;
    uint64_t lastPreCoreRowWork_ = 0;
    uint64_t notLastCoreNum_ = 0;
    uint64_t notLastPreCoreRowWork_ = 0;
    uint64_t lastCoreLastBlock_ = 0;
    uint64_t lastBlockSize_ = 0;
    uint64_t perCoreDoBlockNum_ = 0;
};
} // namespace optiling
#endif // APPLY_ADAM_W_QUANT_ARCH35_TILING_H
