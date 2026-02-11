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
 * \file weight_quant_batch_matmul_v2_tiling_custom_backward_compatible.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_BACKWARD_COMPATIBLE_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_BACKWARD_COMPATIBLE_H

#include "../../weight_quant_batch_matmul_v2_tiling_custom.h"

namespace optiling {
class WeightQuantBatchMatmulV2TilingCustomBackwardCompatible : public WeightQuantBatchMatmulV2TilingCustom {
public:
    explicit WeightQuantBatchMatmulV2TilingCustomBackwardCompatible(gert::TilingContext* context)
        : WeightQuantBatchMatmulV2TilingCustom(context)
    {}
    WeightQuantBatchMatmulV2TilingCustomBackwardCompatible(
        gert::TilingContext* context, WeightQuantBatchMatmulV2TilingData* out)
        : WeightQuantBatchMatmulV2TilingCustom(context, out)
    {}

    ~WeightQuantBatchMatmulV2TilingCustomBackwardCompatible() override = default;

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
};

} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_TILING_CUSTOM_BACKWARD_COMPATIBLE_H
