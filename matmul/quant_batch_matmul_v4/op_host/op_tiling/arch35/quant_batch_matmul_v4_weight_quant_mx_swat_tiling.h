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
 * \file quant_batch_matmul_v4_weight_quant_mx_swat_tiling.h
 * \brief
 */

#pragma once

#include "quant_batch_matmul_v4_tiling.h"

namespace optiling {
class QuantBatchMatmulV4WeightQuantMxSwatTiling : public QuantBatchMatmulV4TilingBase {
public:
    explicit QuantBatchMatmulV4WeightQuantMxSwatTiling(gert::TilingContext* context)
        : QuantBatchMatmulV4TilingBase(context)
    {}
    ~QuantBatchMatmulV4WeightQuantMxSwatTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    bool CalcUBSize(uint64_t vecSingleN, uint64_t vecSingleK) const override
    {
        (void)vecSingleN;
        (void)vecSingleK;
        return true;
    }
    bool SetQuantType(const gert::StorageShape* quantScaleShape, const gert::StorageShape* quantOffsetShape) override
    {
        (void)quantScaleShape;
        (void)quantOffsetShape;
        return true;
    }

private:
    bool IsWeightQuantMxSwatScenario() const;
    void PrintSwatTilingData(bool debugLevel) const;
    void DumpSwatTilingDataToLog(bool debugLevel) const;

    qbmmv4_tiling::QuantBatchMatmulV4WeightQuantMxSwatTilingData tilingData_;
};
} // namespace optiling
