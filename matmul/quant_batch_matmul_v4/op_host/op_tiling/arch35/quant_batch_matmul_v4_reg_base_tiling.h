/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_reg_base_tiling.h
 * \brief
 */

#pragma once

#include "quant_batch_matmul_v4_basic_block_tiling.h"
#include "quant_batch_matmul_v4_tiling.h"

namespace optiling {
class QuantBatchMatmulV4RegBase : public QuantBatchMatmulV4TilingBase {
public:
    explicit QuantBatchMatmulV4RegBase(gert::TilingContext* context) : QuantBatchMatmulV4TilingBase(context)
    {
        tilingSolver_.Init();
    }
    ~QuantBatchMatmulV4RegBase() override = default;

protected:
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

    QuantBatchMatmulV4BasicBlockTiling tilingSolver_;

private:
    ge::graphStatus InstantiateTilingData();
    void PrintTilingData(bool debugLevel);
    void SetBubTiling();
    void GetBubTilingA8W4(int64_t& nBubSize, int64_t& kBubSize) const;
    void GetBubTilingA8W4BySize(int64_t& nBubSize, int64_t& kBubSize, int64_t& kBl1Size, int64_t& nBl1Size) const;
    void SetMatmulTiling();
    uint64_t GetGroupNumBub(uint64_t kDimSzie) const;
    uint64_t GetBubSize(uint64_t bubN, uint64_t bubD, bool isWeightNz) const;
    void PrintCVTilingData(const bool debugLevel) const;
    int64_t DumpCVTilingDataToLog(const bool debugLevel) const;
    void PrintMatMulTiling() const;

    qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams tilingData_;
};
} // namespace optiling
