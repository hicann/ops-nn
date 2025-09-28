/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_reg_base_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE (28) USING BLANK LINES.
 * 
 * 
 * 
 * 
 * 
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_REG_BASE_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_REG_BASE_TILING_H

#include "weight_quant_batch_matmul_v2_basic_block_tiling.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(WeightQuantBatchMatmulV2RegBaseTilingData)
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimN);
TILING_DATA_FIELD_DEF(uint8_t, cubeBlockDimM);
TILING_DATA_FIELD_DEF(uint8_t, vecCoreParallel);
TILING_DATA_FIELD_DEF(uint8_t, reserve1);

TILING_DATA_FIELD_DEF(uint16_t, AL1Pingpong);
TILING_DATA_FIELD_DEF(uint16_t, BL1Pingpong);

TILING_DATA_FIELD_DEF(uint64_t, kSize);
TILING_DATA_FIELD_DEF(uint64_t, nSize);
TILING_DATA_FIELD_DEF(uint64_t, groupSize);
TILING_DATA_FIELD_DEF(uint64_t, mSize);
TILING_DATA_FIELD_DEF(uint64_t, nBubSize);
TILING_DATA_FIELD_DEF(uint64_t, kBubSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100300, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100310, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100311, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100301, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10100300, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10100310, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100200, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100210, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100211, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100201, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100100, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100110, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100111, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_100101, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101300, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101310, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101311, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101301, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10101300, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_10101310, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101200, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101210, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101211, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101201, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101100, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101110, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101111, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_101101, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100300, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100310, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100311, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100301, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_11100300, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_11100310, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100200, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100210, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100211, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100201, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100100, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100110, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100111, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1100101, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101300, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101310, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101311, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101301, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_11101300, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_11101310, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101200, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101210, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101211, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101201, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101100, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101110, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101111, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2_1101101, WeightQuantBatchMatmulV2RegBaseTilingData)
REGISTER_TILING_DATA_CLASS(WeightQuantBatchMatmulV2RegBaseTilingDataOp, WeightQuantBatchMatmulV2RegBaseTilingData)

class WeightQuantBatchMatmulV2RegBase : public WeightQuantBatchMatmulV2Tiling
{
public:
    explicit WeightQuantBatchMatmulV2RegBase(gert::TilingContext* context) : WeightQuantBatchMatmulV2Tiling(context)
    {
        if (context->GetCompileInfo() == nullptr) {
            InitCompileInfo();
        }
        tilingSolver_.Init();
    }

    ~WeightQuantBatchMatmulV2RegBase() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    void PrintCVTilingData(bool debugLevel) const;
    ge::graphStatus PostTiling() override;
    const WeightQuantBatchMatmulV2RegBaseTilingData GetTilingData() const
    {
        return *tilingData_;
    }
    std::unique_ptr<WeightQuantBatchMatmulV2RegBaseTilingData> tilingData_ = nullptr;

private:
    void SetBubTiling();
    void SetAdditionalParam();
    void GetBubTilingA16W4NZ(int64_t& nBubSize, int64_t& kBubSize) const;
    void GetBubTilingA16W4ND(int64_t& nBubSize, int64_t& kBubSize) const;
    void GetBubTilingA16W8NDPerGroup(int64_t& nBubSize, int64_t& kBubSize) const;
    void SetMatmulTiling();

    WeightQuantBatchMatmulV2BasicBlockTiling tilingSolver_;

private:
    ge::graphStatus InstantiateTilingData();
};
} // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_REG_BASE_TILING_H