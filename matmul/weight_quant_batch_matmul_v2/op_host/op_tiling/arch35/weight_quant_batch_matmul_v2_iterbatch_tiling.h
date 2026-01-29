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
 * \file weight_quant_batch_matmul_v2_iterbatch_tiling.h
 * \brief
 */

#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_ITERBATCH_TILING_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_ITERBATCH_TILING_H
#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h"
#include "../weight_quant_batch_matmul_v2_tiling.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
namespace optiling {
namespace weight_quant_batch_matmul_v2 {

class WeightQuantBatchMatmulV2IterbatchTiling : public WeightQuantBatchMatmulV2TilingASW {
public:
    explicit WeightQuantBatchMatmulV2IterbatchTiling(gert::TilingContext* context) :
        WeightQuantBatchMatmulV2TilingASW(context) {}
    ~WeightQuantBatchMatmulV2IterbatchTiling() override = default;
protected:
    ge::graphStatus DoOpTiling() override;
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    void GetBroadCastInfo(uint64_t& broadcastNum, uint64_t& innerBatchNum, bool& isBroadcastA, bool& isBroadcastB);

    void CalL1Tiling();
    bool CheckBatch();
    uint32_t CalcIterBatch();
    uint32_t GetGcd(uint32_t numA, uint32_t numB) const;
private:
    uint64_t leftL1Size_ = 0;
};
}  // namespace weight_quant_batch_matmul_v2
}  // namespace optiling
#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_ITERBATCH_TILING_H