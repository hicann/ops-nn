/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_sliding_window_mx_basic_api_tiling.h
 * \brief
 */
#pragma once

#include "adaptive_sliding_window_tiling.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace optiling {

class AdaptiveSlidingWindowMXBasicAPITiling : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowMXBasicAPITiling(gert::TilingContext* context);
    AdaptiveSlidingWindowMXBasicAPITiling(
        gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3BasicAPITilingData* out);
    ~AdaptiveSlidingWindowMXBasicAPITiling() override = default;

    ge::graphStatus DoLibApiTiling() override;

private:
    void Reset();
    void CalculateNBufferNum4MX();
    void UpdateAFullLoadStatus();

protected:
    bool IsCapable() override;
    uint64_t GetBatchCoreCnt() const override;
    const void* GetTilingData() const override;
    uint64_t GetApiLevel(NpuArch npuArch) const override;
    bool CalcBasicBlock() override;
    void AnalyseFullLoadInfo() override;
    void CalcTailRoundBasicBlockSplit() override;
    bool CalL1Tiling() override;
    void SetTilingData() override;

    DequantBmm::QuantBatchMatmulV3BasicAPITilingData tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3BasicAPITilingData& tilingData_;
};
} // namespace optiling
