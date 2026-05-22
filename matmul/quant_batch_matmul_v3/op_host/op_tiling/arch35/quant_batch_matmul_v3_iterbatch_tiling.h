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
 * \file quant_batch_matmul_v3_iterbatch_tiling.h
 * \brief Iter-batch tiling implementation for QuantBatchMatmulV3.
 */
#pragma once
#include "util/math_util.h"
#include "../quant_batch_matmul_v3_tiling_base.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_apt_tiling_key.h"

namespace optiling {

class QuantBatchMatmulV3IterbatchTiling : public QuantBatchMatmulV3TilingBase {
public:
    explicit QuantBatchMatmulV3IterbatchTiling(gert::TilingContext* context);
    ~QuantBatchMatmulV3IterbatchTiling() override = default;

    // 1. Query platform information such as core count and memory sizes.
    ge::graphStatus GetPlatformInfo() override;
    // 2. Parse input, output, and attribute metadata.
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3. Compute operator tiling data.
    ge::graphStatus DoOpTiling() override;
    // 4. Compute libapi tiling data for direct consumers such as MC2.
    ge::graphStatus DoLibApiTiling() override;
    // 5. Compute the tiling key.
    uint64_t GetTilingKey() const override;
    // 6. Compute workspace size.
    ge::graphStatus GetWorkspaceSize() override;
    // 7. Persist tiling data to the runtime context.
    ge::graphStatus PostTiling() override;

protected:
    bool IsCapable() override;
    bool CheckDtype() const override;
    void GetBroadCastInfo(uint64_t& broadcastNum, uint64_t& innerBatchNum, bool& isBroadcastA, bool& isBroadcastB);
    bool CheckShape(
        const std::vector<gert::Shape*>& mandatoryShape, const gert::StorageShape* biasShape,
        const gert::StorageShape* pertokenShape, const std::vector<int64_t>& dimValueOfMKN) const override;
    uint32_t CalcIterBatch();
    void SetTilingData();
    uint64_t GetBiasMode() const;
    uint64_t GetKernelType() const;
    DequantBmm::QuantBatchMatmulV3TilingDataParams tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3TilingDataParams& tilingData_;

private:
    BasicRunInfoTiling basicTiling_;
    void Reset();
};
} // namespace optiling
