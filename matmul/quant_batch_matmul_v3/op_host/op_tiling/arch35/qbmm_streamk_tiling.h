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
 * \file qbmm_streamk_tiling.h
 * \brief StreamK tiling implementation for MX QuantBatchMatmulV3.
 */
#pragma once

#include "../quant_batch_matmul_v3_tiling_base.h"
#include "quant_batch_matmul_v3_tiling_util.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"
#include "matmul/quant_batch_matmul_v3/op_kernel/arch35/quant_batch_matmul_v3_apt_tiling_key.h"

namespace optiling {

class QBMMV3StreamKTiling : public QuantBatchMatmulV3TilingBase {
public:
    explicit QBMMV3StreamKTiling(gert::TilingContext* context);
    ~QBMMV3StreamKTiling() override = default;

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

protected:
    bool IsCapable() override;
    ge::graphStatus CheckContext() override;
    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape*>& mandatoryShape, const gert::StorageShape* biasShape,
                    const gert::StorageShape* pertokenShape,
                    const std::vector<int64_t>& dimValueOfMKN) const override;

private:
    void Reset();
    bool IsMxInput() const;
    bool CalcBaseBlock();
    bool CalcL1Tiling();
    void SetTilingData();
    void CalculateNBufferNum4MX();
    uint64_t GetBiasMode() const;
    uint64_t GetKernelType() const;

    DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData tilingDataSelf_;
    DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData& tilingData_;
    BasicRunInfoTiling basicTiling_;
    uint64_t mCnt_ = 1UL;
    uint64_t nCnt_ = 1UL;
    uint64_t streamKCnt_ = 1UL;
    uint64_t kL1_ = 1UL;
    uint64_t scaleKL1_ = 1UL;
};
} // namespace optiling
