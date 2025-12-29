/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_quant_matmul_asw_tiling.h
 * \brief
 */
#ifndef FUSED_QUANT_MATMUL_ASW_TILING_H
#define FUSED_QUANT_MATMUL_ASW_TILING_H


#include "fused_quant_matmul_common.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"
#include "../../../op_kernel/arch35/fused_quant_mat_mul_tilingkey.h"

namespace optiling {

using FusedQuantMatMulCompileInfo = QuantBatchMatmulV3CompileInfo;

class FusedQuantMatMulASWTiling : public AdaptiveSlidingWindowTiling {
public:
    explicit FusedQuantMatMulASWTiling(gert::TilingContext *context) : AdaptiveSlidingWindowTiling(context) {}
    ~FusedQuantMatMulASWTiling() override = default;

    uint64_t GetTilingKey() const override;

protected:
    uint32_t GetX1Idx() const override { return X1_INDEX_FQMM; }
    uint32_t GetX2Idx() const override { return X2_INDEX_FQMM; }
    uint32_t GetBiasIdx() const override { return BIAS_INDEX_FQMM; }
    uint32_t GetPertokenIdx() const override { return X1_SCALE_INDEX_FQMM; }
    uint32_t GetScaleIdx() const override { return X2_SCALE_INDEX_FQMM; }
    uint32_t GetYScaleIdx() const { return Y_SCALE_INDEX_FQMM; }
    uint32_t GetX1OffsetIdx() const { return X1_OFFSET_INDEX_FQMM; }
    uint32_t GetOffsetIdx() const override { return X2_OFFSET_INDEX_FQMM; }
    uint32_t GetYOffsetIdx() const { return Y_OFFSET_INDEX_FQMM; }
    uint32_t GetX2TableIdx() const { return X2_TABLE_INDEX_FQMM; }
    uint32_t GetX3Idx() const { return X3_INDEX_FQMM; }

    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                    const std::vector<const gert::StorageShape *> &optionalInputShape,
                    const std::vector<int64_t> &dimValueOfMKN) const;
    bool AnalyzeX2TableAttr(const gert::StorageShape *x2TableShape);
    bool CheckUnsupportedOptionInputs();
    ge::graphStatus CheckContext() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckShapeInRangeForMandtoryInputs(size_t x1ShapeLen, size_t x2ShapeLen) const;
    bool AnalyzeAttrs() override;
    uint64_t fusedOpType_ = 0UL;
};
} // namespace optiling
#endif // FUSED_QUANT_MATMUL_ASW_TILING_H