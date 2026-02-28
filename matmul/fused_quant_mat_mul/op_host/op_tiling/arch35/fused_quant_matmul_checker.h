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
 * \file fused_quant_matmul_checker.h
 * \brief
 */
#ifndef FUSED_QUANT_MATMUL_CHECKER_H
#define FUSED_QUANT_MATMUL_CHECKER_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"

#include "../fused_quant_matmul_common.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_checker_for_mmads8s4.h"

namespace optiling {

class FusedQuantMatMulChecker : public QuantBatchMatmulV3Checker4MmadS8S4 {
public:
    FusedQuantMatMulChecker(gert::TilingContext *context, const QuantBatchMatmulInfo &inputParams)
        : QuantBatchMatmulV3Checker4MmadS8S4(context, inputParams) {}

    ~FusedQuantMatMulChecker() = default;
    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                    const std::vector<const gert::StorageShape *> &optionalInputShape,
                    const std::vector<int64_t> &dimValueOfMKN) const;

protected:
    uint32_t GetX1Idx() const override { return X1_INDEX_FQMM; }
    uint32_t GetX2Idx() const override { return X2_INDEX_FQMM; }
    uint32_t GetBiasIdx() const override { return BIAS_INDEX_FQMM; }
    uint32_t GetPertokenIdx() const override { return X1_SCALE_INDEX_FQMM; }
    uint32_t GetScaleIdx() const override { return X2_SCALE_INDEX_FQMM; }
    uint32_t GetOffsetIdx() const override { return X2_OFFSET_INDEX_FQMM; }
    uint32_t GetX2TableIdx() const { return X2_TABLE_INDEX_FQMM; }
    bool CheckShapeInRangeForOptionalInputs(const gert::StorageShape *scaleShape, const gert::StorageShape *biasShape,
                                            const gert::StorageShape *offsetShape) const;
    bool CheckDimValue(const gert::StorageShape *scaleShape, const gert::StorageShape *biasShape,
                       const gert::StorageShape *offsetShape, const std::vector<int64_t> &DimValueOfMKN) const;
};
} // namespace optiling
#endif // FUSED_QUANT_MATMUL_CHECKER_H