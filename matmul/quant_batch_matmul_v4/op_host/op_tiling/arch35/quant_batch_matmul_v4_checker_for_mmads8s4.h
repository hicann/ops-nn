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
 * \file quant_batch_matmul_v4_checker_for_mmads8s4.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V4_CHECKER_FOR_MMADS8S4_H
#define QUANT_BATCH_MATMUL_V4_CHECKER_FOR_MMADS8S4_H
#include "../../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_checker_for_mmads8s4.h"

namespace optiling {

class QuantBatchMatmulV4Checker4MmadS8S4 : public QuantBatchMatmulV3Checker4MmadS8S4 {
public:
    QuantBatchMatmulV4Checker4MmadS8S4(gert::TilingContext *context, const QuantBatchMatmulInfo &inputParams)
        : QuantBatchMatmulV3Checker4MmadS8S4(context, inputParams) {}

    ~QuantBatchMatmulV4Checker4MmadS8S4() override = default;

    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape, const gert::StorageShape *biasShape,
                    const gert::StorageShape *pertokenShape, const std::vector<int64_t> &dimValueOfMKN) const override;

protected:

    // 根据V3/V4原型获取输入index
    uint32_t GetX1Idx() const override
    {
        return X1_INDEX_V4;
    }
    uint32_t GetX2Idx() const override
    {
        return X2_INDEX_V4;
    }
    uint32_t GetScaleIdx() const override
    {
        return X2_SCALE_INDEX_V4;
    }
    uint32_t GetOffsetIdx() const override
    {
        return X2_OFFSET_INDEX_V4;
    }
    uint32_t GetBiasIdx() const override
    {
        return BIAS_INDEX_V4;
    }
    uint32_t GetPertokenIdx() const override
    {
        return X1_SCALE_INDEX_V4;
    }
    uint32_t GetX2TableIdx() const
    {
        return X2_TABLE_INDEX_V4;
    }

    bool CheckDtypesInRange() const override;
    bool CheckABDtypes() const override;
    bool CalcSingleLutSize(const ge::DataType bDtype,
                           const ge::DataType x2TableDtype,
                           uint64_t &singleLutSize) const;
    bool CheckX2TableShape() const;
    bool CheckDimValue(const gert::StorageShape *scaleShape,
                       const gert::StorageShape *biasShape,
                       const gert::StorageShape *offsetShape,
                       const std::vector<int64_t> &dimValueOfMKN) const;
    bool ExtraInputCheck() const override;
};
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V4_CHECKER_FOR_MMADS8S4_H