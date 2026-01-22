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
 * \file quant_batch_matmul_v4_asw_tiling.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_V4_ASW_TILING_H
#define QUANT_BATCH_MATMUL_V4_ASW_TILING_H
#include "../../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_tiling.h"
#include "../quant_batch_matmul_v4_compile_info.h"


namespace optiling {

constexpr uint64_t STEP_K_VALUE_4 = 4;

class AdaptiveSlidingWindowTilingV4 : public AdaptiveSlidingWindowTiling {
public:
    explicit AdaptiveSlidingWindowTilingV4(gert::TilingContext *context) : AdaptiveSlidingWindowTiling(context) {}
    ~AdaptiveSlidingWindowTilingV4() override = default;
    uint64_t GetTilingKey() const override;

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

    bool CheckDtype() const override;
    bool CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
                    const gert::StorageShape *biasShape,
                    const gert::StorageShape *pertokenShape,
                    const gert::StorageShape *x2TableShape,
                    const std::vector<int64_t> &dimValueOfMKN) const;
    ge::graphStatus CheckContext() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CalcBasicBlock() override;
    void IsABFullLoad() override;
    void IsBFullLoad() override;
    void CalcTailBasicBlockAfullLoad() override;
    void CalcTailBasicBlock() override;
    bool IsCalL1TilingDepth4MmadS8S4() const override;
    void CalL1TilingDepth4MmadS8S4(uint64_t leftL1Size) override;
    bool Is4BitInput(ge::DataType dtype, bool isLut = false) const;
    bool Is8BitInput(ge::DataType dtype, bool isLut = false) const;
    // 根据size(字节数)和数据类型，计算得出shape(元素个数)
    uint64_t GetShapeWithDataType(uint64_t size, ge::DataType dtype, bool isLut = false) const;
    // 根据shape(元素个数)和数据类型，计算得出size(字节数)
    uint64_t GetSizeWithDataType(uint64_t shape, ge::DataType dtype, bool isLut = false) const;
    bool SetPlatformInfoForTiling() override;

    std::unique_ptr<QuantBatchMatmulV4CompileInfo> compileInfoPtr_;
};
}  // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V4_ASW_TILING_H