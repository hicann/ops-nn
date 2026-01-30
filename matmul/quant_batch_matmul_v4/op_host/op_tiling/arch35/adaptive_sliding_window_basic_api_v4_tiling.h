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
 * \file adaptive_sliding_window_basic_api_v4_tiling.h
 * \brief
 */
#ifndef ADAPTIVE_SLIDING_WINDOW_BASIC_API_V4_TILING_H
#define ADAPTIVE_SLIDING_WINDOW_BASIC_API_V4_TILING_H
#include "../../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_basic_api_tiling.h"
#include "../quant_batch_matmul_v4_compile_info.h"


namespace optiling {


class AdaptiveSlidingWindowBasicTilingV4 : public AdaptiveSlidingWindowBasicAPITiling {
public:
    explicit AdaptiveSlidingWindowBasicTilingV4(gert::TilingContext *context) : AdaptiveSlidingWindowBasicAPITiling(context) {}
    ~AdaptiveSlidingWindowBasicTilingV4() override = default;
    uint64_t GetTilingKey() const override;

protected:

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

    bool IsCapable() override;
    bool CheckDtype() const override;
    ge::graphStatus CheckContext() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckInputValidInPerblockMode(const gert::Shape& scaleShape, const gert::StorageShape *pertokenShape,
                                       const gert::Shape& x1Shape, const gert::Shape& x2Shape) const;
    bool CheckDimValidInPerblockMode(size_t x1ShapeLen, size_t x2ShapeLen,
                                     size_t pertokenShapeLen, size_t scaleShapeLen) const;
    bool CheckBatchValidInPerblockMode(const gert::Shape& scaleShape, const gert::Shape& pertoken,
                                       const gert::Shape& x1Shape, const gert::Shape& x2Shape) const;
    bool CheckGroupValidInPerblockMode() const;
    bool CheckShapeValidInPerblockMode(const gert::Shape& scaleShape,
                                       const gert::Shape& pertoken, const gert::Shape& x1Shape,
                                       const gert::Shape& x2Shape) const;
    bool SetPlatformInfoForTiling() override;
 
    std::unique_ptr<QuantBatchMatmulV4CompileInfo> compileInfoPtr_;
};
}  // namespace optiling
#endif  // ADAPTIVE_SLIDING_WINDOW_BASIC_API_V4_TILING_H