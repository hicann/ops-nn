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
 * \file fused_quant_matmul_swiglu_tiling.h
 * \brief
 */
#ifndef FUSED_QUANT_MATMUL_SWIGLU_H
#define FUSED_QUANT_MATMUL_SWIGLU_H
#include "fused_quant_matmul_asw_tiling.h"

namespace optiling {

constexpr uint32_t MIN_SHAPE_M = 16;
constexpr uint32_t MIN_SHAPE_N = 64;
constexpr uint32_t ALIGN_M = 16;
constexpr uint32_t ALIGN_N = 64;

BEGIN_TILING_DATA_DEF(FusedQuantMatmulSwigluBaseParams)
TILING_DATA_FIELD_DEF(uint8_t, isDeqQuant);
TILING_DATA_FIELD_DEF(uint8_t, isQuant);
TILING_DATA_FIELD_DEF(uint8_t, isFullLoadA);
TILING_DATA_FIELD_DEF(uint8_t, reserved0);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreM);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreN);
TILING_DATA_FIELD_DEF(uint32_t, mLoops);
TILING_DATA_FIELD_DEF(uint32_t, nLoops);
TILING_DATA_FIELD_DEF(uint32_t, singleMTail);
TILING_DATA_FIELD_DEF(uint32_t, singleNTail);
TILING_DATA_FIELD_DEF(uint32_t, reserved1);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FusedQuantMatmulSwigluBaseParamsOp, FusedQuantMatmulSwigluBaseParams)

BEGIN_TILING_DATA_DEF(FusedQuantMatmulSwigluTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FusedQuantMatmulSwigluBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FusedQuantMatMul_8192, FusedQuantMatmulSwigluTilingData)

class FusedQuantMatMulSwigluTiling : public FusedQuantMatMulASWTiling {
public:
    FusedQuantMatmulSwigluTilingData tilingData;
    explicit FusedQuantMatMulSwigluTiling(gert::TilingContext *context) : FusedQuantMatMulASWTiling(context) {}
    ~FusedQuantMatMulSwigluTiling() override = default;

protected:
    uint32_t GetX3Idx() const;
    uint32_t GetYScaleIdx() const;

    bool IsFusedSwigluType();

    void SetFormat();
    void PrintMatmulTilingData();
    void PrintBaseParamsTilingData();
    ge::graphStatus InitTilingData();
    void SetBaseBlockTiling();

    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;

    bool IsCapable() override;

    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
};
}
#endif // FUSED_QUANT_MATMUL_SWIGLU_H
