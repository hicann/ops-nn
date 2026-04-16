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
 * \file quant_batch_matmul_v4_pergroup_arch35_tiling.h
 * \brief
 */

#pragma once

#include "../quant_batch_matmul_v4_pergroup_tiling.h"
#include "quant_batch_matmul_v4_tiling.h"
#include "../quant_batch_matmul_v4_tiling_info.h"

namespace optiling {

struct QuantBatchMatmulPergroupArch35Info : public optiling::QuantBatchMatmulInfo {
    // Extra metadata for int4 K-G quantification path on arch35.
    ge::DataType x2OffsetDtype = ge::DT_UNDEFINED;
};

class QuantBatchMatmulV4PergroupArch35Tiling : public QuantBatchMatmulV4PergroupTiling {
public:
    explicit QuantBatchMatmulV4PergroupArch35Tiling(gert::TilingContext* contextIn)
        : QuantBatchMatmulV4PergroupTiling(contextIn)
    {}
    QuantBatchMatmulV4PergroupArch35Tiling(gert::TilingContext* contextIn, QuantBatchMatmulV3TilingData* out)
        : QuantBatchMatmulV4PergroupTiling(contextIn, out)
    {}
    ~QuantBatchMatmulV4PergroupArch35Tiling() override = default;

protected:
    const gert::Shape GetShape(const size_t index);
    const gert::Shape GetOptionShape(const size_t index);
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus CheckContext() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    bool IsCapable() override;
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool SetPlatformInfoForTiling() override;
    ge::graphStatus CalcDequantTiling(uint32_t baseM, uint32_t baseN, uint32_t groupSizeK);
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    bool CheckPergroupAttrs() const;
    bool CheckPergroupDtype() const;
    bool CheckPergroupShape();
    bool CheckPergroupBasicShapeConstraints() const;
    bool CheckPergroupDimAndOutput(
        const gert::Shape& x1Shape, const gert::Shape& x2Shape, const gert::Shape& x1ScaleShape,
        const gert::Shape& x2ScaleShape, const gert::Shape& x2OffsetShape);
    bool CheckPergroupScaleShape(const gert::Shape& x1ScaleShape, const gert::Shape& x2ScaleShape, const gert::Shape& x2OffsetShape) const;
    bool CheckPergroupInputFormat() const;
    QuantBatchMatmulPergroupArch35Info inputParamsPergroup_;
};
} // namespace optiling

