/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file quant_batch_matmul_v3_pergroup_tiling.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_PERGROUP_TILING_H
#define QUANT_BATCH_MATMUL_V4_PERGROUP_TILING_H
#include <cstdint>
#include <vector>
#include <string>
#include "tiling_base/tiling_templates_registry.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "tiling/tiling_api.h"
#include "quant_batch_matmul_v4_msd_tiling.h"
#include "quant_batch_matmul_v4_tiling_info.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.h"

namespace optiling {
using namespace std;

struct QuantBatchMatmulPergroupInfo : public QuantBatchMatmulInfo {
    QuantBatchMatmulV4QuantType antiQuantType;
    ge::DataType x2OffsetDtype;
};

REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV4_1040, QuantBatchMatmulV3TilingData)

class QuantBatchMatmulV4PergroupTiling : public QuantBatchMatmulV3BasicTiling {
public:
    explicit QuantBatchMatmulV4PergroupTiling(gert::TilingContext *contextIn)
     : QuantBatchMatmulV3BasicTiling(contextIn)
    {}
    QuantBatchMatmulV4PergroupTiling(gert::TilingContext *contextIn, QuantBatchMatmulV3TilingData *out)
     : QuantBatchMatmulV3BasicTiling(contextIn, out)
    {}
    ~QuantBatchMatmulV4PergroupTiling() override = default;

protected:
    const gert::Shape GetShape(const size_t index);
    const gert::Shape GetOptionShape(const size_t index);
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus CheckContext() override;
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckInputLen(size_t actualLen, size_t expectedLen, const string& inputParamName);
    bool CheckInputsLen(
        const gert::Shape& x1Shape, const gert::Shape& x2Shape, const gert::Shape& x1ScaleShape,
        const gert::Shape& x2ScaleShape, const gert::Shape& x2OffsetShape);
    bool CheckInputsShape(
        const gert::Shape& x1Shape, const gert::Shape& x2Shape, const gert::Shape& x1ScaleShape,
        const gert::Shape& x2ScaleShape, const gert::Shape& x2OffsetShape);
    bool CheckFormat();
    ge::graphStatus CalcDequantTiling(uint32_t baseM, uint32_t baseN, uint32_t groupSizeK);
    QuantBatchMatmulPergroupInfo inputParamsPergroup_;
    uint64_t singleCoreM_ = 0;
    uint64_t singleCoreN_ = 0;
};
}   // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V4_PERGROUP_TILING_H