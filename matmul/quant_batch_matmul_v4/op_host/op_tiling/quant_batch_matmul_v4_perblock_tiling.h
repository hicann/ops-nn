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
 * \file quant_batch_matmul_v4_perblock_tiling.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_PERBLOCK_TILING_H
#define QUANT_BATCH_MATMUL_V4_PERBLOCK_TILING_H

#include <cstdint>
#include <vector>

#include "op_host/tiling_templates_registry.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "tiling/tiling_api.h"

#include "quant_batch_matmul_v4_compile_info.h"
#include "quant_batch_matmul_v4_msd_tiling.h"
#include "quant_batch_matmul_v4_tiling_info.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.h"

namespace optiling {

constexpr int32_t GROUP_M_OFFSET = 32;
constexpr int32_t GROUP_N_OFFSET = 16;
constexpr uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
constexpr int32_t CV_PARALL_NUM = 2;

struct QuantBatchMatmulPerblockInfo: public QuantBatchMatmulMsdInfo {
};

class QuantBatchMatmulV4PerblockTiling : public QuantBatchMatmulV3BasicTiling
{
public:
    explicit QuantBatchMatmulV4PerblockTiling(gert::TilingContext* contextIn)
        : QuantBatchMatmulV3BasicTiling(contextIn), context(contextIn)
    {}
    QuantBatchMatmulV4PerblockTiling(gert::TilingContext* contextIn, QuantBatchMatmulV3TilingData* out)
        : QuantBatchMatmulV3BasicTiling(contextIn, out), context(contextIn)
    {}
    ~QuantBatchMatmulV4PerblockTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    bool AnalyzeScaleInputs();
    bool AnalyzeBiasInputs();
    void Reset();
    ge::graphStatus CheckContext() override;
    ge::graphStatus InstantiateTilingData();
    void PrintTilingData();
    void PrintMatmulTilingData();
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool AnalyzeX1ScaleInput();
    bool AnalyzeX2ScaleInput();
    ge::graphStatus SetBasicTilingValue();
    void SetTilingData();
    const gert::Shape GetShape(const size_t index);
    const gert::Shape GetOptionShape(const size_t index);
    std::unique_ptr<QuantBatchMatmulV4PerblockTilingData> tilingDataManager_;
    QuantBatchMatmulV4PerblockTilingData* tilingData_ = nullptr;
    bool compileInfoInit_ = false;

private:
    gert::TilingContext* context;
    QuantBatchMatmulV3Trans trans_ = QuantBatchMatmulV3Trans::NO_TRANS;
};

} // namespace optiling
#endif // QUANT_BATCH_MATMUL_V4_PERBLOCK_TILING_H