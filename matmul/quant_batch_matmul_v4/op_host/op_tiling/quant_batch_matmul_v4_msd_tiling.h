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
 * \file quant_batch_matmul_v4_msd_tiling.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_MSD_TILING_H
#define QUANT_BATCH_MATMUL_V4_MSD_TILING_H

#include <cstdint>
#include <vector>

#include "tiling/tiling_api.h"
#include "quant_batch_matmul_v4_compile_info.h"
#include "quant_batch_matmul_v4_tiling_info.h"
#include "op_host/tiling_base.h"


#include "op_host/tiling_templates_registry.h"
#include "common/op_host/op_tiling/tiling_type.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/platform_util.h"
namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;

class QuantBatchMatmulV4MsdTiling : public QuantBatchMatmulV3BasicTiling
{
public:
    explicit QuantBatchMatmulV4MsdTiling(gert::TilingContext* contextIn)
        : QuantBatchMatmulV3BasicTiling(contextIn), context(contextIn)
    {}
    QuantBatchMatmulV4MsdTiling(gert::TilingContext* contextIn, QuantBatchMatmulV3TilingData* out)
        : QuantBatchMatmulV3BasicTiling(contextIn, out), context(contextIn)
    {}

    ~QuantBatchMatmulV4MsdTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    bool AnalyzeScaleInputs();
    bool AnalyzeYOffsetInputs();
    void InitPlatformInfo(const QuantBatchMatmulV3CompileInfo* compileInfoPtr_, matmul_tiling::PlatformInfo& platformInfo) const;
    ge::graphStatus CheckContext();
    ge::graphStatus InstantiateTilingData();
    void PrintTilingData() const;
    void PrintMatmulTilingData() const;
    bool AnalyzeAttrs();
    bool AnalyzeDtype();
    bool AnalyzeInputs();
    bool SetMatmulTilingPerGroup();
    bool DoOpTilingPerGroup();
    bool DoOpTilingPerChannel();
    void SetMatmulTilingFromBasicTiling();
    bool AnalyzeScalePerChannel(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                                int64_t x1ScaleShapeLen, int64_t x2ScaleShapeLen);
    bool AnalyzeScalePerGroup(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape,
                              int64_t x1ScaleShapeLen, int64_t x2ScaleShapeLen);
    const gert::Shape GetShape(const size_t index) const;
    const gert::Shape GetOptionShape(const size_t index) const;
    std::unique_ptr<QuantBatchMatmulV4MsdTilingData> tilingDataManager_;
    QuantBatchMatmulV4MsdTilingData* tilingData_ = nullptr;
private:
    gert::TilingContext* context;
    QuantBatchMatmulV4QuantType antiQuantType;
    uint64_t actualMSize;
};
}   // namespace optiling
#endif  // QUANT_BATCH_MATMUL_V4_MSD_TILING_H