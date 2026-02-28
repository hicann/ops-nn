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
 * \file quant_batch_matmul_inplace_add_tiling.h
 * \brief
 */
#ifndef QUANT_BATCH_MATMUL_INPLACE_ADD_TILING_H
#define QUANT_BATCH_MATMUL_INPLACE_ADD_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include <graph/utils/type_utils.h>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "../quant_batch_matmul_inplace_add_host_utils.h"
#include "../../op_kernel/arch35/quant_batch_matmul_inplace_add_tiling_data.h"
#include "../../op_kernel/arch35/quant_batch_matmul_inplace_add_tiling_key.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_basic_api_tiling.h"

namespace optiling {

class QuantBatchMatmulInplaceAddTiling : public AdaptiveSlidingWindowBasicAPITiling {
public:
    explicit QuantBatchMatmulInplaceAddTiling(gert::TilingContext* context);
    QuantBatchMatmulInplaceAddTiling(gert::TilingContext* context, QMMIA::QuantBatchMatmulInplaceAddTilingData* out);
    ~QuantBatchMatmulInplaceAddTiling() override = default;

protected:
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 5、 计算TilingKey
    uint64_t GetTilingKey() const override;
    void Reset();

private:
    bool AnalyzeAttrs() override;
    bool AnalyzeDtype() override;
    bool AnalyzeInputs() override;
    bool CheckDtype();
    bool IsFp8Dtype(const ge::DataType dtype) const;
    bool CheckParamsForMxQuant(const gert::Shape &x1ScaleShape, const gert::Shape &x2ScaleShape) const;
    bool CheckShapeVaild(const gert::Shape &x1Shape, const gert::Shape &x2Shape) const;
    QMMIA::QuantBatchMatmulInplaceAddTilingData tilingDataSelf_;
    QMMIA::QuantBatchMatmulInplaceAddTilingData& tilingData_;
};
} // namespace optiling

#endif