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
 * \file quant_batch_matmul_inplace_add_cube_basic_api_tiling.h
 * \brief HIFLOAT8 TT cube basic-api tiling strategy for QuantBatchMatmulInplaceAdd.
 */
#ifndef QUANT_BATCH_MATMUL_INPLACE_ADD_CUBE_BASIC_API_TILING_H
#define QUANT_BATCH_MATMUL_INPLACE_ADD_CUBE_BASIC_API_TILING_H

#include "quant_batch_matmul_inplace_add_helper.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_cube_basic_api_tiling.h"

namespace optiling {

class QuantBatchMatmulInplaceAddCubeBasicAPITiling
    : public QuantBatchMatmulInplaceAddHelper<AdaptiveSlidingWindowCubeBasicAPITiling> {
public:
    explicit QuantBatchMatmulInplaceAddCubeBasicAPITiling(gert::TilingContext* context);
    ~QuantBatchMatmulInplaceAddCubeBasicAPITiling() override = default;

protected:
    bool IsCapable() override;
    const void* GetTilingData() const override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    uint64_t GetKernelType() const override;
    void SetTilingData() override;

private:
    void Reset();

    QMMIA::QuantBatchMatmulInplaceAddTilingData tilingDataSelf_;
    QMMIA::QuantBatchMatmulInplaceAddTilingData& tilingData_;
};

} // namespace optiling

#endif
