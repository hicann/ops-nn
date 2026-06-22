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
 * \file quant_batch_matmul_inplace_add_mx_basic_api_tiling.h
 * \brief mxFP8 basic-api tiling strategy for QuantBatchMatmulInplaceAdd.
 */
#ifndef QUANT_BATCH_MATMUL_INPLACE_ADD_MX_BASIC_API_TILING_H
#define QUANT_BATCH_MATMUL_INPLACE_ADD_MX_BASIC_API_TILING_H

#include "quant_batch_matmul_inplace_add_helper.h"
#include "../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_mx_basic_api_tiling.h"

namespace optiling {

class QuantBatchMatmulInplaceAddMXBasicAPITiling
    : public QuantBatchMatmulInplaceAddHelper<AdaptiveSlidingWindowMXBasicAPITiling> {
public:
    explicit QuantBatchMatmulInplaceAddMXBasicAPITiling(gert::TilingContext* context);
    ~QuantBatchMatmulInplaceAddMXBasicAPITiling() override = default;

protected:
    bool IsCapable() override;
    const void* GetTilingData() const override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    uint64_t GetKernelType() const override;
    void SetTilingData() override;

private:
    void Reset();
    void UpdateTilingData();
    void SetWithoutBatchTilingData();

    QMMIA::QuantBatchMatmulInplaceAddTilingData basicTilingData_;
    QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData withoutBatchTilingData_;
    bool useWithoutBatchTilingData_ = false;
};

} // namespace optiling

#endif
