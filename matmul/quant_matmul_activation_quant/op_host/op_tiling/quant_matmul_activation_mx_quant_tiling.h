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
 * \file quant_matmul_activation_mx_quant_tiling.h
 * \brief mxFP8 basic-api tiling strategy for QuantMatmulActivationQuant.
 */
#ifndef QUANT_MATMUL_ACTIVATION_MX_QUANT_TILING_H
#define QUANT_MATMUL_ACTIVATION_MX_QUANT_TILING_H

#include "quant_matmul_activation_quant_helper.h"
#include "matmul/quant_batch_matmul_v3/op_host/op_tiling/arch35/adaptive_sliding_window_mx_basic_api_tiling.h"

namespace optiling {

class QuantMatmulActivationQuantMXBasicAPITiling
    : public QuantMatmulActivationQuantHelper<AdaptiveSlidingWindowMXBasicAPITiling> {
public:
    explicit QuantMatmulActivationQuantMXBasicAPITiling(gert::TilingContext* context);
    ~QuantMatmulActivationQuantMXBasicAPITiling() override = default;

protected:
    bool IsCapable() override;
    bool CheckCoreNum() const override;
    const void* GetTilingData() const override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus UpdateTilingData();
    uint64_t GetTilingKey() const override;
    uint64_t GetKernelType() const override;
    uint64_t GetBatchMode() const override;

private:
    void Reset();
    ge::graphStatus ValidateTilingData() const;
    void SetFinalTilingData();

    QMMAQ::QMMAQTilingData tilingData_;
    QMMAQ::QMMAQWithoutBatchTilingData withoutBatchTilingData_;
};

} // namespace optiling

#endif
