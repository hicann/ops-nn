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
 * \file adaptive_avg_pool2d_grad_simt_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H_
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H_

#include "adaptive_avg_pool2d_grad_base_tiling.h"
#include "../op_kernel/arch35/adaptive_avg_pool2d_grad_struct.h"

using namespace AdaptiveAvgPool2dGradOp;
namespace optiling {
class AdaptiveAvgPool2dGradTilingSimt : public AdaptiveAvgPool2dGradTilingBase {
public:
    explicit AdaptiveAvgPool2dGradTilingSimt(gert::TilingContext* context) : AdaptiveAvgPool2dGradTilingBase(context)
    {}
    ~AdaptiveAvgPool2dGradTilingSimt() override
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    bool NeedInt64(int64_t isize, int64_t osize) const;
    AdaptiveAvgPool2dGradSimtTiling* tilingData_ = context_->GetTilingData<AdaptiveAvgPool2dGradSimtTiling>();
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADAPTIVE_AVG_POOL2D_GRAD_SIMT_H_
