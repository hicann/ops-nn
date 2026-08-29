/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file avg_pool3_d_grad_ksize_one_tiling.h
 * \brief KsizeOne scheme tiling for 3D average pooling backward (arch35).
 *        When kD=kH=kW=1, backward is a simple stride-sampled DataCopy + Div.
 */

#ifndef OP_IMPL_AVG_POOL3_D_GRAD_KSIZE_ONE_TILING_H_
#define OP_IMPL_AVG_POOL3_D_GRAD_KSIZE_ONE_TILING_H_

#include "avg_pool3_d_grad_tiling_base.h"
#include "avg_pool3_d_grad_tiling_common.h"
#include "../../op_kernel/arch35/avg_pool3_d_grad_tiling_data.h"
#include "../../op_kernel/arch35/avg_pool3_d_grad_tiling_key.h"

namespace optiling {

class AvgPool3DGradKsizeOneTiling : public AvgPool3DGradTilingBase {
public:
    explicit AvgPool3DGradKsizeOneTiling(gert::TilingContext* context) : AvgPool3DGradTilingBase(context) {}
    ~AvgPool3DGradKsizeOneTiling() override {}

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    void DoBlockTiling();
    void CalcDivisor();
    ge::graphStatus SetTilingData();

    int64_t ubBufferSize_ = 0;
    int64_t elementsPerCore_ = 0;
    int64_t tailCoreElements_ = 0;
    int64_t totalElements_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t divisor_ = 1;
};

} // namespace optiling

#endif // OP_IMPL_AVG_POOL3_D_GRAD_KSIZE_ONE_TILING_H_
