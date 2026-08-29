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
 * \file avg_pool3_d_grad_simt_tiling.h
 * \brief SIMT scheme tiling for 3D average pooling backward (arch35).
 */

#ifndef OP_IMPL_AVG_POOL3_D_GRAD_SIMT_TILING_H_
#define OP_IMPL_AVG_POOL3_D_GRAD_SIMT_TILING_H_

#include "avg_pool3_d_grad_tiling_base.h"
#include "avg_pool3_d_grad_tiling_common.h"
#include "../../op_kernel/arch35/avg_pool3_d_grad_tiling_data.h"
#include "../../op_kernel/arch35/avg_pool3_d_grad_tiling_key.h"

namespace optiling {

class AvgPool3DGradSimtTiling : public AvgPool3DGradTilingBase {
public:
    explicit AvgPool3DGradSimtTiling(gert::TilingContext* context) : AvgPool3DGradTilingBase(context) {}
    ~AvgPool3DGradSimtTiling() override {}

protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
};

} // namespace optiling

#endif // OP_IMPL_AVG_POOL3_D_GRAD_SIMT_TILING_H_
