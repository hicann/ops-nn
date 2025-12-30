/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss_grad_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_GRAD_V2_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_GRAD_V2_TILING_H_

#include "../../op_kernel/arch35/smooth_l1_loss_grad_v2_tiling_key.h"
#include "smooth_l1_loss_grad_v2_tiling_base.h"
#include "tiling_base/tiling_base.h"

namespace optiling {

class SmoothL1LossGradV2TilingClass : public Ops::NN::Optiling::TilingBaseClass{
public:
    explicit SmoothL1LossGradV2TilingClass(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus CalcReduceMeanCof();
    ge::graphStatus CalcSigma();
    ge::graphStatus GetDoutIsScalar();
    ge::graphStatus DoScalarDagOpTiling();
    ge::graphStatus DoTensorDagOpTiling();

    template <typename T>
    ge::graphStatus DoDagTilingForType();

    template <typename T>
    ge::graphStatus DoScalarDagTilingForType();
private:
    uint64_t tilingKey = 0;
    const char *reducationStr = "";
    ge::DataType inputDtype;
    uint32_t doutIsScalar = 0;
    float sigma = 1.0f;
    float negSigma = 1.0f;
    float invertSigma = 1.0f;
    float reduceMeanCof = 1.0f;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_GRAD_V2_H_