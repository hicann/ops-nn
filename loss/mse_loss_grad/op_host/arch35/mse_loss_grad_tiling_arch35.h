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
 * \file mse_loss_grad_tiling.h
 * \brief mse_loss_grad_tiling
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MSE_LOSS_GRAD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MSE_LOSS_GRAD_TILING_H

#include "register/op_impl_registry.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {

class MseLossGradTilingClass : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit MseLossGradTilingClass(gert::TilingContext* context) : Ops::NN::Optiling::TilingBaseClass(context)
    {}

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
    ge::graphStatus CheckDoutIsScalar();
    ge::graphStatus DoScalarDagOpTiling();
    ge::graphStatus DoTensorDagOpTiling();

private:
    uint64_t tilingKey = 0;
    const char* reducationStr = "";
    ge::DataType inputDtype;
    float reduceMeanCof = 2.0f;
    uint32_t doutIsScalar = 0;
};

struct MseLossGradCompileInfo {
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_MSE_LOSS_GRAD_TILING_H