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
 * \file l1_loss_grad_tiling.h
 * \brief l1_loss_grad_tiling
 */
 
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_L1_LOSS_GRAD_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_L1_LOSS_GRAD_TILING_H
 
#include "../../op_kernel/arch35/l1_loss_grad_tiling_key.h"
#include "tiling_base/tiling_base.h"
#include "register/op_impl_registry.h"

using namespace Ops::NN::Optiling;
namespace optiling {

class L1LossGradTiling : public TilingBaseClass {
public:
    explicit L1LossGradTiling(gert::TilingContext* context) : TilingBaseClass(context) {};

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus DoTiling();
    ge::graphStatus CheckGradsIsScalar();
    ge::graphStatus CaluateReduceElts();
private:
    uint64_t tilingKey_ = 0;
    uint32_t inputGradsIsScalar_ = 0;
    float reduceElts_ = 1.0;
    ge::DataType outputDtype_;
};

}  // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_L1_LOSS_GRAD_TILING_H