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
 * \file kl_div_loss_grad_tiling.h
 * \brief  kl_div_loss_grad_tiling.h head file
 */
#ifndef KL_DIV_LOSS_GRAD_TILING_H_
#define KL_DIV_LOSS_GRAD_TILING_H_

#include "kl_div_loss_grad_tiling_arch35.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_base.h"

#include "../../../../loss/kl_div_loss_grad/op_kernel/arch35/kl_div_loss_grad_dag.h"
#include "../../../../loss/kl_div_loss_grad/op_kernel/arch35/kl_div_loss_grad_tiling_key.h"
namespace optiling {
using namespace Ops::NN::Optiling;

class KlDivLossGradTiling : public TilingBaseClass
{
public:
    explicit KlDivLossGradTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckInputShape();
    float CalcReductionCof();
    ge::graphStatus CalcDiffDtype();

    ge::graphStatus RunFp16BroadcastTiling(float reducationCof);
    ge::graphStatus RunFp32BroadcastTiling(float reducationCof);
private:
    const char *reducationStr = "";  
    ge::DataType outputYDtype;
    ge::DataType inputGradDtype;
    ge::DataType inputInputDtype;
    ge::DataType inputTargetDtype;
    uint64_t tilingKey_ = 0;
    bool logTarget_ = false;
    float reducationCof_ = 0.0f;
};
}  // namespace optiling
#endif  // KL_DIV_LOSS_GRAD_TILING_H_