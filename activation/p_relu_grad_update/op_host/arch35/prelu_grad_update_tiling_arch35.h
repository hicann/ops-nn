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
 * \file prelu_grad_update_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_PRELUGRAD_UPDATE_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_PRELUGRAD_UPDATE_TILING_H

#include "register/tilingdata_base.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;

namespace optiling {

class PReluGradUpdateTiling : public TilingBaseClass
{
public:
    explicit PReluGradUpdateTiling(gert::TilingContext* context) : TilingBaseClass(context){};

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
    ge::graphStatus CheckAndInferShape(std::vector<gert::Shape>& inputShapes);

private:
    uint64_t tilingKey = 0;

    ge::DataType inputFeturesDtype;
    ge::DataType inputWeightsDtype;
    ge::DataType inputGradientsDtype;
    ge::DataType outputBackpropsDxDtype;
    ge::DataType outputBackpropsDaDtype;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_PRELUGRAD_UPDATE_TILING_H