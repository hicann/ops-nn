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
 * \file relu_grad_v2_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_RELU_GRAD_V2_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_RELU_GRAD_V2_TILING_H

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "atvoss/elewise/elewise_tiling.h"

using Ops::NN::Optiling::TilingBaseClass;
using namespace Ops::Base;

namespace optiling {
struct ReluGradV2CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};
class ReluGradV2Tiling {
public:
    explicit ReluGradV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();

private:
    uint64_t dType = 0;
    uint64_t schMode = 0;
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_RELU_GRAD_V2_TILING_H