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
 * \file apply_gradient_descent_tiling_arch35.h
 * \brief apply_gradient_descent_tiling
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_GRADIENT_DESCENT_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_GRADIENT_DESCENT_TILING_H

#include "register/tilingdata_base.h"
#include "op_common/atvoss/elewise/elewise_tiling.h"
#include "op_common/atvoss/elewise/elewise_base_struct.h"
#include "../../op_kernel/arch35/apply_gradient_descent_tiling_struct.h"

namespace optiling {

struct ApplyGradientDescentCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class ApplyGradientDescentTiling
{
public:
    explicit ApplyGradientDescentTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    ApplyGradientDescentNs::ApplyGradientDescentTilingData* tiling;

protected:
    ge::graphStatus SetTilingData();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CheckShape();

private:
    gert::TilingContext* tilingContext;
    uint64_t dType_ = 0;
    uint64_t schMode_ = 0;
    ge::DataType outputDtype;
    ge::DataType varDtype;
    ge::DataType alphaDtype;
    ge::DataType deltaDtype;
};
}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_APPLY_GRADIENT_DESCENT_TILING_H