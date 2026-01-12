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
 * \file swish_grad_regbase_optiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_SWISHGRAD_REGBASE_OPTILING
#define OPS_BUILD_IN_OP_TILING_RUNTIME_SWISHGRAD_REGBASE_OPTILING

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "activation/swish_grad/op_kernel/arch35/swish_grad_tilingdata.h"

namespace optiling {
class SwishGradTiling {
public:
    explicit SwishGradTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunTiling();
    SwishGradTilingData* tiling = nullptr;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetTilingData();
    ge::graphStatus SetAttr();

private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
    ge::DataType inputDtype1;
    ge::DataType inputDtype2;
    uint64_t dType = 0;
    uint64_t schMode = 0;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_SWISHGRAD_REGBASE_OPTILING