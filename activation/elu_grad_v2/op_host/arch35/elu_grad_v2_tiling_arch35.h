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
 * \file elu_grad_v2_tiling_arch35.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_GRAD_V2_TILING_ARCH35_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_GRAD_V2_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "../../op_kernel/elu_grad_v2_tiling_struct.h"
#include <exe_graph/runtime/tiling_context.h>

using namespace Ops::Base;
using namespace EluGradV2Ns;
namespace optiling {
struct EluGradV2CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class EluGradV2Tiling {
public:
    explicit EluGradV2Tiling(gert::TilingContext *context) : tilingContext(context){};
    ge::graphStatus RunTiling();
    EluGradV2TilingData *tiling;

protected:
    ge::graphStatus SetTilingData(bool is_result);
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetAttr();
private:
    uint64_t dType = 0;
    uint64_t schMode = 0;
    bool isResult = false;
    ge::DataType gradsDtype;
    ge::DataType activationsDtype;
    ge::DataType outputDtype;
    gert::TilingContext *tilingContext;
};
} // namespace optiling
 
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_GRAD_V2_TILING_ARCH35_H