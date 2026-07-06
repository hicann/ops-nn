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
 * \file elu_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_TILING_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/arch35/elu_tiling_struct.h"
#include "../../op_kernel/arch35/elu_struct.h"

namespace optiling {
using namespace EluNs;

struct EluCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class EluTilingArch35 {
public:
    explicit EluTilingArch35(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunTiling();
    EluTilingData* tiling = nullptr;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus SetTilingData();

private:
    ge::graphStatus CheckShape();

    gert::TilingContext* tilingContext;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    uint64_t schMode = 0;
    uint64_t dType = 0;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_TILING_H
