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
 * \file elu_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_TILING_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/arch35/elu_tiling_struct.h"

using namespace EluNs;
namespace optiling {

struct EluCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class EluTiling {
public:
    explicit EluTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
    EluTilingData* tiling;

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus SetTilingData();

private:
    ge::graphStatus CheckShape();

    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ELU_TILING_H