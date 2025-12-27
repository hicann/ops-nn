/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_GELU_REGBASE_OPTILING
#define OPS_BUILD_IN_OP_TILING_RUNTIME_GELU_REGBASE_OPTILING

#include "register/tilingdata_base.h"
#include "op_common/atvoss/elewise/elewise_tiling.h"

namespace optiling {

struct GeluCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class GeluTiling
{
public:
    explicit GeluTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();
protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
    uint64_t dType = 0;
};
}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_GELU_REGBASE_OPTILING