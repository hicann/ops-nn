/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swish_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SWISH_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SWISH_H_

#include "op_host/tiling_base.h"

using namespace Ops::NN::Optiling;
namespace optiling {

class SwishTiling
{
public:
    explicit SwishTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus SetAttr();

private:
    uint64_t schMode = 0;
    uint64_t attrWork = 0;
    float attrScale = 1;
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
};

struct SwishCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};
} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_SWISH_H_

