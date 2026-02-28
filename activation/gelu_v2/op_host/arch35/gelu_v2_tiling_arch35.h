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
 * \file gelu_v2_tiling_arch35.h
 * \brief
 */
#ifndef ACTIVATION_GELU_V2_OP_HOST_GELU_V2_TILING_ARCH35_H
#define ACTIVATION_GELU_V2_OP_HOST_GELU_V2_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "op_common/atvoss/elewise/elewise_tiling.h"
#include "op_host/tiling_base.h"

namespace optiling {
using namespace Ops::Base;

class GeluV2Tiling
{
public:
    explicit GeluV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();
    ge::graphStatus CheckValid();
    const gert::Shape& EnsureNotScalar(const gert::Shape& inShape);

private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype;
    ge::DataType inputDtype;
    uint64_t dType = 0;
    uint64_t approximate = 0;
    std::string approximateStr = "";
};

struct GeluV2CompileInfoArch35 {
    uint32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

}  // namespace optiling
#endif  // ACTIVATION_GELU_V2_OP_HOST_GELU_V2_TILING_ARCH35_H
