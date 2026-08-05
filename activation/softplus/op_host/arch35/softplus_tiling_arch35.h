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
 * \file softplus_tiling_arch35.h
 * \brief
 */
#ifndef _ACTIVATION_HOST_SOFTPLUS_TILING_H_
#define _ACTIVATION_HOST_SOFTPLUS_TILING_H_
#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"

namespace optiling {
using namespace Ops::Base;

struct SoftplusCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class SoftplusTiling {
public:
    explicit SoftplusTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckShape();

private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    uint64_t dType = 0;
};

} // namespace optiling
#endif // _ACTIVATION_HOST_SOFTPLUS_TILING_H_
