
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
 * \file log_sigmoid_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_LOG_SIGMOID_TILING_ARCH35_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_LOG_SIGMOID_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "tiling_base/tiling_base.h"

using namespace Ops::Base;
namespace optiling {
class LogSigmoidTiling
{
public:
    explicit LogSigmoidTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CheckDtype();
    ge::graphStatus CheckShape();

private:
    uint64_t dType = 0;
    ge::DataType outputDtype;
    gert::TilingContext* tilingContext;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_LOG_SIGMOID_TILING_H
