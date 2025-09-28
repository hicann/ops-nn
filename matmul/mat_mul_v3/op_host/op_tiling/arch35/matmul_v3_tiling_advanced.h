/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_tiling_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__
#define __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__

#include "runtime/tiling_context.h"
#include "matmul_v3_common_advanced.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3Tiling {
public:
    explicit MatMulV3Tiling(gert::TilingContext *context) : context_(context){};
    virtual ~MatMulV3Tiling() = default;
    virtual ge::graphStatus DoTiling();

protected:
    virtual ge::graphStatus GetShapeAttrsInfo();

    virtual ge::graphStatus GetArgs();

    virtual ge::graphStatus CheckArgs();

protected:
    gert::TilingContext *context_ = nullptr;
    MatMulV3Args args_;
};
}
}
#endif // __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__
