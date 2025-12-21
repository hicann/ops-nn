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

    virtual void GetFormat();

    virtual ge::graphStatus GetShape();

    virtual ge::graphStatus BaseOpSpecificCheck();

    virtual ge::graphStatus CheckSelfSlice(int64_t (&dims)[2]);

    virtual ge::graphStatus CheckMat2Transpose(int64_t (&dims)[2]);

    virtual bool CheckIsNonContiguous(int64_t (&mkDims)[2], int64_t (&knDims)[2]);
protected:
    gert::TilingContext *context_ = nullptr;
    MatMulV3Args args_;
    bool isSelfSlice_ = false;
};
}
}
#endif // __OP_HOST_MATMUL_V3_ADVANCED_TILING_H__
