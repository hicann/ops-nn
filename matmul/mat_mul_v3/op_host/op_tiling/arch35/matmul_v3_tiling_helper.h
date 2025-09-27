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
 * \file matmul_v3_tiling_helper.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_TILING_HELPER_H__
#define __OP_HOST_MATMUL_V3_TILING_HELPER_H__

#include "matmul_v3_common_advanced.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_key.h"

namespace optiling {
namespace matmul_v3_advanced {
class MatMulV3TilingHelper {
public:
    static void ResetBase(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo);
    static void CalL1Tiling(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args, MatMulV3RunInfo &runInfo);
    static MatMulV3L0C2Out GetL0C2Out(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
        const MatMulV3RunInfo &runInfo);
    static bool CheckIfDoubleAswt(const MatmulV3CompileInfo &compileInfo, const MatMulV3Args &args,
                                  const uint64_t batchC);
};
}
}

#endif // __OP_HOST_MATMUL_V3_TILING_HELPER_H__
