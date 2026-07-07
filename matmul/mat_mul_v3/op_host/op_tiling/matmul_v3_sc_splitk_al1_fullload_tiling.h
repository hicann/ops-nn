/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_v3_sc_splitk_al1_fullload_tiling.h
 * \brief Single-core split-K + A L1 full load tiling helpers (tilingKey 65569).
 */
#ifndef __OP_HOST_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_TILING_H__
#define __OP_HOST_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_TILING_H__

#include "matmul_v3_tiling.h"
#include "matmul_v3_common.h"
#include "matmul_v3_compile_info.h"

namespace optiling {
namespace matmul_v3 {

class MatmulV3ScSplitKAl1FullLoadTiling {
public:
    static bool DoTiling(const char *opName, const MatmulV3Args &args, const MatmulV3CompileInfo &compileInfo,
        uint64_t aDtypeSize, uint64_t bDtypeSize, MatmulV3RunInfo &runInfo);

    static bool CheckTilingOk(const char *opName, const MatmulV3RunInfo &runInfo,
        const MatmulV3CompileInfo &compileInfo, uint64_t aDtypeSize, uint64_t bDtypeSize);
};

} // namespace matmul_v3
} // namespace optiling

#endif // __OP_HOST_MATMUL_V3_SC_SPLITK_AL1_FULLLOAD_TILING_H__
