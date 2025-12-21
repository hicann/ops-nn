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
 * \file matmul_v3_asw_basic_tiling.cc
 * \brief
 */
#include "matmul_v3_asw_basic_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;
namespace optiling {
namespace matmul_v3_advanced {

constexpr uint64_t FP32_SPLIT_K_THRESHOLD = 8192UL;
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3AswBasicApiTiling, ASCEND910_95, BASIC_ASWT);

bool MatMulV3AswBasicApiTiling::IsCapable()
{
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, BASIC_BLOCK_SIZE_256);
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, BASIC_BLOCK_SIZE_256);
    if (mCore * nCore > compileInfo_.aicNum) {
        OP_LOGD(args_.opName, "mCnt_[%lu] and nCnt_[%lu] is not enter in matmulv3 basic api", mCore, nCore);
        return false;
    }
    if (args_.bFormat != ge::FORMAT_ND || args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for basic api");
        return false;
    }
    if (args_.aDtypeSize == DATA_SIZE_FP32 && !args_.isHf32 && args_.bFormat == ge::FORMAT_ND &&
        args_.kValue > FP32_SPLIT_K_THRESHOLD) {
        OP_LOGD(args_.opName, "fp32 big k is not supported for basic api");
        return false;
    }

    OP_LOGI(args_.opName, "MatMulV3 tiling enable state basic api");
    return true;
}

uint64_t MatMulV3AswBasicApiTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}
} // namespace matmul_v3_advanced
} // namespace optiling