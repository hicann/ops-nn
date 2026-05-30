/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/* !
 * \file matmul_v3_to_multi_mul_tiling.cc
 * \brief
 */

#include "matmul_v3_to_multi_mul_tiling.h"
#include "matmul_tiling_registry.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3ToVectorTiling, DAV_3510, TO_MULTI_MUL);

bool MatMulV3ToVectorTiling::IsCapable()
{
    // 未使能高精度模式
    if (!args_.isForceGrpAccForFp32) {
        return false;
    }
    // !ATrans && BTrans
    if (args_.isATrans || !args_.isBTrans) {
        return false;
    }
    // 暂时不支持bias
    if (args_.hasBias) {
        return false;
    }
    if (args_.aDtypeSize != sizeof(float) || args_.bDtypeSize != sizeof(float)) {
        return false;
    }
    return true;
}

void MatMulV3ToVectorTiling::CalcBasicBlock()
{
    uint64_t mCore = ops::CeilDiv(args_.mValue, runInfo_.baseM);
    uint64_t nCore = ops::CeilDiv(args_.nValue, runInfo_.baseN);
    if (mCore == 0UL || nCore == 0UL) {
        return;
    }
    if (mCore <= nCore) {
        runInfo_.baseM = ops::CeilDiv(args_.mValue, mCore);
        mCore = ops::CeilDiv(args_.mValue, runInfo_.baseM);
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseN = ops::CeilDiv(args_.nValue, nCore);
    } else {
        runInfo_.baseN = ops::CeilDiv(args_.nValue, nCore);
        nCore = ops::CeilDiv(args_.nValue, runInfo_.baseN);
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilDiv(args_.mValue, mCore);
    }

    while (runInfo_.baseN >= runInfo_.baseM * NUM_TWO && nCore > 0UL && nCore < runInfo_.usedCoreNum / NUM_TWO) {
        nCore = nCore * NUM_TWO;
        mCore = runInfo_.usedCoreNum / nCore;
        runInfo_.baseM = ops::CeilDiv(args_.mValue, mCore);
        runInfo_.baseN = ops::CeilDiv(args_.nValue, nCore);
        mCore = ops::CeilDiv(args_.mValue, static_cast<uint64_t>(runInfo_.baseM));
        nCore = ops::CeilDiv(args_.nValue, static_cast<uint64_t>(runInfo_.baseN));
    }

    while (runInfo_.baseM >= runInfo_.baseN * NUM_TWO && mCore > 0UL && mCore < runInfo_.usedCoreNum / NUM_TWO) {
        mCore = mCore * NUM_TWO;
        nCore = runInfo_.usedCoreNum / mCore;
        runInfo_.baseM = ops::CeilDiv(args_.mValue, mCore);
        runInfo_.baseN = ops::CeilDiv(args_.nValue, nCore);
        mCore = ops::CeilDiv(args_.mValue, static_cast<uint64_t>(runInfo_.baseM));
        nCore = ops::CeilDiv(args_.nValue, static_cast<uint64_t>(runInfo_.baseN));
    }
}

ge::graphStatus MatMulV3ToVectorTiling::DoOpTiling()
{
    uint64_t m = args_.mValue;
    uint64_t n = args_.nValue;
    uint64_t k = args_.kValue;
    runInfo_.baseM = BASE;
    runInfo_.baseN = BASE;
    runInfo_.baseK = BASE;
    runInfo_.usedCoreNum = compileInfo_.aivNum;
    uint64_t mCore = ops::CeilDiv(m, runInfo_.baseM);
    uint64_t nCore = ops::CeilDiv(n, runInfo_.baseN);
    if (mCore * nCore < compileInfo_.aivNum) {
        CalcBasicBlock();
    }
    mCore = ops::CeilDiv(m, runInfo_.baseM);
    nCore = ops::CeilDiv(n, runInfo_.baseN);
    runInfo_.usedCoreNum = mCore * nCore;
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3ToVectorTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
        .SetModel(MatMulV3Model::TO_MULTI_MUL)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .GetTilingKey();
}

ge::graphStatus MatMulV3ToVectorTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulToVectorBasicTilingData>(tiling);
}
}
}