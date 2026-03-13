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
 * \file matmul_v3_mn_equal_one_tiling.cc
 * \brief
 */

#include "matmul_v3_mn_equal_one_tiling.h"
#include "matmul_tiling_registry.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3MNEqOneTiling, DAV_3510, MN_EQUAL_ONE);

bool MatMulV3MNEqOneTiling::IsCapable()
{
    // m=1 || n=1
    if (args_.mValue != 1UL && args_.nValue != 1UL) {
        return false;
    }
    if (args_.aDtypeSize != sizeof(float) || args_.bDtypeSize != sizeof(float)) {
        return false;
    }
    if (args_.mValue == 1UL && (args_.nValue < BASE_K || args_.nValue % BASE_MN != 0)) {
        return false;
    } 
    if (args_.nValue == 1UL && (args_.mValue < BASE_K || args_.mValue % BASE_MN != 0)) {
        return false;
    }
    if (args_.kValue % BASE_K != 0) {
        return false;
    }
    return true;
}

ge::graphStatus MatMulV3MNEqOneTiling::DoOpTiling()
{
    // 仅考虑K循环 N根据BASE_MN分多核 不再切分
    uint64_t m = args_.mValue;
    uint64_t n = args_.nValue;
    uint64_t k = args_.kValue;
    // 按照BASE_MN=32切分需要使用的总核数
    uint64_t useAllCoreNum = m == 1 ? ops::CeilDiv(n, BASE_MN) : ops::CeilDiv(m, BASE_MN);
    // 尾核需要处理的MN方向个数
    uint64_t tailMN =  m == 1 ? n % BASE_MN : m % BASE_MN;
    uint64_t useCoreNum = useAllCoreNum >= compileInfo_.aivNum ? compileInfo_.aivNum : useAllCoreNum;
    uint64_t loopK = k / BASE_K;
    runInfo_.oneInfo.tailMN = tailMN;
    runInfo_.oneInfo.loopK = loopK;
    runInfo_.oneInfo.useAllCoreNum = useAllCoreNum;
    runInfo_.usedCoreNum = useCoreNum;
    return ge::GRAPH_SUCCESS;
}


uint64_t MatMulV3MNEqOneTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
        .SetModel(MatMulV3Model::MN_EQUAL_ONE)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .GetTilingKey();
}

ge::graphStatus MatMulV3MNEqOneTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3MNEqOneBasicTilingData>(tiling);
}
}
}