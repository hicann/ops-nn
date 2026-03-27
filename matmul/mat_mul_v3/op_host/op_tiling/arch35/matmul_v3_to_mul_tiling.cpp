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
 * \file matmul_v3_to_mul_tiling.cc
 * \brief
 */

#include "matmul_v3_to_mul_tiling.h"
#include "matmul_tiling_registry.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3ToMulTiling, DAV_3510, TO_MUL);

bool MatMulV3ToMulTiling::IsCapable()
{
    // 未使能高精度模式
    if (!args_.isForceGrpAccForFp32) {
        return false;
    }
    // m=1 || n=1
    if (args_.mValue != 1UL && args_.nValue != 1UL) {
        return false;
    }
    if (args_.aDtypeSize != sizeof(float) || args_.bDtypeSize != sizeof(float)) {
        return false;
    }
    if (args_.kValue < LIMIT_K) {
        return false;
    }
    return true;
}

ge::graphStatus MatMulV3ToMulTiling::DoOpTiling()
{
    uint64_t m = args_.mValue;
    uint64_t n = args_.nValue;
    uint64_t k = args_.kValue;
    uint64_t shapeMN = n * m;
    uint64_t ubCount = compileInfo_.ubSize / sizeof(float);
    uint64_t biasCount = 0;
    // 根据内外轴设置baseMN baseK
    bool dataCopyMode = false;
    if ((m == 1 && !args_.isBTrans) || (n == 1 && args_.isATrans)) {
        dataCopyMode = true;
    }
    uint64_t baseMN = BASE_MN;
    uint64_t baseK = BASE_K;
    uint64_t tmpMN = ops::CeilDiv(shapeMN, compileInfo_.aivNum);
    // (k, n) || (k, m)
    if (dataCopyMode) {
        baseMN = std::min(ops::CeilAlign(tmpMN, ALIGN_NUM), BASE_MN);
        biasCount = args_.hasBias ? baseMN : 0;
        baseK = (ubCount - baseMN - biasCount) / NUM_FIVE / baseMN;
    } else { // (m, k) || (n, k)
        biasCount = args_.hasBias ? tmpMN : 0;
        uint64_t tmpK = ubCount <= (tmpMN + biasCount) ? 0 : (ubCount - tmpMN - biasCount) / NUM_FIVE / tmpMN;
        baseK = std::max(ops::FloorAlign(tmpK, ALIGN_NUM), BASE_K);
        biasCount = args_.hasBias ? 1 : 0;
        baseMN = std::min(tmpMN, ubCount / (NUM_FIVE * baseK + biasCount + 1));
    }
    // 尾核需要处理的MN方向个数
    uint64_t tailMN = shapeMN % baseMN;
    // 每个核K方向尾次处理的个数
    uint64_t tailK = k % baseK;
    uint64_t loopK = ops::CeilDiv(k, baseK);
    // 计算实际核数
    uint64_t tileNum = ops::CeilDiv(shapeMN, baseMN);
    uint64_t useCoreNum = tileNum >= compileInfo_.aivNum ? compileInfo_.aivNum : tileNum;

    runInfo_.mmToMulInfo.baseMN = baseMN;
    runInfo_.mmToMulInfo.tailMN = tailMN;
    runInfo_.mmToMulInfo.baseK = baseK;
    runInfo_.mmToMulInfo.tailK = tailK;
    runInfo_.mmToMulInfo.loopK = loopK;
    runInfo_.mmToMulInfo.tileNum = tileNum;
    runInfo_.usedCoreNum = useCoreNum;
    runInfo_.mmToMulInfo.dataCopyMode = dataCopyMode;
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3ToMulTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .SetBatchModel(MatMulV3BatchModel::BATCH_MODEL)
        .SetModel(MatMulV3Model::TO_MUL)
        .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
        .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
        .GetTilingKey();
}

ge::graphStatus MatMulV3ToMulTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulToMulBasicTilingData>(tiling);
}
}
}