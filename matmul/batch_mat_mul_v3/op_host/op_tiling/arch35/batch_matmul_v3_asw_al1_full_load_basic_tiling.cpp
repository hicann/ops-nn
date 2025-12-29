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
 * \file batch_matmul_v3_asw_al1_full_load_basic_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_al1_full_load_basic_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswAL1FullLoadBasicTiling, ASCEND910_95, AL1_FULL_LOAD_BASIC);

bool BatchMatMulV3AswAL1FullLoadBasicTiling::IsCapable()
{
    if (batchInfo_->batchA > 1UL) { // matrix A should not have batch when AL1FullLoad
        return false;
    }
    if (args_.mValue > BASIC_BLOCK_SIZE_256) { // m should be less then 256
        return false;
    }
    // get align m,k,n value
    uint64_t alignMValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignKaValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignKbValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignNValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignMatASize = batchInfo_->batchA * alignMValue * alignKaValue * args_.aDtypeSize;
    uint64_t alignMatBSize = batchInfo_->batchB * alignKbValue * alignNValue * args_.bDtypeSize;
    // each core needs to loop at least 4 batch for MatB
    // 总数据量大于2轮 batch大于4轮
    if (alignMatBSize < compileInfo_.l1Size * compileInfo_.aicNum &&
        batchInfo_->batchB * ops::CeilDiv(args_.nValue, BASIC_BLOCK_SIZE_256) < 4UL * compileInfo_.aicNum) {
        return false;
    }
    if (alignMatASize * NUM_TWO > compileInfo_.l1Size) {
        return false;
    }
    return true;
}

ge::graphStatus BatchMatMulV3AswAL1FullLoadBasicTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    // 复用基础API A全载
    MatMulV3BasicAswtTiling::DoAL1FullLoad(args_.batchInfo->batchB, args_.batchInfo->batchBias);

    // l1开2db后依然只使用了一半的空间，则开启4 db。该字段仅在基础api场景生效
    uint64_t alignMValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignKaValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t aL1TensorSize = alignKaValue * alignMValue * args_.aDtypeSize;                        // a全载数据量
    uint64_t bL1TensorSize = runInfo_.baseN * runInfo_.baseK * runInfo_.stepKb * args_.bDtypeSize; // stepN即buffer数
    uint64_t dtypeSize = GetSizeByDataType(ge::DT_FLOAT);
    // 是否开启dbLoc
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * dtypeSize * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    // A全载更新stepM
    runInfo_.stepM = ops::CeilAlign(args_.mValue, runInfo_.baseM);
    // A全载对matB开启4buffer
    if (bL1TensorSize * NUM_FOUR + aL1TensorSize <= compileInfo_.l1Size) {
        runInfo_.l1BufferNum = NUM_FOUR;
    } else {
        runInfo_.l1BufferNum = NUM_TWO;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchMatMulV3AswAL1FullLoadBasicTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::BASIC)
        .SetFullLoad(MatMulV3FullLoad::A_FULL_LOAD)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL) // 赋值apiLevel为basic
        .GetTilingKey();
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling