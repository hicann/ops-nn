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
 * \file batch_matmul_v3_asw_basic_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_basic_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/log_format_util.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
using StrideIndexPairs = std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>>;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswBasicTiling, DAV_3510, ASW_BASIC);
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswBasicTiling, DAV_RESV, ASW_BASIC); // supportMmadS8S4平台

bool BatchMatMulV3AswBasicTiling::IsCapable()
{
    bool isEqualBatch = batchInfo_->batchA0 == batchInfo_->batchB0 && batchInfo_->batchA1 == batchInfo_->batchB1 &&
                        batchInfo_->batchA2 == batchInfo_->batchB2 && batchInfo_->batchA3 == batchInfo_->batchB3;
    if (!isEqualBatch) {
        return false;
    }
    if (batchInfo_->batchBias > 1UL) {
        return false;
    }
    bool isSupportType = (args_.aType == ge::DT_FLOAT16 || args_.aType == ge::DT_BF16) &&
                         (args_.bType == ge::DT_FLOAT16 || args_.bType == ge::DT_BF16) &&
                         (args_.cType == ge::DT_FLOAT16 || args_.cType == ge::DT_BF16 || args_.cType == ge::DT_FLOAT);
    if ((!isSupportType) && args_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT(
            args_.opName, "mat2", Ops::Base::ToString(args_.bFormat).c_str(),
            Ops::NN::FormatString("%s when the dtype of %s is %s", "ND", "input", "FP32").c_str());
        return false;
    }
    return true;
}

ge::graphStatus BatchMatMulV3AswBasicTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);

    MatMulV3TilingHelper::GetRebalanceBlock(compileInfo_, args_, runInfo_, context_);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);

    // l1开2db后依然只使用了一半的空间，则开启4 db。该字段仅在基础api场景生效
    uint64_t abL1TensorSize = runInfo_.baseK * runInfo_.stepKa * (runInfo_.baseM + runInfo_.baseN) * args_.aDtypeSize;
    if (args_.hasBias) {
        abL1TensorSize += runInfo_.baseN * sizeof(args_.biasType);
    }
    if (abL1TensorSize * NUM_FOUR <= compileInfo_.l1Size) {
        runInfo_.l1BufferNum = NUM_FOUR;
    } else {
        runInfo_.l1BufferNum = NUM_TWO;
    }

    // 特殊处理3D非连续场景
    if (context_->InputIsView(0) && MatMulV3TilingHelper::IsTransposeNonContiguous(context_, 0) &&
        context_->InputIsView(1) && MatMulV3TilingHelper::IsTransposeNonContiguous(context_, 1)) {
        runInfo_.innerBatch = batchInfo_->batchC;
    }
    // 确认是否切换tensor api
    CheckTensorApiSupport();
    return ge::GRAPH_SUCCESS;
}

void BatchMatMulV3AswBasicTiling::CheckTensorApiSupport()
{
    bool isBatchMatmul = strcmp(context_->GetNodeType(), "BatchMatMulV3") == 0;
    bool isNonContiguous = context_->InputIsView(0) && MatMulV3TilingHelper::IsTransposeNonContiguous(context_, 0) &&
                           context_->InputIsView(1) && MatMulV3TilingHelper::IsTransposeNonContiguous(context_, 1);
    // FP32切K判断
    bool isFp32 = (args_.aType == ge::DT_FLOAT && args_.bType == ge::DT_FLOAT);
    bool isNdFormat = (args_.aFormat == ge::FORMAT_ND && args_.bFormat == ge::FORMAT_ND);
    uint64_t fp32SplitKThreshold =
        args_.kValue > FP32_K_SWITCH_THRESHOLD ? FP32_SPLIT_K_THRESHOLD2 : FP32_SPLIT_K_THRESHOLD1;
    bool isSplitK = false;
    // 连续且非全载场景才支持切K
    if (!isNonContiguous && isFp32 && !args_.isHf32 && isNdFormat && args_.kValue > fp32SplitKThreshold &&
        fullLoad_ == MatMulV3FullLoad::NONE_FULL_LOAD) {
        isSplitK = true;
    }
    // 非切K且连续场景下才允许切换tensor api实现
    apiLevel_ = (isBatchMatmul && !isNonContiguous && !isSplitK) ? MatMulV3ApiLevel::TENSOR_LEVEL_BASIC :
                                                                   MatMulV3ApiLevel::BASIC_LEVEL;
    // 1952当前只支持基础API
    if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        apiLevel_ = MatMulV3ApiLevel::BASIC_LEVEL;
    }
}

uint64_t BatchMatMulV3AswBasicTiling::GetTilingKey() const
{
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::BASIC)
        .SetApiLevel(apiLevel_)
        .GetTilingKey();
}

ge::graphStatus BatchMatMulV3AswBasicTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3BasicTilingData>(tiling);
}

uint64_t BatchMatMulV3AswBasicTiling::GetNumBlocks() const
{
    return compileInfo_.aicNum;
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling