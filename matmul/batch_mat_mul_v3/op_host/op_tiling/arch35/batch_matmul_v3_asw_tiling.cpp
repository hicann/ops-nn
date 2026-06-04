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
 * \file batch_matmul_v3_asw_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_asw_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "batch_matmul_v3_tiling_key.h"
#include "matmul/common/op_host/math_util.h"

namespace optiling {
namespace batch_matmul_v3_advanced {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswTiling, DAV_3510, BASE);
MM_REGISTER_TILING_TEMPLATE(BatchMatMulV3, BatchMatMulV3AswTiling, DAV_RESV, BASE); // supportMmadS8S4平台

ge::graphStatus BatchMatMulV3AswTiling::DoOpTiling()
{
    MatMulV3TilingHelper::ResetBase(compileInfo_, args_, runInfo_);
    MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);

    runInfo_.mixInfo.ubDB = runInfo_.baseM * runInfo_.baseN * DATA_SIZE_FP32 <= compileInfo_.ubSize ? DB_SIZE : 1UL;

    // 特殊处理3D非连续场景
    if (context_->InputIsView(0) && MatMulV3TilingHelper::IsTransposeNonContiguous(context_, 0) &&
        context_->InputIsView(1) && MatMulV3TilingHelper::IsTransposeNonContiguous(context_, 1)) {
        runInfo_.innerBatch = batchInfo_->batchC;
    }
    // 确认是否切换tensor api
    CheckTensorApiSupport();
    return ge::GRAPH_SUCCESS;
}

void BatchMatMulV3AswTiling::CheckTensorApiSupport()
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
    apiLevel_ = (isBatchMatmul && !isNonContiguous && !isSplitK) ? MatMulV3ApiLevel::TENSOR_LEVEL :
                                                                   MatMulV3ApiLevel::HIGH_LEVEL;
    // 1952当前只支持基础API
    if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        apiLevel_ = MatMulV3ApiLevel::HIGH_LEVEL;
    }
}

uint64_t BatchMatMulV3AswTiling::GetTilingKey() const
{
    if (tilingKeyObj != nullptr) {
        tilingKeyObj->SetTrans(args_.isATrans, args_.isBTrans);
        tilingKeyObj->SetModel(MatMulV3Model::BASIC);
        return tilingKeyObj->GetTilingKey();
    }
    return BatchMatMulV3TilingKey()
        .SetTrans(args_.isATrans, args_.isBTrans)
        .SetModel(MatMulV3Model::BASIC)
        .SetApiLevel(apiLevel_)
        .GetTilingKey();
}

uint64_t BatchMatMulV3AswTiling::GetNumBlocks() const { return compileInfo_.aicNum; }

ge::graphStatus BatchMatMulV3AswTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<BatchMatMulV3TilingData>(tiling);
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling