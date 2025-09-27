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
 * \file batch_matmul_v3_tiling_advanced.cc
 * \brief
 */

#include "batch_matmul_v3_tiling_advanced.h"

#include "register/op_def_registry.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"

#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"

namespace optiling {
namespace batch_matmul_v3_advanced {

ge::graphStatus BatchMatMulV3Tiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MatMulV3BatchInfo tempBatchInfo;
    OP_TILING_CHECK((GetBatchInfo(*context_, args_, tempBatchInfo) != ge::GRAPH_SUCCESS),
       CUBE_INNER_ERR_REPORT(args_.opName, "GetBatchInfo failed"),
       return ge::GRAPH_FAILED);
    args_.batchInfo = &tempBatchInfo;
    MatMulTilingCfg tilingCfg(false, context_->GetCompileInfo(), static_cast<void *>(&args_));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    platform_ascendc::SocVersion socVersion =
        static_cast<const MatmulV3CompileInfo *>(tilingCfg.compileInfo)->socVersion;
    MMRegisterCfg registerCfg{ "BatchMatMulV3", socVersion, strategy::GetBatchMatMulV3Priorities(socVersion) };
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}

ge::graphStatus BatchMatMulV3Tiling::GetBmmBiasInfo(const gert::TilingContext &context, MatMulV3Args& args,
                                                    MatMulV3BatchInfo& batchInfo)
{
    if (!args_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasShape = context.GetInputShape(2)->GetOriginShape();
    auto outputShape = context.GetOutputShape(0)->GetOriginShape();
    size_t biasDims = biasShape.GetDimNum();
    size_t cDims = outputShape.GetDimNum();
    uint64_t batchBias3 = 1UL;
    uint64_t batchBias2 = 1UL;
    uint64_t batchBias1 = 1UL;
    uint64_t batchBias0 = 1UL;
    // 先校验bias的尾值是否与output尾值相等
    if (biasShape[biasDims - FINAL_SHAPE_DIM] != outputShape[cDims - FINAL_SHAPE_DIM]) {
        OP_LOGE(args.opName, "Last dim of bias is not equal to last dim of output.");
        return ge::GRAPH_FAILED;
    }
    if (biasDims >= NUM_TWO) {
        if (biasShape[biasDims - NO_BATCH_SHAPE_DIM] != 1) { // bias的倒数第二维必须为1
            OP_LOGE(args.opName, "M of bias must be 1.");
            return ge::GRAPH_FAILED;
        }
    }
    // 若为batchbias，继续做后续校验
    if (biasDims > NUM_TWO) {
        if (batchInfo.batchA0 != batchInfo.batchB0 || batchInfo.batchA1 != batchInfo.batchB1 ||
            batchInfo.batchA2 != batchInfo.batchB2 || batchInfo.batchA3 != batchInfo.batchB3)  {
            OP_LOGE(args.opName, "BatchBias scene, the batch of A and B must be equal.");
            return ge::GRAPH_FAILED;
        }
        batchBias3 = biasDims > NO_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - ONE_BATCH_SHAPE_DIM) : 1UL;
        batchBias2 = biasDims > ONE_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - TWO_BATCH_SHAPE_DIM) : 1UL;
        batchBias1 = biasDims > TWO_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - THREE_BATCH_SHAPE_DIM) : 1UL;
        batchBias0 = biasDims > THREE_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
        if (!(batchBias3 == batchInfo.batchC3 && batchBias2 == batchInfo.batchC2 && batchBias1 == batchInfo.batchC1 &&
            batchBias0 == batchInfo.batchC0)) {
            OP_LOGE(args.opName, "The batch of bias must be equal to the batch of C.");
            return ge::GRAPH_FAILED;
        }
    }
    batchInfo.batchBias = batchBias3 * batchBias2 * batchBias1 * batchBias0;
    OP_LOGI(args.opName, "Check BatchMatMulV3 with bias success.");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchMatMulV3Tiling::GetBatchInfo(const gert::TilingContext &context, MatMulV3Args& args,
                                                  MatMulV3BatchInfo& batchInfo)
{
    auto aShape = context.GetInputShape(0)->GetOriginShape();
    auto bShape = context.GetInputShape(1)->GetOriginShape();
    auto cShape = context.GetOutputShape(0)->GetOriginShape();

    size_t aDims = aShape.GetDimNum();
    size_t bDims = bShape.GetDimNum();
    size_t cDims = cShape.GetDimNum();
    if (aDims > BATCH_DIM_MAX || bDims > BATCH_DIM_MAX) {
      OP_LOGE(args.opName,
              "The current input dimensions is greater than 6 where x1_dims is (%zu) and x2_dims is (%zu)",
              aDims, bDims);
      return ge::GRAPH_FAILED;
    }
    batchInfo.batchA3 = aDims > NO_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchA2 = aDims > ONE_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchA1 = aDims > TWO_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchA0 = aDims > THREE_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchB3 = bDims > NO_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchB2 = bDims > ONE_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchB1 = bDims > TWO_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchB0 = bDims > THREE_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchC3 = cDims > NO_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchC2 = cDims > ONE_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchC1 = cDims > TWO_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchC0 = cDims > THREE_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo.batchA = batchInfo.batchA0 * batchInfo.batchA1 * batchInfo.batchA2 * batchInfo.batchA3;
    batchInfo.batchB = batchInfo.batchB0 * batchInfo.batchB1 * batchInfo.batchB2 * batchInfo.batchB3;
    batchInfo.batchC = batchInfo.batchC0 * batchInfo.batchC1 * batchInfo.batchC2 * batchInfo.batchC3;

    //Check if one of the batch size is zero
    bool isBatchZero = (batchInfo.batchA == 0UL || batchInfo.batchB == 0UL);
    if (isBatchZero) {
      OP_LOGE(args.opName, "One of the batch size is zero");
      return ge::GRAPH_FAILED;
    }

    // when BatchB == 1, adjust M = batchA * M, batchA = 1
    MergeBatchAndMAxis(args, batchInfo); // check if batch merge to M

    // Check if batch info is valid, if batch is M broadcast to N, return failed.
    bool batch3Invalid = batchInfo.batchA3 != batchInfo.batchB3 && batchInfo.batchA3 != 1UL && batchInfo.batchB3 != 1UL;
    bool batch2Invalid = batchInfo.batchA2 != batchInfo.batchB2 && batchInfo.batchA2 != 1UL && batchInfo.batchB2 != 1UL;
    bool batch1Invalid = batchInfo.batchA1 != batchInfo.batchB1 && batchInfo.batchA1 != 1UL && batchInfo.batchB1 != 1UL;
    bool batch0Invalid = batchInfo.batchA0 != batchInfo.batchB0 && batchInfo.batchA0 != 1UL && batchInfo.batchB0 != 1UL;
    if (batch3Invalid || batch2Invalid || batch1Invalid || batch0Invalid) {
        OP_LOGE("[BatchMatMulV3]", "Is M broadcast to N situation, do not support!");
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((GetBmmBiasInfo(context, args, batchInfo) != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "GetBmmBiasInfo failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void BatchMatMulV3Tiling::MergeBatchAndMAxis(MatMulV3Args& args, MatMulV3BatchInfo& batchInfo)
{
    if (batchInfo.batchB != 1UL || args.isATrans){
        return;
    }
    OP_LOGD("[BatchMatMulV3]", "Merge Batch and M axis");
    // when BatchB == 1, adjust M = batchA * M, batchA = 1
    args.mValue = batchInfo.batchA * args.mValue;
    batchInfo.batchA3 = 1UL;
    batchInfo.batchA2 = 1UL;
    batchInfo.batchA1 = 1UL;
    batchInfo.batchA0 = 1UL;
    batchInfo.batchA = 1UL;
    batchInfo.batchC3 = 1UL;
    batchInfo.batchC2 = 1UL;
    batchInfo.batchC1 = 1UL;
    batchInfo.batchC0 = 1UL;
    batchInfo.batchC = 1UL;
    return;
}
}
}