/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_tiling_advanced.cpp
 * \brief BatchMatMulV3 tiling implementation.
 */

#include "batch_matmul_v3_tiling_advanced.h"

#include "register/op_def_registry.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"

#include "batch_matmul_v3_tiling_strategy.h"
#include "batch_matmul_v3_common_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/common/op_host/log_format_util.h"

namespace optiling {
namespace batch_matmul_v3_advanced {

// ====== Phase 7: ValidateBias (bias[-1]==c[-1] by base, bias[-2]==1 by BMM) ======
ge::graphStatus BatchMatMulV3Tiling::ValidateBias()
{
    if (MatMulV3Tiling::ValidateBias() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (!args_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasShape = context_->GetInputShape(2)->GetOriginShape();
    size_t biasDims = biasShape.GetDimNum();
    if (biasDims >= NUM_TWO && biasShape[biasDims - NO_BATCH_SHAPE_DIM] != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            args_.opName, "bias", Ops::Base::ToString(biasShape).c_str(),
            Ops::NN::FormatString("%s of %s must be equal to %d", "M-axis", "bias", 1).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 8-1: ExtractMatrixBatchInfo ======
ge::graphStatus BatchMatMulV3Tiling::ExtractMatrixBatchInfo()
{
    auto aShape = context_->GetInputShape(0)->GetOriginShape();
    auto bShape = context_->GetInputShape(1)->GetOriginShape();
    auto cShape = context_->GetOutputShape(0)->GetOriginShape();

    size_t aDims = aShape.GetDimNum();
    size_t bDims = bShape.GetDimNum();
    size_t cDims = cShape.GetDimNum();
    if (aDims > BATCH_DIM_MAX || bDims > BATCH_DIM_MAX) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            args_.opName, "self, mat2", Ops::NN::FormatString("%zu, %zu", aDims, bDims).c_str(),
            Ops::NN::FormatString("The shape dims of %s must be %s %llu", "self, mat2", "less than or equal to",
                                  BATCH_DIM_MAX)
                .c_str());
        return ge::GRAPH_FAILED;
    }
    batchInfo_.batchA3 = aDims > NO_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchA2 = aDims > ONE_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchA1 = aDims > TWO_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchA0 = aDims > THREE_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchB3 = bDims > NO_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchB2 = bDims > ONE_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchB1 = bDims > TWO_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchB0 = bDims > THREE_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchC3 = cDims > NO_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - ONE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchC2 = cDims > ONE_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - TWO_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchC1 = cDims > TWO_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - THREE_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchC0 = cDims > THREE_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
    batchInfo_.batchA = batchInfo_.batchA0 * batchInfo_.batchA1 * batchInfo_.batchA2 * batchInfo_.batchA3;
    batchInfo_.batchB = batchInfo_.batchB0 * batchInfo_.batchB1 * batchInfo_.batchB2 * batchInfo_.batchB3;
    batchInfo_.batchC = batchInfo_.batchC0 * batchInfo_.batchC1 * batchInfo_.batchC2 * batchInfo_.batchC3;
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 8-2: ValidateMatrixBatchInfo ======
ge::graphStatus BatchMatMulV3Tiling::ValidateMatrixBatchInfo()
{
    bool isBatchZero = (batchInfo_.batchA == 0UL || batchInfo_.batchB == 0UL);
    if (isBatchZero) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "x1, x2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(context_->GetInputShape(0)->GetOriginShape()).c_str(),
                                  Ops::Base::ToString(context_->GetInputShape(1)->GetOriginShape()).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be a positive number", "Batch-axis", "x1, x2").c_str());
        return ge::GRAPH_FAILED;
    }

    MergeBatchAndMAxis(batchInfo_);

    bool batch3Invalid = batchInfo_.batchA3 != batchInfo_.batchB3 && batchInfo_.batchA3 != 1UL &&
                         batchInfo_.batchB3 != 1UL;
    bool batch2Invalid = batchInfo_.batchA2 != batchInfo_.batchB2 && batchInfo_.batchA2 != 1UL &&
                         batchInfo_.batchB2 != 1UL;
    bool batch1Invalid = batchInfo_.batchA1 != batchInfo_.batchB1 && batchInfo_.batchA1 != 1UL &&
                         batchInfo_.batchB1 != 1UL;
    bool batch0Invalid = batchInfo_.batchA0 != batchInfo_.batchB0 && batchInfo_.batchA0 != 1UL &&
                         batchInfo_.batchB0 != 1UL;
    if (batch3Invalid || batch2Invalid || batch1Invalid || batch0Invalid) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "self, mat2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(context_->GetInputShape(0)->GetOriginShape()).c_str(),
                                  Ops::Base::ToString(context_->GetInputShape(1)->GetOriginShape()).c_str())
                .c_str(),
            Ops::NN::FormatString(
                "The batch-axis of %s must meet the broadcast principle: The batch-axis in the corresponding "
                "positions must be equal, or one of the batch-axis in the corresponding positions must be 1",
                "self, mat2")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 8-3: ExtractOptionalBatchInfo (bias batch extraction) ======
ge::graphStatus BatchMatMulV3Tiling::ExtractOptionalBatchInfo()
{
    if (args_.hasBias) {
        auto biasShape = context_->GetInputShape(2)->GetOriginShape();
        size_t biasDims = biasShape.GetDimNum();
        uint64_t batchBias3 = 1UL;
        uint64_t batchBias2 = 1UL;
        uint64_t batchBias1 = 1UL;
        uint64_t batchBias0 = 1UL;
        if (biasDims > NUM_TWO) {
            batchBias3 = biasDims > NO_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - ONE_BATCH_SHAPE_DIM) : 1UL;
            batchBias2 = biasDims > ONE_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - TWO_BATCH_SHAPE_DIM) : 1UL;
            batchBias1 = biasDims > TWO_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - THREE_BATCH_SHAPE_DIM) : 1UL;
            batchBias0 = biasDims > THREE_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - FOUR_BATCH_SHAPE_DIM) : 1UL;
        }
        batchInfo_.batchBias = batchBias3 * batchBias2 * batchBias1 * batchBias0;
    }
    args_.batchInfo = &batchInfo_;
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 9: ValidateOptionalBatchInfo (bias batch-axis validation) ======
ge::graphStatus BatchMatMulV3Tiling::ValidateOptionalBatchInfo()
{
    if (!args_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    auto biasShape = context_->GetInputShape(2)->GetOriginShape();
    size_t biasDims = biasShape.GetDimNum();
    if (biasDims > NUM_TWO) {
        if (batchInfo_.batchA0 != batchInfo_.batchB0 || batchInfo_.batchA1 != batchInfo_.batchB1 ||
            batchInfo_.batchA2 != batchInfo_.batchB2 || batchInfo_.batchA3 != batchInfo_.batchB3) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                args_.opName, "self, mat2",
                Ops::NN::FormatString("%s, %s",
                                      Ops::Base::ToString(context_->GetInputShape(0)->GetOriginShape()).c_str(),
                                      Ops::Base::ToString(context_->GetInputShape(1)->GetOriginShape()).c_str())
                    .c_str(),
                Ops::NN::FormatString("When optional parameter %s exists, %s of %s must be the same", "bias",
                                      "batch-axis", "self, mat2")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        auto outputShape = context_->GetOutputShape(0)->GetOriginShape();
        uint64_t batchBias3 = biasDims > NO_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - ONE_BATCH_SHAPE_DIM) : 1UL;
        uint64_t batchBias2 = biasDims > ONE_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - TWO_BATCH_SHAPE_DIM) : 1UL;
        uint64_t batchBias1 = biasDims > TWO_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - THREE_BATCH_SHAPE_DIM) : 1UL;
        uint64_t batchBias0 = biasDims > THREE_BATCH_SHAPE_DIM ? biasShape.GetDim(biasDims - FOUR_BATCH_SHAPE_DIM) :
                                                                 1UL;
        if (!(batchBias3 == batchInfo_.batchC3 && batchBias2 == batchInfo_.batchC2 &&
              batchBias1 == batchInfo_.batchC1 && batchBias0 == batchInfo_.batchC0)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                args_.opName, "bias, out",
                Ops::NN::FormatString("%s, %s", Ops::Base::ToString(biasShape).c_str(),
                                      Ops::Base::ToString(outputShape).c_str())
                    .c_str(),
                Ops::NN::FormatString("%s of %s must be equal to %s of %s", "Batch-axis", "bias", "batch-axis", "out")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 10: Registry priorities ======
std::vector<int32_t> BatchMatMulV3Tiling::GetRegistryPriorities(NpuArch npuArch) const
{
    return strategy::GetBatchMatMulV3Priorities(npuArch);
}

void BatchMatMulV3Tiling::MergeBatchAndMAxis(MatMulV3BatchInfo& batchInfo)
{
    if (batchInfo.batchB != 1UL || args_.isATrans) {
        return;
    }
    if (batchInfo.batchA > static_cast<uint64_t>(INT32_MAX) / args_.mValue) {
        OP_LOGI(args_.opName, "m value will exceed int32 max value after merge axis, stop merging !");
        return;
    }
    OP_LOGD(args_.opName, "Merge Batch and M axis");
    args_.mValue = batchInfo.batchA * args_.mValue;
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
}
} // namespace batch_matmul_v3_advanced
} // namespace optiling
