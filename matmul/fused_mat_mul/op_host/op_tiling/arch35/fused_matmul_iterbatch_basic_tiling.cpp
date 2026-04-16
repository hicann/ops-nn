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
 * \file fused_matmul_iterbatch_basic_tiling.cpp
 * \brief
 */
#include "fused_matmul_iterbatch_basic_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;
// ------------------------------ SetNeedNdDma -------------------------------------------//
void SetNeedNdDmaDefault(
    const MatmulV3CompileInfo& /* compileInfo */, const MatMulV3Args& /* args */, MatMulV3RunInfo& /* runInfo*/)
{
    return;
}

void SetNeedNdDmaDav3510(const MatmulV3CompileInfo& compileInfo, const MatMulV3Args& args, MatMulV3RunInfo& runInfo)
{
    uint64_t restUB = compileInfo.ubSize - runInfo.iterBatchL0 * runInfo.baseM * runInfo.baseN;
    uint64_t singleBatchSize = args.mValue * runInfo.baseN;
    uint64_t iterBatchX3 = std::min(runInfo.iterBatchL0, restUB / NUM_TWO / singleBatchSize);
    runInfo.needNdDma = (iterBatchX3 > 1) && (args.batchX3 == 1);
}

using SetNeedNdDmaFunc = void (*)(const MatmulV3CompileInfo&, const MatMulV3Args&, MatMulV3RunInfo&);

const static std::map<NpuArch, SetNeedNdDmaFunc> SetNeedNdDmaFuncMap = {
    {NpuArch::DAV_3510, SetNeedNdDmaDav3510},
};
} // namespace

namespace optiling {
namespace fused_matmul {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(
    FusedMatMul, FusedMatMulIterBatchApiTiling, DAV_3510, ITER_BATCH_BASICAPI_INHERITED_FROM_BMMV3);
MM_REGISTER_TILING_TEMPLATE(
    FusedMatMul, FusedMatMulIterBatchApiTiling, DAV_RESV, ITER_BATCH_BASICAPI_INHERITED_FROM_BMMV3);

void FusedMatMulIterBatchApiTiling::SetNeedNdDma()
{
    auto iter = (SetNeedNdDmaFuncMap.find(compileInfo_.npuArch) == SetNeedNdDmaFuncMap.end()) ?
                    SetNeedNdDmaDefault :
                    SetNeedNdDmaFuncMap.at(compileInfo_.npuArch);
    iter(compileInfo_, args_, runInfo_);
}

bool FusedMatMulIterBatchApiTiling::IsCapable()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    bool status = BatchMatMulV3IterBatchBasicApiTiling::IsCapable();
    if (!status) {
        OP_LOGD(args_.opName, "IterBatch model is not supported for this shape");
        return false;
    }
    if (opType == "add" || opType == "mul") {
        // when optype is "add" or "mul", aivNum must == aicNum * 2
        if (compileInfo_.aivNum != (compileInfo_.aicNum * NUM_TWO)) {
            OP_LOGW(args_.opName, "FusedMatMul IterBatch model only support aivNum == aicNum *2");
            return false;
        }
    }
    OP_LOGI(args_.opName, "FusedMatMul tiling enable iterbatch basic api");
    return true;
}

uint64_t FusedMatMulIterBatchApiTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    // has x3 must be add or mul
    if (args_.hasX3Input) {
        return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
            .SetBatchModel(MatMulV3BatchModel::SINGLE_BIAS_MODEL)
            .SetL0C2Out(MatMulV3L0C2Out::ND_FIXPIPE_1_2)
            .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
            .GetTilingKey();
    }
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetBatchModel(MatMulV3BatchModel::SINGLE_BIAS_MODEL)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

ge::graphStatus FusedMatMulIterBatchApiTiling::DoOpTiling() {
    ge::graphStatus status = BatchMatMulV3IterBatchBasicApiTiling::DoOpTiling();
    SetNeedNdDma();
    return status;
}

} // namespace fused_matmul
} // namespace optiling