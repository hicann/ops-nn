/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file prelu_grad_reduce_tiling_arch35.cpp
 * \brief tiling for prelu_grad_reduce
 */

#include <vector>
#include "prelu_grad_reduce_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "activation/p_relu_grad_reduce/op_kernel/arch35/prelu_grad_reduce_dag.h"
#include "activation/p_relu_grad_reduce/op_kernel/arch35/prelu_grad_reduce_struct.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "atvoss/reduce/reduce_tiling.h"
#include "atvoss/reduce/reduce_tiling_data.h"
#include "op_host/tiling_util.h"
#include "error_util.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

using namespace Ops::Base;

namespace optiling
{
static constexpr int32_t SIZE8 = 8;
static constexpr int32_t SIZE4 = 4;
static constexpr int32_t SIZE2 = 2;
static ge::graphStatus DoTiling(gert::TilingContext* context, ReduceOpInputParam& opInput, ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;

    if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE8) {
        status = Tiling4ReduceOp<PreluGradReduce::PreluGradReduceDag<int64_t, int64_t>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE4) {
        status = Tiling4ReduceOp<PreluGradReduce::PreluGradReduceDag<float, float>::OpDag>(context, opInput, key);
    } else if (ge::GetSizeByDataType(opInput.inputDtype) == SIZE2) {
        status = Tiling4ReduceOp<PreluGradReduce::PreluGradReduceDag<half, float>::OpDag>(context, opInput, key);
    }
    OP_TILING_CHECK(
        (status == ge::GRAPH_FAILED),
        VECTOR_INNER_ERR_REPORT_TILIING(
            context->GetNodeName(), "PReluGradReduce Tiling failed, dtype should be in (bfloat16/float16/float/int32/int64)"),
        return ge::GRAPH_FAILED);
    return status;
}

static ge::graphStatus GetPreluGradReduceAxes(gert::TilingContext* context, ReduceOpInputParam& opInput)
{
    auto features = context->GetInputShape(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, features);
    auto weight = context->GetInputShape(2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, weight);
    gert::Shape featureShape = Ops::NN::OpTiling::EnsureNotScalar(features->GetStorageShape());
    gert::Shape weightShape = Ops::NN::OpTiling::EnsureNotScalar(weight->GetStorageShape());
    size_t featureDimNum = featureShape.GetDimNum();
    size_t weightDimNum = weightShape.GetDimNum();

    opInput.axes.clear();
    if (weightShape.GetShapeSize() == 1L) {
        for (size_t i = 0; i < featureDimNum; i++) {
            opInput.axes.push_back(i);
        }
    } else if (weightDimNum == featureDimNum - 1 && weightDimNum > 1) {
        opInput.axes.push_back(0);
        for (size_t i = 1; i < featureDimNum; i++) {
            if (featureShape.GetDim(i) != weightShape.GetDim(i) && weightShape.GetDim(i) == 1) {
                opInput.axes.push_back(i);
            }
        }
    } else {
        for (size_t i = 0; i < featureDimNum; i++) {
            opInput.axes.push_back(i);
        }
        if (opInput.axes.size() > 1) {
           opInput.axes.erase(opInput.axes.begin() + 1); 
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4PreluGradReduce(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    ReduceOpInputParam opInput;
    OP_TILING_CHECK((ReduceOpTmpl::GetInputParam(context, opInput, 3) == ge::GRAPH_FAILED),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "PReluGradReduce get input param failed"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((GetPreluGradReduceAxes(context, opInput) == ge::GRAPH_FAILED),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "PReluGradReduce get reduce axes failed"),
                    return ge::GRAPH_FAILED);

    ReduceTilingKey key;
    OP_TILING_CHECK((DoTiling(context, opInput, key) == ge::GRAPH_FAILED),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "DoTiling Failed for PreluGradReduce"),
                    return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(key.patternID, key.loopARCount, key.loopInnerARCount);
    OP_LOGI(context->GetNodeName(), "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu",
            key.patternID, key.loopARCount, key.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4PreluGradReduce(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ReduceOpCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->vectorCoreNum == 0UL),
        OP_LOGE(context->GetNodeName(), "TilingPrepare4PreluGradReduce GetHardwareInfo Failed, vectorCoreNum:%lu",
                                        compileInfo->vectorCoreNum),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= CACHE_BUF_SIZE,
                OP_LOGE(context->GetNodeName(),
                        "TilingPrepare4PreluGradReduce GetHardwareInfo Failed, ubSize:%lu, at least:%lu.",
                        compileInfo->ubSize, CACHE_BUF_SIZE),
                return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        compileInfo->cacheLineSize == 0UL,
        OP_LOGE(context->GetNodeName(), "TilingPrepare4PreluGradReduce GetHardwareInfo Failed, cacheLineSize:%lu.",
                compileInfo->cacheLineSize),
        return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        compileInfo->ubBlockSize == 0UL,
        OP_LOGE(context->GetNodeName(), "TilingPrepare4PreluGradReduce GetHardwareInfo Failed, ubBlockSize:%lu.",
                compileInfo->ubBlockSize),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        compileInfo->vRegSize == 0UL,
        OP_LOGE(context->GetNodeName(), "TilingPrepare4PreluGradReduce GetHardwareInfo Failed, vRegSize:%lu.",
                compileInfo->vRegSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
            compileInfo->vectorCoreNum, compileInfo->ubSize, compileInfo->cacheLineSize, compileInfo->ubBlockSize,
            compileInfo->vRegSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PReluGradReduce).Tiling(Tiling4PreluGradReduce).TilingParse<ReduceOpCompileInfo>(TilingPrepare4PreluGradReduce);
}  // namespace optiling