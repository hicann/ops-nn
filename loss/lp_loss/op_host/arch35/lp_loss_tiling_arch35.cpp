/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file lp_loss_tiling_arch35.cpp
 * \brief
 */

#include "tiling/tiling_api.h"
#include "loss/lp_loss/op_kernel/arch35/lp_loss_dag.h"
#include "loss/lp_loss/op_kernel/arch35/lp_loss_tiling_key.h"

#include "lp_loss_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "util/platform_util.h"
#include "atvoss/reduce/reduce_util.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling
{
static const int64_t ASCEND_WORKSPACE = 16 * 1024 * 1024;
static const uint32_t REDUCTION_MEAN = 2;
static const map<std::string, uint32_t> STR_2_INT = {{"none", 0}, {"sum", 1}, {"mean", 2}};
static const map<ge::DataType, uint32_t> DTYEP_2_INT_KEY{{ge::DT_FLOAT16, 10}, {ge::DT_FLOAT, 20}, {ge::DT_BF16, 30}};
class LpLossTiling
{
public:
    explicit LpLossTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling(const ReduceOpCompileInfo* compileInfo);

protected:
    ge::graphStatus SetTilingData();
    ge::graphStatus CheckShape();
    ge::graphStatus TilingEle();
    ge::graphStatus TilingReduce(const ReduceOpCompileInfo* compileInfo);

private:
    ge::DataType outputDtype;
    gert::TilingContext* tilingContext;
    LpLossTilingKey key;
    LpLossTilingData* tiling = nullptr;
    uint32_t reduction;
};

ge::graphStatus LpLossTiling::CheckShape()
{
    auto predictStorageShape = tilingContext->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, predictStorageShape);
    const gert::Shape& inputPShape = EnsureNotScalar(predictStorageShape->GetStorageShape());
    auto labelStorageShape = tilingContext->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, labelStorageShape);
    const gert::Shape& inputLShape = EnsureNotScalar(labelStorageShape->GetStorageShape());
    auto inputPDesc = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputPDesc);
    auto inputLDesc = tilingContext->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, inputLDesc);
    auto outputDesc = tilingContext->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, outputDesc);
    this->outputDtype = outputDesc->GetDataType();
    auto iter = DTYEP_2_INT_KEY.find(this->outputDtype);
    OP_CHECK_IF(iter == DTYEP_2_INT_KEY.end(),
                    OP_LOGE(tilingContext->GetNodeName(), "output dtype is not support."),
                    return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        inputPDesc->GetDataType() != inputLDesc->GetDataType(),
        OP_LOGE(tilingContext->GetNodeName(), "input predict and lable dtype not same"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputPShape != inputLShape,
               OP_LOGE(tilingContext->GetNodeName(), "input predict and lable shape not same"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpLossTiling::SetTilingData()
{
    OP_LOGD(tilingContext->GetNodeName(), "Enter SetTilingData");

    const uint64_t tilingKey = GET_TPL_TILING_KEY(key.ReduceTiling.patternID, key.ReduceTiling.loopARCount,
                                                  key.ReduceTiling.loopInnerARCount, key.Reduction, key.Dtype);
    OP_LOGI(tilingContext->GetNodeName(),
            "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu, Reducetiong is : %u, Dtype is %u",
            key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount, tilingKey,
            key.Reduction, key.Dtype);
    if (static_cast<int32_t>(this->reduction) < 1) {
        size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
        currentWorkspace[0] = ASCEND_WORKSPACE;
        tilingContext->SetBlockDim(tiling->baseTiling.blockNum);
    }
    tilingContext->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpLossTiling::TilingEle()
{
    ElewiseBaseTiling eleBaseTiling(tilingContext);
    key.Dtype = DTYEP_2_INT_KEY.at(this->outputDtype);
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        // fp16需要cast成fp32处理
        OP_CHECK_IF(eleBaseTiling.DoTiling<LpLoss::LpLossOp<half>::OpDag>(tiling->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for fp16"),
                        return ge::GRAPH_FAILED);
    } else if (this->outputDtype == ge::DT_FLOAT) {
        OP_CHECK_IF(eleBaseTiling.DoTiling<LpLoss::LpLossOp<float>::OpDag>(tiling->baseTiling) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for fp32"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "current dtype not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpLossTiling::TilingReduce(const ReduceOpCompileInfo* compileInfo)
{
    ReduceOpInputParam opInput;
    OP_CHECK_IF((ReduceOpTmpl::GetInputParam(tilingContext, opInput, 0) == ge::GRAPH_FAILED),
                    OP_LOGE(tilingContext->GetNodeName(), "ReduceOp get x dtype failed"),
                    return ge::GRAPH_FAILED);

    opInput.axes.resize(opInput.shape.size());
    for (size_t i = 0; i < opInput.shape.size(); i++) {
        opInput.axes[i] = i;
    }
    if (this->outputDtype == ge::DT_FLOAT16 || this->outputDtype == ge::DT_BF16) {
        if (static_cast<int32_t>(this->reduction) == 1) {
            OP_CHECK_IF((Tiling4ReduceOp<LpLoss::LpLossSumDag<half, float>::OpDag>(
                                 tilingContext, opInput, key.ReduceTiling, compileInfo, &(tiling->reduceTiling)) == ge::GRAPH_FAILED),
                            OP_LOGE(tilingContext->GetNodeName(), "LpLoss Tiling failed"),
                            return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF((Tiling4ReduceOp<LpLoss::LpLossMeanDag<half, float>::OpDag>(
                                 tilingContext, opInput, key.ReduceTiling, compileInfo, &(tiling->reduceTiling)) == ge::GRAPH_FAILED),
                            OP_LOGE(tilingContext->GetNodeName(), "LpLoss Tiling failed"),
                            return ge::GRAPH_FAILED);
        }
    } else {
        if (static_cast<int32_t>(this->reduction) == 1) {
            OP_CHECK_IF((Tiling4ReduceOp<LpLoss::LpLossSumDag<float, float>::OpDag>(
                                 tilingContext, opInput, key.ReduceTiling, compileInfo, &(tiling->reduceTiling)) == ge::GRAPH_FAILED),
                            OP_LOGE(tilingContext->GetNodeName(), "LpLoss Tiling failed"),
                            return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF((Tiling4ReduceOp<LpLoss::LpLossMeanDag<float, float>::OpDag>(
                                 tilingContext, opInput, key.ReduceTiling, compileInfo, &(tiling->reduceTiling)) == ge::GRAPH_FAILED),
                            OP_LOGE(tilingContext->GetNodeName(), "LpLoss Tiling failed"),
                            return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LpLossTiling::RunTiling(const ReduceOpCompileInfo* compileInfo)
{
    OP_CHECK_IF(CheckShape() == ge::GRAPH_FAILED, OP_LOGE(tilingContext, "check shape failed"),
               return ge::GRAPH_FAILED);
    auto attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    tiling = tilingContext->GetTilingData<LpLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tiling);
    auto P = attrs->GetAttrPointer<int>(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, P);
    OP_CHECK_IF(*P != 1,
                    OP_LOGE(tilingContext->GetNodeName(), "Lploss only support p is 1."),
                    return ge::GRAPH_FAILED);
    const string reductionStr = string(attrs->GetAttrPointer<char>(1));
    auto iter = STR_2_INT.find(reductionStr);
    OP_CHECK_IF(iter == STR_2_INT.end(),
                    OP_LOGE(tilingContext->GetNodeName(), "reduction is not support."),
                    return ge::GRAPH_FAILED);
    this->reduction = iter->second;
    key.Reduction = this->reduction;
    if (reductionStr == "none") {
        OP_CHECK_IF(TilingEle() != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for elewise"),
                        return ge::GRAPH_FAILED);
    } else if (reductionStr == "mean" || reductionStr == "sum") {
        OP_CHECK_IF(TilingReduce(compileInfo) != ge::GRAPH_SUCCESS,
                        OP_LOGE(tilingContext->GetNodeName(), "do tiling failed for reduce"),
                        return ge::GRAPH_FAILED);
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "reduction %s is not supported",
                                        reductionStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return SetTilingData();
}

static ge::graphStatus TilingForLpLoss(gert::TilingContext* context)
{
    OP_LOGD("LpLossTiling", "Enter TilingForLpLoss");
    if (context == nullptr) {
        OP_LOGE("LpLossTiling", "Tiling context is null");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = static_cast<const ReduceOpCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // 走新的模板tiling
    OP_LOGD("LpLossTiling", "Enter new LpLossTiling");
    LpLossTiling tiling(context);
    return tiling.RunTiling(compileInfo);
}

ge::graphStatus TilingPrepareForLpLoss(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ReduceOpCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->vectorCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(
        (compileInfo->vectorCoreNum == 0UL),
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vectorCoreNum:%lu",
                                        compileInfo->vectorCoreNum),
        return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= CACHE_BUF_SIZE,
                    OP_LOGE(context->GetNodeName(),
                                                    "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%lu.",
                                                    compileInfo->ubSize, CACHE_BUF_SIZE),
                    return ge::GRAPH_FAILED);
    compileInfo->ubSize = ubSize;

    compileInfo->cacheLineSize = Ops::Base::GetCacheLineSize(context);
    OP_CHECK_IF(
        compileInfo->cacheLineSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, cacheLineSize:%lu.",
                                        compileInfo->cacheLineSize),
        return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(
        compileInfo->ubBlockSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, ubBlockSize:%lu.",
                                        compileInfo->ubBlockSize),
        return ge::GRAPH_FAILED);

    compileInfo->vRegSize = Ops::Base::GetVRegSize(context);
    OP_CHECK_IF(
        compileInfo->vRegSize == 0UL,
        OP_LOGE(context->GetNodeName(), "ReduceOp GetHardwareInfo Failed, vRegSize:%lu.",
                                        compileInfo->vRegSize),
        return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "GetCoreNum:%lu, ubSize:%lu, cacheLineSize:%lu, ubBlockSize:%lu, vRegSize:%lu",
            compileInfo->vectorCoreNum, compileInfo->ubSize, compileInfo->cacheLineSize, compileInfo->ubBlockSize,
            compileInfo->vRegSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LpLoss).Tiling(TilingForLpLoss).TilingParse<ReduceOpCompileInfo>(TilingPrepareForLpLoss);
}  // namespace optiling