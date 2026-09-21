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
 * \file cla_gate_backward_tiling_arch35.cpp
 * \brief
 */

#include "cla_gate_backward_tiling_arch35.h"
#include <cstring>
#include <string>
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"
#include "activation/cla_gate_backward/op_kernel/arch35/cla_gate_backward_tiling_data.h"
#include "activation/cla_gate_backward/op_kernel/arch35/cla_gate_backward_tiling_key.h"

using namespace std;
using namespace ge;
using namespace AscendC;

namespace optiling {

const std::set<ge::DataType> ClaGateBackwardTiling::INPUT_SUPPORT_DTYPE_SET = {ge::DT_FLOAT16, ge::DT_BF16};

int64_t ClaGateBackwardTiling::AlignUp(int64_t value, int64_t align) { return (value + align - 1) / align * align; }

int64_t ClaGateBackwardTiling::MaxI(int64_t a, int64_t b) { return a > b ? a : b; }

int64_t ClaGateBackwardTiling::CeilDiv(int64_t a, int64_t b) { return b <= 0 ? 0 : (a + b - 1) / b; }

int64_t ClaGateBackwardTiling::ReduceTmpBytes(int64_t rows, int64_t headDim)
{
    if (rows <= 0) {
        return 0;
    }
    uint32_t maxTmpBytes = 0;
    uint32_t minTmpBytes = 0;
    const ge::Shape shape({rows, headDim});
    AscendC::GetReduceSumMaxMinTmpSize(shape, ge::DataType::DT_FLOAT, AscendC::ReducePattern::AR, true, false,
                                       maxTmpBytes, minTmpBytes);
    return static_cast<int64_t>(maxTmpBytes);
}

int64_t ClaGateBackwardTiling::ReduceTmpAllocBytes(int64_t rows, int64_t headDim)
{
    return AlignUp(MaxI(ReduceTmpBytes(rows, headDim), MIN_REDUCE_TMP_BYTES), MIN_REDUCE_TMP_BYTES);
}

ge::graphStatus ClaGateBackwardTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    params_.totalCoreNum = platform.GetCoreNumAiv();
    if (params_.totalCoreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "parameter", "invalid", "coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    params_.ubSize = static_cast<int64_t>(ubSize);
    if (params_.ubSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize", std::to_string(params_.ubSize).c_str(),
                                              "The value of ubSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClaGateBackwardTiling::CheckDtype()
{
    auto gradDesc = context_->GetInputDesc(INPUT_GRAD_MERGED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradDesc);
    auto dtype = gradDesc->GetDataType();
    if (INPUT_SUPPORT_DTYPE_SET.count(dtype) == 0) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "grad_merged",
                                  ge::TypeUtils::DataTypeToSerialString(dtype).c_str(), "DT_BF16 or DT_FLOAT16");
        return ge::GRAPH_FAILED;
    }
    static const char* inputNames[] = {"grad_merged", "global_attn", "local_attn", "global_gate_logits",
                                       "local_gate_logits"};
    for (int64_t idx = INPUT_GLOBAL_ATTN; idx <= INPUT_LOCAL_GATE_LOGITS; ++idx) {
        auto desc = context_->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        if (desc->GetDataType() != dtype) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), inputNames[idx],
                                                  ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(),
                                                  "must be the same dtype as grad_merged");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClaGateBackwardTiling::CheckLayout()
{
    const char* inputAttnLayout = LAYOUT_TND;
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        auto layoutPtr = attrs->GetAttrPointer<char>(ATTR_INPUT_ATTN_LAYOUT);
        if (layoutPtr != nullptr) {
            inputAttnLayout = layoutPtr;
        }
    }
    if (std::strcmp(inputAttnLayout, LAYOUT_TND) != 0) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "input_attn_layout", inputAttnLayout, LAYOUT_TND);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClaGateBackwardTiling::ValidateShapeAndGetHeads()
{
    auto gradShapePtr = context_->GetInputShape(INPUT_GRAD_MERGED);
    auto globalAttnShapePtr = context_->GetInputShape(INPUT_GLOBAL_ATTN);
    auto localAttnShapePtr = context_->GetInputShape(INPUT_LOCAL_ATTN);
    auto globalLogitsShapePtr = context_->GetInputShape(INPUT_GLOBAL_GATE_LOGITS);
    auto localLogitsShapePtr = context_->GetInputShape(INPUT_LOCAL_GATE_LOGITS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, globalAttnShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, localAttnShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, globalLogitsShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, localLogitsShapePtr);
    const auto& grad = gradShapePtr->GetStorageShape();
    const auto& globalAttn = globalAttnShapePtr->GetStorageShape();
    const auto& localAttn = localAttnShapePtr->GetStorageShape();
    const auto& globalLogits = globalLogitsShapePtr->GetStorageShape();
    const auto& localLogits = localLogitsShapePtr->GetStorageShape();

    if (grad.GetDimNum() != TND_DIM_NUM || globalAttn.GetDimNum() != TND_DIM_NUM ||
        localAttn.GetDimNum() != TND_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "grad_merged/global_attn/local_attn",
                                     (std::to_string(grad.GetDimNum()) + "," + std::to_string(globalAttn.GetDimNum()) +
                                      "," + std::to_string(localAttn.GetDimNum()))
                                         .c_str(),
                                     std::to_string(TND_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }
    if (grad != globalAttn || grad != localAttn) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "grad_merged, global_attn and local_attn",
            (Ops::Base::ToString(grad) + ", " + Ops::Base::ToString(globalAttn) + ", " + Ops::Base::ToString(localAttn))
                .c_str(),
            "grad_merged, global_attn and local_attn must have the same shape [T, N, D]");
        return ge::GRAPH_FAILED;
    }

    const int64_t t = globalAttn.GetDim(DIM_T);
    const int64_t n = globalAttn.GetDim(DIM_N);
    const int64_t d = globalAttn.GetDim(DIM_D);
    if (d != HEAD_DIM_128 && d != HEAD_DIM_256) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "head_dim", std::to_string(d).c_str(),
                                  (std::to_string(HEAD_DIM_128) + " or " + std::to_string(HEAD_DIM_256)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (n > HEAD_NUM_MAX) {
        OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "head_num", std::to_string(n).c_str(),
                                  (std::to_string(HEAD_NUM_MAX)).c_str());
        return ge::GRAPH_FAILED;
    }
    if (t < 0 || n < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "T/N",
                                              (std::to_string(t) + "," + std::to_string(n)).c_str(),
                                              "dynamic shape not handled here");
        return ge::GRAPH_FAILED;
    }

    if (globalLogits != localLogits) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "global_gate_logits and local_gate_logits",
            (Ops::Base::ToString(globalLogits) + " and " + Ops::Base::ToString(localLogits)).c_str(),
            "global_gate_logits and local_gate_logits must have the same shape [T, N]");
        return ge::GRAPH_FAILED;
    }
    if (globalLogits.GetDimNum() != GATE_LOGITS_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "global_gate_logits",
                                     std::to_string(globalLogits.GetDimNum()).c_str(),
                                     std::to_string(GATE_LOGITS_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }
    if (globalLogits.GetDim(DIM_T) != t || globalLogits.GetDim(DIM_N) != n) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context_->GetNodeName(), "global_gate_logits and global_attn",
            (Ops::Base::ToString(globalLogits) + " and " + Ops::Base::ToString(globalAttn)).c_str(),
            "gate logits [T, N] must match global_attn");
        return ge::GRAPH_FAILED;
    }

    params_.headNum = n;
    params_.headDim = d;
    params_.totalHeads = t * n;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClaGateBackwardTiling::SplitInterCore()
{
    const int64_t elemBytes = DTYPE_BYTES;
    const int64_t headsForMinCopy = (MIN_COPY_BYTES + params_.headDim * elemBytes - 1) / (params_.headDim * elemBytes);
    const int64_t minCopyDivisor = std::min<int64_t>(params_.totalHeads, std::max<int64_t>(headsForMinCopy, 1));
    int64_t coresByCopy = params_.totalHeads / minCopyDivisor;
    coresByCopy = std::max<int64_t>(coresByCopy, 1);
    params_.usedCoreNum = std::min(params_.totalCoreNum, coresByCopy);
    params_.usedCoreNum = std::max<int64_t>(params_.usedCoreNum, 1);
    params_.baseCoreHeads = params_.totalHeads / params_.usedCoreNum;
    params_.extraCoreCount = params_.totalHeads % params_.usedCoreNum;
    params_.coreHeadsMax = params_.baseCoreHeads + (params_.extraCoreCount > 0 ? 1 : 0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClaGateBackwardTiling::SolveBatch()
{
    const int64_t ubBudget = params_.ubSize;
    // batch 系数（忽略Reduce临时空间和对齐多余空间）
    const int64_t perBatch = BATCH_INIT_TND_COEF * params_.headDim + BATCH_INIT_SCALAR_COEF;

    // batch 最大值
    int64_t batch = std::min<int64_t>(ubBudget / perBatch, params_.coreHeadsMax);
    const int64_t alignOverhead = BATCH_INIT_SCALAR_COEF * (AlignUp(batch, VEC_LANES_FP32) - batch); // 对齐用的多余空间
    const int64_t tmpOverhead = ReduceTmpAllocBytes(batch, params_.headDim); // Reduce临时空间
    batch = std::min<int64_t>((ubBudget - alignOverhead - tmpOverhead) / perBatch, params_.coreHeadsMax);

    if (batch < 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize", std::to_string(ubBudget).c_str(),
                                              "UB is too small to hold one batch");
        return ge::GRAPH_FAILED;
    }

    const int64_t headCoreHeads = params_.baseCoreHeads + (params_.extraCoreCount > 0 ? 1 : 0);
    const int64_t tailCoreHeads = params_.baseCoreHeads;
    params_.batch = batch;
    params_.headCoreLoopCount = CeilDiv(headCoreHeads, batch);
    params_.headCoreHeadsPerLoop = params_.headCoreLoopCount > 0 ? CeilDiv(headCoreHeads, params_.headCoreLoopCount) :
                                                                   0;
    params_.tailCoreLoopCount = CeilDiv(tailCoreHeads, batch);
    params_.tailCoreHeadsPerLoop = params_.tailCoreLoopCount > 0 ? CeilDiv(tailCoreHeads, params_.tailCoreLoopCount) :
                                                                   0;
    params_.reduceTmpSize = ReduceTmpBytes(batch, params_.headDim);
    return ge::GRAPH_SUCCESS;
}

void ClaGateBackwardTiling::FillTilingData()
{
    data_->usedCoreNum = params_.usedCoreNum;
    data_->baseCoreHeads = params_.baseCoreHeads;
    data_->extraCoreCount = params_.extraCoreCount;
    data_->batch = params_.batch;
    data_->headCoreLoopCount = params_.headCoreLoopCount;
    data_->headCoreHeadsPerLoop = params_.headCoreHeadsPerLoop;
    data_->tailCoreLoopCount = params_.tailCoreLoopCount;
    data_->tailCoreHeadsPerLoop = params_.tailCoreHeadsPerLoop;
    data_->reduceTmpSize = params_.reduceTmpSize;
}

ge::graphStatus ClaGateBackwardTiling::FillAndSetBlockDim()
{
    FillTilingData();
    context_->SetBlockDim(static_cast<uint32_t>(params_.usedCoreNum));
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = SYSTEM_WORKSPACE;
    context_->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClaGateBackwardTiling::Run()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_);

    if (GetPlatformInfo() != ge::GRAPH_SUCCESS || CheckDtype() != ge::GRAPH_SUCCESS ||
        CheckLayout() != ge::GRAPH_SUCCESS || ValidateShapeAndGetHeads() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    data_ = context_->GetTilingData<ClaGateBackwardTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, data_);
    *data_ = {};
    data_->headNum = params_.headNum;
    data_->headDim = params_.headDim;
    data_->totalHeads = params_.totalHeads;

    if (params_.totalHeads == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "T*N", "0",
                                              "ClaGateBackward does not support empty tensor");
        return ge::GRAPH_FAILED;
    }

    if (SplitInterCore() != ge::GRAPH_SUCCESS || SolveBatch() != ge::GRAPH_SUCCESS ||
        FillAndSetBlockDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(context_,
            "ClaGateBackward tiling: TN=%ld, N=%ld, D=%ld, cores=%ld, base=%ld, extra=%ld, "
            "batch=%ld, headLoop=%ld/%ld, tailLoop=%ld/%ld, reduceTmp=%ld",
            params_.totalHeads, params_.headNum, params_.headDim, params_.usedCoreNum, params_.baseCoreHeads,
            params_.extraCoreCount, params_.batch, params_.headCoreLoopCount, params_.headCoreHeadsPerLoop,
            params_.tailCoreLoopCount, params_.tailCoreHeadsPerLoop, params_.reduceTmpSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForClaGateBackward(gert::TilingContext* context)
{
    ClaGateBackwardTiling tiling(context);
    return tiling.Run();
}

ge::graphStatus TilingPrepareForClaGateBackward(gert::TilingParseContext* context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ClaGateBackward)
    .Tiling(TilingForClaGateBackward)
    .TilingParse<ClaGateBackwardCompileInfo>(TilingPrepareForClaGateBackward);

} // namespace optiling
