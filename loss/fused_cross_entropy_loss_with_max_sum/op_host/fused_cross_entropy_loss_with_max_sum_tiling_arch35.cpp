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
 * \file fused_cross_entropy_loss_with_max_sum_tiling_arch35.cpp
 * \brief FusedCrossEntropyLossWithMaxSum regbase(ascend950) tiling
 */

#include <cstring>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "fused_cross_entropy_loss_with_max_sum_tiling.h"
#include "../op_kernel/arch35/fused_cross_entropy_loss_with_max_sum_tiling_data.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
namespace {
constexpr uint32_t INPUT_VOCAB_INDEX = 5;
constexpr uint32_t INPUT_LOGITS_MAX_INDEX = 0;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr int64_t RESERVED_UB = 2 * 1024;
constexpr int64_t SCALAR_QUEUE_NUM = 5;   // logitsMax/sumExp/predicted/loss队列 + invSum的TBuf
constexpr int64_t SCALAR_BUF_BYTES = 256; // 标量buffer按一个向量寄存器宽度（64个fp32）预留
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t HALF_BYTES = 2;
constexpr int64_t MEM_QUEUE_NUM = 3;                // 省显存路径：sumExp/predicted/loss三个队列
constexpr size_t WORKSPACE_SIZE = 16 * 1024 * 1024; // 系统workspace（保持参数槽位对齐所必需）
} // namespace

class FusedCrossEntropyLossWithMaxSumRegbaseTiling {
public:
    explicit FusedCrossEntropyLossWithMaxSumRegbaseTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus DoTilingFull();
    ge::graphStatus DoTilingForMemory();
    void SplitRows(int64_t bt, int64_t rowCores, int64_t totalCores);
    void SplitV(int64_t bt, int64_t v, int64_t vPerLoop);
    static int64_t CeilAlign(int64_t a, int64_t b) { return (a + b - 1) / b * b; }
    static int64_t FloorAlign(int64_t a, int64_t b) { return a / b * b; }

private:
    gert::TilingContext* context_ = nullptr;
    FusedCrossEntropyLossWithMaxSumRegBaseTilingData* tilingData_ = nullptr;
    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
};

ge::graphStatus FusedCrossEntropyLossWithMaxSumRegbaseTiling::Init()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ <= 0, OP_LOGE(context_, "coreNum must greater than zero, but is %ld", coreNum_),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0UL, OP_LOGE(context_, "ubSize must greater than zero"), return ge::GRAPH_FAILED);
    ubSize_ = static_cast<int64_t>(ubSize);
    OP_LOGI(context_, "FusedCrossEntropyLossWithMaxSum regbase tiling, coreNum is %ld, ubSize is %ld", coreNum_,
            ubSize_);
    tilingData_ = context_->GetTilingData<FusedCrossEntropyLossWithMaxSumRegBaseTilingData>();
    OP_CHECK_IF(tilingData_ == nullptr, OP_LOGE(context_, "get tilingdata ptr failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(memset_s(tilingData_, sizeof(FusedCrossEntropyLossWithMaxSumRegBaseTilingData), 0,
                         sizeof(FusedCrossEntropyLossWithMaxSumRegBaseTilingData)) != EOK,
                OP_LOGE(context_, "memset tilingdata failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void FusedCrossEntropyLossWithMaxSumRegbaseTiling::SplitRows(int64_t bt, int64_t rowCores, int64_t totalCores)
{
    // 按行切分：rowCores个核分担bt行（former核每核formerRows行，latter核每核latterRows行），
    // totalCores为总启动核数（v切分时 rowCores * vCores）
    int64_t formerCoreNum = bt % rowCores;
    int64_t latterRows = bt / rowCores;
    tilingData_->formerCoreNum = formerCoreNum;
    tilingData_->formerRows = latterRows + 1;
    tilingData_->latterRows = latterRows;
    context_->SetBlockDim(totalCores);
}

void FusedCrossEntropyLossWithMaxSumRegbaseTiling::SplitV(int64_t bt, int64_t v, int64_t vPerLoop)
{
    // 行数不足以占满全部核时，把每行的v维再切给多个核处理（v天然无归约、逐列独立），提高核利用率。
    // 每核至少分到一个完整UB tile（v > vPerLoop）才切，vChunk按V_ALIGN对齐保证UB行距约束。
    tilingData_->vCores = 1;
    tilingData_->vChunk = v;
    if (bt >= coreNum_) {
        return;
    }
    int64_t vCoresMax = coreNum_ / bt;
    int64_t vCoresNeed = (v + vPerLoop - 1) / vPerLoop;
    int64_t vCores = vCoresMax < vCoresNeed ? vCoresMax : vCoresNeed;
    if (vCores <= 1) {
        return;
    }
    tilingData_->vCores = vCores;
    tilingData_->vChunk = CeilAlign((v + vCores - 1) / vCores, FUSED_CE_MAX_SUM_V_ALIGN);
}

ge::graphStatus FusedCrossEntropyLossWithMaxSumRegbaseTiling::DoTilingFull()
{
    auto tensor = context_->GetOptionalInputTensor(INPUT_VOCAB_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensor);
    int64_t bt = tensor->GetStorageShape().GetDim(DIM_INDEX0);
    int64_t v = tensor->GetStorageShape().GetDim(DIM_INDEX1);
    OP_CHECK_IF(bt <= 0 || v <= 0, OP_LOGE(context_, "invalid vocab shape [%ld, %ld]", bt, v), return ge::GRAPH_FAILED);
    auto dataType = tensor->GetDataType();
    int64_t dtypeSize = dataType == ge::DT_FLOAT ? FLOAT_BYTES : HALF_BYTES;
    int64_t vocabDtypeId = FUSED_CE_MAX_SUM_DTYPE_BF16;
    if (dataType == ge::DT_FLOAT) {
        vocabDtypeId = FUSED_CE_MAX_SUM_DTYPE_FP32;
    } else if (dataType == ge::DT_FLOAT16) {
        vocabDtypeId = FUSED_CE_MAX_SUM_DTYPE_FP16;
    }

    // UB预算：vocab双缓冲 + softmax双缓冲，每列开销为 A_PER_LOOP * DB * (dtypeSize + fp32)
    int64_t scalarReserve = SCALAR_QUEUE_NUM * SCALAR_BUF_BYTES + RESERVED_UB;
    int64_t ubAvail = ubSize_ - scalarReserve;
    int64_t perColBytes = FUSED_CE_MAX_SUM_A_PER_LOOP * DOUBLE_BUFFER_NUM * (dtypeSize + FLOAT_BYTES);
    int64_t vMax = FloorAlign(ubAvail / perColBytes, FUSED_CE_MAX_SUM_V_ALIGN);
    int64_t vAligned = CeilAlign(v, FUSED_CE_MAX_SUM_V_ALIGN);
    int64_t vPerLoop = vMax < vAligned ? vMax : vAligned;
    OP_CHECK_IF(vPerLoop < FUSED_CE_MAX_SUM_V_ALIGN, OP_LOGE(context_, "ub is not enough, vPerLoop is %ld", vPerLoop),
                return ge::GRAPH_FAILED);

    // bt不足占满核时按v维二次切核，总核数 = rowCores * vCores（不超过coreNum_）
    SplitV(bt, v, vPerLoop);
    int64_t rowCores = bt < coreNum_ / tilingData_->vCores ? bt : coreNum_ / tilingData_->vCores;
    SplitRows(bt, rowCores, rowCores * tilingData_->vCores);

    tilingData_->vPerLoop = vPerLoop;
    tilingData_->vLen = v;
    tilingData_->elementsNumber = 0;
    tilingData_->vocabDtypeId = vocabDtypeId;

    constexpr uint64_t tilingKey = 0; // TILINGKEY_FULL
    OP_LOGI(context_,
            "DoTilingFull: bt is %ld, v is %ld, vPerLoop is %ld, formerCoreNum is %ld, formerRows is %ld, "
            "latterRows is %ld, vCores is %ld, vChunk is %ld, tilingKey is %lu",
            bt, v, vPerLoop, tilingData_->formerCoreNum, tilingData_->formerRows, tilingData_->latterRows,
            tilingData_->vCores, tilingData_->vChunk, tilingKey);
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCrossEntropyLossWithMaxSumRegbaseTiling::DoTilingForMemory()
{
    auto tensor = context_->GetInputTensor(INPUT_LOGITS_MAX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tensor);
    int64_t bt = tensor->GetStorageShape().GetDim(DIM_INDEX0);
    OP_CHECK_IF(bt <= 0, OP_LOGE(context_, "invalid bt %ld", bt), return ge::GRAPH_FAILED);

    int64_t rowCores = bt < coreNum_ ? bt : coreNum_;
    SplitRows(bt, rowCores, rowCores);

    // UB预算：sumExp/predicted/loss三个队列均双缓冲
    int64_t ubAvail = ubSize_ - RESERVED_UB;
    int64_t elementsNumber = FloorAlign(ubAvail / (MEM_QUEUE_NUM * DOUBLE_BUFFER_NUM * FLOAT_BYTES),
                                        FUSED_CE_MAX_SUM_V_ALIGN);
    OP_CHECK_IF(elementsNumber < FUSED_CE_MAX_SUM_V_ALIGN,
                OP_LOGE(context_, "ub is not enough, elementsNumber is %ld", elementsNumber), return ge::GRAPH_FAILED);

    tilingData_->vPerLoop = 0;
    tilingData_->vLen = 0;
    tilingData_->elementsNumber = elementsNumber;
    tilingData_->vCores = 1; // 省显存路径只有bt一维，不切v
    tilingData_->vChunk = 0;

    constexpr uint64_t tilingKey = 1; // TILINGKEY_MEMORY
    OP_LOGI(context_,
            "DoTilingForMemory: bt is %ld, elementsNumber is %ld, formerCoreNum is %ld, formerRows is %ld, "
            "latterRows is %ld, tilingKey is %lu",
            bt, elementsNumber, tilingData_->formerCoreNum, tilingData_->formerRows, tilingData_->latterRows,
            tilingKey);
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedCrossEntropyLossWithMaxSumRegbaseTiling::DoTiling()
{
    // kernel ABI 固定包含 workspace 参数，必须分配 workspace 保证运行时参数槽位对齐
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    auto tensor = context_->GetOptionalInputTensor(INPUT_VOCAB_INDEX);
    if (tensor == nullptr) {
        return DoTilingForMemory();
    }
    return DoTilingFull();
}

ge::graphStatus TilingFusedCrossEntropyLossWithMaxSumRegbase(gert::TilingContext* context)
{
    OP_LOGD(context, "Start TilingFusedCrossEntropyLossWithMaxSumRegbase.");
    FusedCrossEntropyLossWithMaxSumRegbaseTiling tilingImpl(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "TilingFusedCrossEntropyLossWithMaxSumRegbase init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "TilingFusedCrossEntropyLossWithMaxSumRegbase do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "End TilingFusedCrossEntropyLossWithMaxSumRegbase.");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
