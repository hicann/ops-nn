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
 * \file inplace_index_add_with_sorted_tiling_arch35.cpp
 * \brief A5 (ascend950) tiling logic — bufCnt=6, 单默认调度模式
 */

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "inplace_index_add_with_sorted_tiling_arch35.h"
#include "platform/platform_infos_def.h"
#include "../../op_kernel/arch35/inplace_index_add_with_sorted_tiling_key.h"

#include <limits>

using namespace std;
using Ops::Base::CeilAlign;

namespace {
const int32_t SIZE_OF_FP16 = 2;
const int32_t SIZE_OF_BF16 = 2;
const int32_t SIZE_OF_FP32 = 4;
const int32_t SIZE_OF_INT32 = 4;
const int32_t MAX_DIM_NUM = 8;

const int32_t INPUT_0 = 0;
const int32_t INPUT_1 = 1;
const int32_t INPUT_2 = 2;
const int32_t INPUT_3 = 3;
const int32_t INPUT_4 = 4;
// maxSize 依赖的 buffer 系数（B/elem）：updateQue(BN=2,fp32)=8（第一、二阶段复用）+
// varInQue(BN=2,fp32)=8 + accumBuf(BN=1,fp32)=4 + outQue(BN=2,T=fp16)=4 = 24 B/elem → BUF_CNT = 24/4 = 6
const int32_t BUF_CNT = 6;
const int32_t BLOCK_SIZE = 32;
const int32_t ELEMENTS_PER_BLOCK = BLOCK_SIZE / SIZE_OF_FP16;
const int64_t UB_INDEX_NUM = 1536;
const int64_t INDEX_BUFFER_SIZE = UB_INDEX_NUM * 2 * SIZE_OF_INT32;
const int64_t WS_ROWS_PER_CORE = 2; // 每核 workspace 行数：前哨兵 + 后哨兵
// 默认 workspace 大小（与 PR #8205 cross_entropy_sumexp_and_index_logit 同款处理）
// Init() 不再依赖 compileInfo->workspaceSize / GetLibApiWorkSpaceSize()，
// 而是固定一个安全默认值 + 后续 GetWorkspaceSizes(1) 显式校验。
constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

} // namespace

namespace optiling {
class InplaceIndexAddWithSortedTiling {
public:
    explicit InplaceIndexAddWithSortedTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    bool TilingDataSet();
    void TilingDataPrint() const;
    bool processFirstDimTilingData();
    bool CheckParam();
    InplaceIndexAddWithSortedTilingData* tilingData_{nullptr};
    gert::TilingContext* tilingContext = nullptr;
    int64_t dimAttr = -1;
    int64_t ubSize = 1;
    int64_t inputCount = 1;
    int64_t updatesCount = 1;
    int64_t indicesCount = 1;
    int64_t updatesOneTime = 1;
    int64_t inputSize = 1;
    int32_t coreNum = 1;

    int32_t usedCoreNum = 1;
    int32_t enableAlpha = 0;
    int64_t eachIndexCount = 1;
    int64_t lastIndexCount = 1;
    int64_t maxSize = 0;
    int64_t eachNum = 1;
    int64_t eachLoop = 1;
    int64_t eachTail = 0;
    int64_t batchNum = 1;
    int64_t eachBatchNum = 1;
    int64_t lastBatchNum = 1;
    int64_t varDimNum = 1;
    int64_t eachUBIndexRound = 1;
    int64_t eachUBIndexCount = 1;
    int64_t eachUBIndexTail = 0;
    int64_t lastUBIndexRound = 1;
    int64_t lastUBIndexCount = 1;
    int64_t lastUBIndexTail = 0;
    uint64_t workspaceSize = DEFAULT_WORKSPACE_SIZE;
};

bool InplaceIndexAddWithSortedTiling::CheckParam()
{
    if (tilingContext->GetInputShape(INPUT_0) == nullptr || tilingContext->GetInputShape(INPUT_1) == nullptr ||
        tilingContext->GetInputShape(INPUT_2) == nullptr || tilingContext->GetInputShape(INPUT_3) == nullptr ||
        tilingContext->GetInputDesc(INPUT_0) == nullptr || tilingContext->GetRawTilingData() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(tilingContext->GetNodeName(), "inputShape/inputDesc",
                                                 "inputShape or inputDesc is nullptr");
        return false;
    }
    auto inputDtype = tilingContext->GetInputDesc(INPUT_0)->GetDataType();
    auto valueDtype = tilingContext->GetInputDesc(INPUT_1)->GetDataType();
    inputSize = ge::GetSizeByDataType(inputDtype);

    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "var", Ops::Base::ToString(inputDtype).c_str(),
                                  "float16 or bfloat16");
        return false;
    }

    if (inputDtype != valueDtype) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "value", Ops::Base::ToString(valueDtype).c_str(),
                                  Ops::Base::ToString(inputDtype).c_str());
        return false;
    }

    auto sortedIdxDtype = tilingContext->GetInputDesc(INPUT_2)->GetDataType();
    auto posDtype = tilingContext->GetInputDesc(INPUT_3)->GetDataType();
    if (sortedIdxDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "sorted_index",
                                  Ops::Base::ToString(sortedIdxDtype).c_str(), "int32");
        return false;
    }
    if (posDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(tilingContext->GetNodeName(), "pos", Ops::Base::ToString(posDtype).c_str(), "int32");
        return false;
    }
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto alphaShape = tilingContext->GetOptionalInputShape(INPUT_4);
    enableAlpha = (alphaShape == nullptr) ? 0 : 1;

    auto inputDimNum = inputShape.GetDimNum();
    if (inputDimNum > MAX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "dim", std::to_string(inputDimNum).c_str(), "<= 8");
        return false;
    }
    if (inputDimNum != updatesShape.GetDimNum()) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(tilingContext->GetNodeName(), "input/updates",
                                                 "the dimNum of input must equal the dimNum of updates");
        return false;
    }
    const int64_t* ptrDim = tilingContext->GetAttrs()->GetAttrPointer<int64_t>(0);
    dimAttr = *ptrDim;
    dimAttr = dimAttr < 0 ? inputDimNum + dimAttr : dimAttr;
    if (dimAttr != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "dim", std::to_string(dimAttr).c_str(),
                                              "Dim only support 0 on the current version");
        return false;
    }
    for (int64_t idx = 0; idx < static_cast<int64_t>(inputDimNum); ++idx) {
        if (dimAttr != idx) {
            if (inputShape.GetDim(idx) != updatesShape.GetDim(idx)) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    tilingContext->GetNodeName(), "self/updates", std::to_string(inputShape.GetDim(idx)).c_str(),
                    "The size of self must match the size of source at dimension " + std::to_string(idx));
                return false;
            }
        }
    }
    return true;
}

ge::graphStatus InplaceIndexAddWithSortedTiling::Init()
{
    if (tilingContext == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("InplaceIndexAddWithSortedTiling", "context", "nullptr",
                                              "Tiling context is null");
        return ge::GRAPH_FAILED;
    }
    // 运行时优先经 PlatformInfo 查询硬件参数：预编译 / StructKernel 等执行路径下
    // TilingParse 不产生 compileInfo（GetCompileInfo() 为 nullptr，见 poisson_nll_loss
    // 同款处理），因此不能对 compileInfo 硬失败；compileInfo 仅作 PlatformInfo 缺失兜底。
    auto platformInfo = tilingContext->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = static_cast<int32_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatform = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<int64_t>(ubSizePlatform);
    } else {
        auto compileInfo = static_cast<const InplaceIndexAddWithSortedCompileInfo*>(tilingContext->GetCompileInfo());
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, compileInfo);
        coreNum = static_cast<int32_t>(compileInfo->totalCoreNum);
        ubSize = static_cast<int64_t>(compileInfo->ubSizePlatForm);
    }

    if (coreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "coreNum", std::to_string(coreNum).c_str(),
                                              "coreNum must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (ubSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                              "ubSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    workspaceSize = DEFAULT_WORKSPACE_SIZE;
    if (tilingContext->GetWorkspaceSizes(1) == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(tilingContext->GetNodeName(), "workspace",
                                                 "GetWorkspaceSizes returned nullptr");
        return ge::GRAPH_FAILED;
    }

    if (!CheckParam()) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(tilingContext, "Tiling initialized, coreNum=%d, ubSize=%ld, workspaceSize=%lu.", coreNum, ubSize,
            workspaceSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplaceIndexAddWithSortedTiling::RunKernelTiling()
{
    OP_LOGD(tilingContext, "Tiling start.");
    auto inputShape = tilingContext->GetInputShape(INPUT_0)->GetStorageShape();
    auto updatesShape = tilingContext->GetInputShape(INPUT_1)->GetStorageShape();
    auto indicesShape = tilingContext->GetInputShape(INPUT_2)->GetStorageShape();
    auto inputDimNum = inputShape.GetDimNum();
    for (int64_t i = 0; i < static_cast<int64_t>(inputDimNum); ++i) {
        auto dimInput = inputShape.GetDim(i);
        auto dimUpdates = updatesShape.GetDim(i);
        if (i < dimAttr) {
            batchNum *= dimUpdates;
        }
        if (i == dimAttr) {
            varDimNum = dimInput;
        }
        if (i >= dimAttr + 1) {
            updatesOneTime *= dimUpdates;
        }
        inputCount *= dimInput;
        updatesCount *= dimUpdates;
    }
    indicesCount = indicesShape.GetDim(INPUT_0);
    if (inputCount == 0 || updatesCount == 0 || indicesCount == 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "input/updates/indices", "0",
                                                  "shape size cannot equal 0");
        return ge::GRAPH_FAILED;
    }
    if (!processFirstDimTilingData() || !TilingDataSet()) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(tilingContext, "Tiling end.");
    return ge::GRAPH_SUCCESS;
}

bool InplaceIndexAddWithSortedTiling::processFirstDimTilingData()
{
    usedCoreNum = indicesCount < coreNum ? indicesCount : coreNum;
    eachIndexCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
    usedCoreNum = (indicesCount + eachIndexCount - 1) / eachIndexCount;
    lastIndexCount = indicesCount - eachIndexCount * (usedCoreNum - 1);

    // ===== 动态 UB 扣除（替代老代码固定 RESERVED_BUFFER_SIZE）=====
    // wsIndexBufSize = Stage B 一次性搬入 workspace 全表 index 的 UB 大小（按实际 usedCoreNum）
    int64_t wsIndexBufSize = CeilAlign<int64_t>(usedCoreNum * WS_ROWS_PER_CORE * SIZE_OF_INT32, BLOCK_SIZE);
    int64_t fixedUbSize = INDEX_BUFFER_SIZE + wsIndexBufSize;
    if (ubSize <= fixedUbSize) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            tilingContext->GetNodeName(), "ubSize",
            (std::to_string(ubSize) + ", fixedUbSize=" + std::to_string(fixedUbSize)).c_str(),
            "UB size is insufficient");
        return false;
    }
    ubSize -= fixedUbSize;

    maxSize = (ubSize / BUF_CNT) / BLOCK_SIZE * BLOCK_SIZE;
    maxSize /= SIZE_OF_FP32;                                       // 转为 element
    maxSize = (maxSize / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK; // 对齐到 32B block（fp16/bf16 16 elements）
    if (maxSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "maxSize", std::to_string(maxSize).c_str(),
                                              "maxSize must be greater than 0");
        return false;
    }
    eachNum = updatesOneTime;
    eachTail = eachNum;
    if (eachNum > maxSize) {
        eachLoop = (eachNum + maxSize - 1) / maxSize;
        eachNum = maxSize;
        eachTail = updatesOneTime - (eachLoop - 1) * eachNum;
    }
    if (eachIndexCount > UB_INDEX_NUM) {
        eachUBIndexRound = (eachIndexCount + UB_INDEX_NUM - 1) / UB_INDEX_NUM;
        eachUBIndexCount = UB_INDEX_NUM;
        eachUBIndexTail = eachIndexCount - (eachUBIndexRound - 1) * UB_INDEX_NUM;
    } else {
        eachUBIndexRound = 1;
        eachUBIndexCount = eachIndexCount;
        eachUBIndexTail = eachIndexCount;
    }
    if (lastIndexCount > UB_INDEX_NUM) {
        lastUBIndexRound = (lastIndexCount + UB_INDEX_NUM - 1) / UB_INDEX_NUM;
        lastUBIndexCount = UB_INDEX_NUM;
        lastUBIndexTail = lastIndexCount - (lastUBIndexRound - 1) * UB_INDEX_NUM;
    } else {
        lastUBIndexRound = 1;
        lastUBIndexCount = lastIndexCount;
        lastUBIndexTail = lastIndexCount;
    }
    return true;
}

bool InplaceIndexAddWithSortedTiling::TilingDataSet()
{
    tilingData_ = tilingContext->GetTilingData<InplaceIndexAddWithSortedTilingData>();
    tilingData_->usedCoreNum = usedCoreNum;
    tilingData_->enableAlpha = enableAlpha;
    tilingData_->eachIndexCount = eachIndexCount;
    tilingData_->lastIndexCount = lastIndexCount;
    tilingData_->batchCount = batchNum;
    tilingData_->eachBatchNum = eachBatchNum;
    tilingData_->lastBatchNum = lastBatchNum;
    tilingData_->inputCount = inputCount;
    tilingData_->indicesCount = indicesCount;
    tilingData_->updatesCount = updatesCount;
    tilingData_->updatesOneTime = updatesOneTime;
    tilingData_->maxSize = maxSize;
    tilingData_->eachNum = eachNum;
    tilingData_->eachLoop = eachLoop;
    tilingData_->eachTail = eachTail;
    tilingData_->varDimNum = varDimNum;
    tilingData_->eachUBIndexRound = eachUBIndexRound;
    tilingData_->eachUBIndexCount = eachUBIndexCount;
    tilingData_->eachUBIndexTail = eachUBIndexTail;
    tilingData_->lastUBIndexRound = lastUBIndexRound;
    tilingData_->lastUBIndexCount = lastUBIndexCount;
    tilingData_->lastUBIndexTail = lastUBIndexTail;

    TilingDataPrint();

    tilingContext->SetBlockDim(usedCoreNum);
    tilingContext->SetScheduleMode(1);
    tilingContext->SetTilingKey(GET_TPL_TILING_KEY(SORTED_SCH_MODE_DEFAULT));

    // ===== workspace 大小计算（本地变量，不入 TilingData）=====
    // 物理分离布局：段 1 index（int32）+ 段 2 data（fp32），system workspace + aligned user workspace
    int64_t wsIndexSize = CeilAlign<int64_t>(usedCoreNum * WS_ROWS_PER_CORE * SIZE_OF_INT32, BLOCK_SIZE);
    constexpr int64_t wsDataFactor = WS_ROWS_PER_CORE * SIZE_OF_FP32;
    if (updatesOneTime > std::numeric_limits<int64_t>::max() / usedCoreNum / wsDataFactor) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            tilingContext->GetNodeName(), "workspace",
            ("Workspace data size overflow, usedCoreNum=" + std::to_string(usedCoreNum) +
             ", updatesOneTime=" + std::to_string(updatesOneTime))
                .c_str());
        return false;
    }
    int64_t wsDataSize = usedCoreNum * wsDataFactor * updatesOneTime;
    if (wsDataSize > std::numeric_limits<int64_t>::max() - wsIndexSize) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(tilingContext->GetNodeName(), "workspace",
                                                 "Total user workspace size overflow");
        return false;
    }
    int64_t wsTotalSize = wsIndexSize + wsDataSize;
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(tilingContext->GetNodeName(), "workspace",
                                                 "Failed to get workspace size holder");
        return false;
    }
    if (workspaceSize > std::numeric_limits<size_t>::max() - static_cast<size_t>(wsTotalSize)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(tilingContext->GetNodeName(), "workspace",
                                                 "Total workspace size overflow");
        return false;
    }
    currentWorkspace[0] = static_cast<size_t>(workspaceSize) + static_cast<size_t>(wsTotalSize);
    return true;
}

void InplaceIndexAddWithSortedTiling::TilingDataPrint() const
{
    OP_LOGI(tilingContext, "usedCoreNum: %u.", tilingData_->usedCoreNum);
    OP_LOGI(tilingContext, "enableAlpha: %u.", tilingData_->enableAlpha);
    OP_LOGI(tilingContext, "eachIndexCount: %ld.", tilingData_->eachIndexCount);
    OP_LOGI(tilingContext, "lastIndexCount: %ld.", tilingData_->lastIndexCount);
    OP_LOGI(tilingContext, "batchNum: %ld.", tilingData_->batchCount);
    OP_LOGI(tilingContext, "eachBatchNum: %ld.", tilingData_->eachBatchNum);
    OP_LOGI(tilingContext, "lastBatchNum: %ld.", tilingData_->lastBatchNum);
    OP_LOGI(tilingContext, "inputCount: %ld.", tilingData_->inputCount);
    OP_LOGI(tilingContext, "indicesCount: %ld.", tilingData_->indicesCount);
    OP_LOGI(tilingContext, "updatesCount: %ld.", tilingData_->updatesCount);
    OP_LOGI(tilingContext, "updatesOneTime: %ld.", tilingData_->updatesOneTime);
    OP_LOGI(tilingContext, "maxSize: %ld.", tilingData_->maxSize);
    OP_LOGI(tilingContext, "eachNum: %ld.", tilingData_->eachNum);
    OP_LOGI(tilingContext, "eachLoop: %ld.", tilingData_->eachLoop);
    OP_LOGI(tilingContext, "eachTail: %ld.", tilingData_->eachTail);
    OP_LOGI(tilingContext, "varDimNum: %ld.", tilingData_->varDimNum);
    OP_LOGI(tilingContext, "eachUBIndexRound: %ld.", tilingData_->eachUBIndexRound);
    OP_LOGI(tilingContext, "eachUBIndexCount: %ld.", tilingData_->eachUBIndexCount);
    OP_LOGI(tilingContext, "eachUBIndexTail: %ld.", tilingData_->eachUBIndexTail);
    OP_LOGI(tilingContext, "lastUBIndexRound: %ld.", tilingData_->lastUBIndexRound);
    OP_LOGI(tilingContext, "lastUBIndexCount: %ld.", tilingData_->lastUBIndexCount);
    OP_LOGI(tilingContext, "lastUBIndexTail: %ld.", tilingData_->lastUBIndexTail);
}

ge::graphStatus TilingInplaceIndexAddWithSorted(gert::TilingContext* context)
{
    InplaceIndexAddWithSortedTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunKernelTiling();
}

ge::graphStatus TilingPrepareForInplaceIndexAddWithSorted(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForInplaceIndexAddWithSorted start.");
    auto compileInfo = context->GetCompiledInfo<InplaceIndexAddWithSortedCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF((compileInfo->ubSizePlatForm <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSizePlatForm",
                                                      std::to_string(compileInfo->ubSizePlatForm).c_str(),
                                                      "Failed to get ub size"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context, "total_ub_size is %lu.", totalUbSize);
    OP_LOGD(context, "TilingPrepareForInplaceIndexAddWithSorted end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplaceIndexAddWithSorted)
    .Tiling(TilingInplaceIndexAddWithSorted)
    .TilingParse<InplaceIndexAddWithSortedCompileInfo>(TilingPrepareForInplaceIndexAddWithSorted);
} // namespace optiling
