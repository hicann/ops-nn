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
 * \file sorted_sparse_segment_mean_grad_tiling_simt_tiling.cpp
 * \brief
 */

#include "sorted_sparse_segment_mean_grad_tiling_simt_tiling.h"

namespace optiling
{

const static uint64_t LARGE_INNER_LT_INNER_LT_OUTTER_TILING_KEY = 200;
const static uint64_t LARGE_INNER_LT_INNER_GT_OUTTER_TILING_KEY = 201;
const static uint64_t LARGE_INNER_GT_INNER_LT_OUTTER_TILING_KEY = 202;
const static uint64_t LARGE_INNER_GT_INNER_GT_OUTTER_TILING_KEY = 203;
const static uint64_t SMALL_INNER_LT_INNER_LT_OUTTER_TILING_KEY = 300;
const static uint64_t SMALL_INNER_LT_INNER_GT_OUTTER_TILING_KEY = 301;
static constexpr size_t WS_SYS_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr int32_t DCACHE_SIZE = 128 * 1024;
constexpr int32_t INNER_POW2_THRESHOLD = 64;
constexpr int32_t AVG_REDUCE_SIZE_POW2 = 1;
constexpr int32_t SMALL_INNER_THREAD = 64;
constexpr int32_t WORKSPACE_INPUT_NUM = 2;
constexpr int32_t MAX_THREAD_NUM = 512;
constexpr int32_t MAX_INNER_NUM = 512;
constexpr int32_t MIN_PER_CORE_INDICES = 0;
constexpr int64_t MAX_UINT32_NUM = 4294967295;
constexpr uint32_t REMAIN_UB_SIZE = 33280;
constexpr int32_t MAX_CORE_NUM = 64;

int64_t SortedSparseSegmentMeanGradSimtTiling::GetUpPow2(int64_t n)
{
    int64_t res = 1;
    while (res < n) {
        res <<= 1;
    }
    return res;
}

void SortedSparseSegmentMeanGradSimtTiling::CalcSegIndexThreadTiling(int32_t& threadNumX, int32_t& threadNumY, int64_t num)
{
    int64_t multiples = num / inputData.outterSize;
    if (multiples <= 1) {
        threadNumX = MAX_THREAD_NUM;
        threadNumY = 1;
    } else if (multiples >= MAX_THREAD_NUM) {
        threadNumX = 1;
        threadNumY = MAX_THREAD_NUM;
    } else {
        // 向上对齐到2的倍数； segmentNum 相当于总循环数量， outterSize 相当于x的线程数量
        multiples = GetUpPow2(multiples);
        threadNumY = static_cast<int32_t>(multiples);
        threadNumX = MAX_THREAD_NUM / static_cast<int32_t>(multiples);
    }
}

void SortedSparseSegmentMeanGradSimtTiling::CalcThreadTiling()
{
    int64_t innerPow2 = GetUpPow2(inputData.innerSize);
    int64_t avgReduceSize = (inputData.outterSize + inputData.outputDim0 - 1) / inputData.outputDim0;
    int64_t avgReduceSizePow2 = GetUpPow2(avgReduceSize);
    if (innerPow2 < INNER_POW2_THRESHOLD && avgReduceSizePow2 > AVG_REDUCE_SIZE_POW2) {
        isSmallInner_ = true;
        threadNum_ = SMALL_INNER_THREAD;
        threadNumX_ = innerPow2;
        threadNumY_ = std::min(avgReduceSizePow2, threadNum_ / innerPow2);
    } else {
        if (innerCore_ > 1) {
            threadNumX_ = std::min(static_cast<int64_t>(MAX_INNER_NUM), perCoreInnerSize_ + 1);
        } else {
            threadNumX_ = std::min(static_cast<int64_t>(MAX_INNER_NUM), inputData.innerSize);
        }
        threadNumY_ = threadNum_ / threadNumX_;
        if (perCoreIndicesNum_ > MIN_PER_CORE_INDICES) {
            threadNumY_ = std::min(threadNumY_, perCoreIndicesNum_);
        } else {
            threadNumY_ = 1;
        }
    }
}

void SortedSparseSegmentMeanGradSimtTiling::CalcBlockTiling()
{
    perCoreIndicesNum_ = inputData.outputDim0 / hardwareData.coreNum;
    resIndicesNum_ = inputData.outputDim0 - perCoreIndicesNum_ * hardwareData.coreNum;
    needCoreNum_ = resIndicesNum_;
    if (perCoreIndicesNum_ > 0) {
        needCoreNum_ = hardwareData.coreNum;
    }
    if (needCoreNum_ < hardwareData.coreNum && inputData.innerSize > MAX_CORE_NUM) {
        innerCore_ = hardwareData.coreNum / needCoreNum_;
        if (innerCore_ < 1) {
            return;
        }
        perCoreInnerSize_ = inputData.innerSize / innerCore_;
        resCoreInnerSize_ = inputData.innerSize - innerCore_ * perCoreInnerSize_;
        needCoreNum_ *= innerCore_;
    }
}

ge::graphStatus SortedSparseSegmentMeanGradSimtTiling::DoOpTiling() 
{
    CalcBlockTiling();
    CalcThreadTiling();
    CalcSegIndexThreadTiling(segThreadNumX_, segThreadNumY_, inputData.segmentNum);
    CalcSegIndexThreadTiling(indexThreadNumX_, indexThreadNumY_, static_cast<int64_t>(inputData.outputDim0));

    SortedSparseSegmentMeanGradSimtTilingData *tilingData = context_->GetTilingData<SortedSparseSegmentMeanGradSimtTilingData>();
    tilingData->needCoreNum = needCoreNum_;
    tilingData->innerSize = inputData.innerSize;
    tilingData->segmentNum = inputData.segmentNum;
    tilingData->outterSize = inputData.outterSize;
    tilingData->threadNumX = threadNumX_;
    tilingData->threadNumY = threadNumY_;
    tilingData->segThreadNumX = segThreadNumX_;
    tilingData->segThreadNumY = segThreadNumY_;
    tilingData->indexThreadNumX = indexThreadNumX_;
    tilingData->indexThreadNumY = indexThreadNumY_;
    tilingData->perCoreIndicesNum = perCoreIndicesNum_;
    tilingData->resIndicesNum = resIndicesNum_;
    tilingData->outputDim0 = inputData.outputDim0;
    tilingData->innerCore = innerCore_;
    tilingData->perCoreInnerSize = perCoreInnerSize_;
    tilingData->resCoreInnerSize = resCoreInnerSize_;
    
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

// 当前实现为simt模板，后续如新增simd模板需要在IsCapable进行分支判断
bool SortedSparseSegmentMeanGradSimtTiling::IsCapable()
{
    return true;
}


uint64_t SortedSparseSegmentMeanGradSimtTiling::GetTilingKey() const {
    if (isSmallInner_) {
        if (inputData.outterSize < MAX_UINT32_NUM) {
            return SMALL_INNER_LT_INNER_LT_OUTTER_TILING_KEY;
        } else {
            return SMALL_INNER_LT_INNER_GT_OUTTER_TILING_KEY;
        }
    }
    if (inputData.innerSize < MAX_UINT32_NUM && inputData.outterSize < MAX_UINT32_NUM) {
        return LARGE_INNER_LT_INNER_LT_OUTTER_TILING_KEY;
    } else if (inputData.innerSize < MAX_UINT32_NUM && inputData.outterSize >= MAX_UINT32_NUM ) {
        return LARGE_INNER_LT_INNER_GT_OUTTER_TILING_KEY;
    } else if (inputData.innerSize >= MAX_UINT32_NUM && inputData.outterSize < MAX_UINT32_NUM) {
        return LARGE_INNER_GT_INNER_LT_OUTTER_TILING_KEY;
    } else {
        return LARGE_INNER_GT_INNER_GT_OUTTER_TILING_KEY;
    }
}


ge::graphStatus SortedSparseSegmentMeanGradSimtTiling::GetWorkspaceSize()
{
    auto sysWorkspace = WS_SYS_SIZE; 
    sysWorkspace += ((inputData.segmentNum + 1) * WORKSPACE_INPUT_NUM + inputData.outputDim0 + 1) * sizeof(int64_t);
    
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradSimtTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "SortedSparseSegmentMeanGradSimtTiling PostTiling enter.");
    context_->SetBlockDim(needCoreNum_);
    context_->SetScheduleMode(1);
    OP_TILING_CHECK(
        static_cast<uint32_t>(hardwareData.ubSize - DCACHE_SIZE) < REMAIN_UB_SIZE,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "SortedSparseSegmentMeanGradSimtTiling: remian UB size is insufficient."),
        return ge::GRAPH_FAILED);
    context_->SetLocalMemorySize(static_cast<uint32_t>(hardwareData.ubSize - DCACHE_SIZE));

    return ge::GRAPH_SUCCESS;
}

void SortedSparseSegmentMeanGradSimtTiling::PrintTilingData() const
{
    OP_LOGD("SortedSparseSegmentMeanGradSimt", "[SortedSparseSegmentMeanGradSimt] PrintTilingData start running");
    std::ostringstream info;
    SortedSparseSegmentMeanGradSimtTilingData *tilingData = context_->GetTilingData<SortedSparseSegmentMeanGradSimtTilingData>();

    info << "needCoreNum: " << tilingData->needCoreNum << ", ";
    info << "innerSize: " << tilingData->innerSize << ", ";
    info << "segmentNum: " << tilingData->segmentNum << ", ";
    info << "outterSize: " << tilingData->outterSize << ", ";
    info << "threadNumX: " << tilingData->threadNumX << ", ";
    info << "threadNumY: " << tilingData->threadNumY << ", ";
    info << "segThreadNumX: " << tilingData->segThreadNumX << ", ";
    info << "segThreadNumY: " << tilingData->segThreadNumY << ", ";
    info << "indexThreadNumX: " << tilingData->indexThreadNumX << ", ";
    info << "indexThreadNumY: " << tilingData->indexThreadNumY << ", ";
    info << "perCoreIndicesNum: " << tilingData->perCoreIndicesNum << ", ";
    info << "resIndicesNum: " << tilingData->resIndicesNum << ", ";
    info << "innerCore: " << tilingData->innerCore << ", ";
    info << "perCoreInnerSize: " << tilingData->perCoreInnerSize << ", ";
    info << "resCoreInnerSize: " << tilingData->resCoreInnerSize << ", ";
    info << "outputDim0: " << tilingData->outputDim0;
    OP_LOGI("SortedSparseSegmentMeanGradSimt", "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(SortedSparseSegmentMeanGrad, SortedSparseSegmentMeanGradSimtTiling, 0);

}  // namespace optiling