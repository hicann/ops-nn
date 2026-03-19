/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_segment_mean_tiling_full_load_tiling.cpp
 * \brief
 */
#include "sparse_segment_mean_tiling_full_load_tiling.h"

namespace optiling {

static constexpr int64_t WS_SYS_SIZE = static_cast<size_t>(16 * 1024 * 1024);
static constexpr uint64_t FULL_LOAD_LARGE_INNER_TILING_KEY = 10;
static constexpr uint64_t FULL_LOAD_SMALL_INNER_TILING_KEY = 11;
static constexpr int64_t MIN_MUL_SIZE = 1024;
static constexpr int64_t BINARY_ADD_TMP_BUFFER_SIZE = 256;
static constexpr int64_t FULL_LOAD_SIZE = 128;
static constexpr int64_t INNER_POW2_THRESHOLD = 64;
static constexpr int64_t MIN_THRESHOLD = 64;
static constexpr int64_t MAX_THRESHOLD = 2048;
static constexpr int64_t HALF_THRESHOLD = 1024;
static constexpr int64_t SIMT_BINARY_ADD_TYPE = 0;
static constexpr int64_t SIMT_LOOP_ADD_TYPE = 1;
static constexpr int64_t SIMT_NORMAL_ADD_TYPE = 2;
static constexpr int64_t FULL_LOAD_DCACHE_SIZE = 32 * 1024;
static constexpr int64_t AGLIN_VALUE = 32;
static constexpr int64_t MAX_THREAD_BLOCKS = 16;
static constexpr int64_t MIN_SIZE = 12000;
static constexpr int64_t MAX_SIZE = 36000;
static constexpr int64_t TOP_SIZE = 36000;
static constexpr int64_t Duplicate_NUM1 = 40;
static constexpr int64_t Duplicate_NUM2 = 70;
static constexpr int64_t NEED_NUM1 = 3;
static constexpr int64_t NEED_NUM2 = 2;


bool SparseSegmentMeanFullLoadTiling::IsCapable()
{
    int64_t allCoreNum = std::min(hardwareData.coreNum, inputData.outterSize);
    fullLoadData.normalCoreIndicesNum = Ops::Base::CeilDiv(inputData.outterSize, allCoreNum);
    fullLoadData.usedCoreNum = Ops::Base::CeilDiv(inputData.outterSize, fullLoadData.normalCoreIndicesNum);

    fullLoadData.xBufferSize =
        Ops::Base::CeilAlign(inputData.gatherSize * inputData.innerSize * inputData.inputBytes, AGLIN_VALUE);
    fullLoadData.indicesBufferSize = Ops::Base::CeilAlign(inputData.outterSize * inputData.indicesBytes, AGLIN_VALUE);
    // 单核分配的indices个数需要大于等于x的索引个数;
    if (inputData.gatherSize > inputData.outterSize / fullLoadData.usedCoreNum) {
        return false;
    }
    // innerSize和outterSize的值需要满足准入条件;
    if (inputData.innerSize > MIN_SIZE &&
        (inputData.outterSize / inputData.gatherSize < Duplicate_NUM1 ||
         inputData.outterSize / fullLoadData.usedCoreNum > NEED_NUM1 || inputData.innerSize >= MAX_SIZE || inputData.innerSize <= MIN_SIZE) &&
        (inputData.outterSize / inputData.gatherSize < Duplicate_NUM2 ||
         inputData.outterSize / fullLoadData.usedCoreNum > NEED_NUM2 || inputData.innerSize >= TOP_SIZE || inputData.innerSize < MAX_SIZE)) {
        return false;
    }
    // BufferSize的总大小需要小于ubSize;
    if (fullLoadData.xBufferSize + fullLoadData.indicesBufferSize + FULL_LOAD_DCACHE_SIZE + BINARY_ADD_TMP_BUFFER_SIZE >
        hardwareData.ubSize) {
        return false;
    }
    return true;
}

uint64_t SparseSegmentMeanFullLoadTiling::GetTilingKey() const
{
    if(fullLoadData.useSimtMode == SIMT_BINARY_ADD_TYPE) {
        return FULL_LOAD_SMALL_INNER_TILING_KEY;
    }
    return FULL_LOAD_LARGE_INNER_TILING_KEY;
}

int64_t SparseSegmentMeanFullLoadTiling::GetUpPow2(int64_t n)
{
    if (n <= 1) {
        return 1;
    }
    int64_t result = 1;
    while (result < n) {
        result <<= 1;
    }
    return result;
}

void SparseSegmentMeanFullLoadTiling::ThreadTiling()
{
    int64_t innerPow2 = GetUpPow2(inputData.innerSize);
    int64_t avgReduceSize = (inputData.outterSize + inputData.segmentNum - 1) / inputData.segmentNum;
    int64_t avgReduceSizePow2 = GetUpPow2(avgReduceSize);
    int64_t threadNum = MAX_THRESHOLD;

    if (innerPow2 < INNER_POW2_THRESHOLD && avgReduceSizePow2 > 1) {
        int64_t oneCoreMaxSegNum = (inputData.segmentNum + hardwareData.coreNum - 1) / hardwareData.coreNum;
        if (oneCoreMaxSegNum > MAX_THREAD_BLOCKS) {
            // 16组线程一次处理不完一个核的segmentNum
            specialBlockTiling_ = true;
            int64_t oneCoreMaxSegNumUp16 = Ops::Base::CeilAlign(oneCoreMaxSegNum, MAX_THREAD_BLOCKS); // 正常核处理的segmentNum需要与16向上对齐
            if (inputData.segmentNum % oneCoreMaxSegNumUp16 > MAX_THREAD_BLOCKS) {
                // 剩下的seg数量大于16，还需要至少2个核处理
                int64_t tmpCoreNum = hardwareData.coreNum - 1;
                int64_t tmpOneCoreMaxSegNum = (inputData.segmentNum + tmpCoreNum - 1) / tmpCoreNum;
                int64_t tmpOneCoreMaxSegNumUp16 = Ops::Base::CeilAlign(tmpOneCoreMaxSegNum, MAX_THREAD_BLOCKS);
                int64_t resSegNum = inputData.segmentNum % tmpOneCoreMaxSegNumUp16;
                normalCoreSegmentNum_ = tmpOneCoreMaxSegNumUp16;
                secondToLastCoreSegmentNum_ = Ops::Base::FloorAlign(resSegNum, MAX_THREAD_BLOCKS);
                lastCoreSegmentNum_ = resSegNum - secondToLastCoreSegmentNum_;
                needCoreNum_ = inputData.segmentNum / tmpOneCoreMaxSegNumUp16 + 2;
            } else {
                // 剩下的seg数量小于等于16，只需还要1个核处理剩余seg
                needCoreNum_ = inputData.segmentNum / oneCoreMaxSegNumUp16 + 1;
                normalCoreSegmentNum_ = oneCoreMaxSegNumUp16;
                lastCoreSegmentNum_ = inputData.segmentNum % oneCoreMaxSegNumUp16;
            }
        }
        threadNum = MIN_THRESHOLD;
        fullLoadData.threadNumX = innerPow2;
        fullLoadData.threadNumY = std::min(avgReduceSizePow2, threadNum / fullLoadData.threadNumX);
        fullLoadData.useSimtMode = SIMT_BINARY_ADD_TYPE;
    } else if (inputData.innerSize > MAX_THRESHOLD) {
        int64_t loopNum = (inputData.innerSize + MAX_THRESHOLD - 1) / MAX_THRESHOLD;
        fullLoadData.threadNumX = (inputData.innerSize + loopNum - 1) / loopNum;
        fullLoadData.threadNumY = threadNum / fullLoadData.threadNumX;
        fullLoadData.useSimtMode = SIMT_LOOP_ADD_TYPE;
    } else {
        fullLoadData.useSimtMode = SIMT_NORMAL_ADD_TYPE;
        fullLoadData.threadNumX = inputData.innerSize;
        if (fullLoadData.threadNumX * avgReduceSize <= HALF_THRESHOLD) {
            threadNum = HALF_THRESHOLD;
        }
        fullLoadData.threadNumY = threadNum / fullLoadData.threadNumX;
    }
}

void SparseSegmentMeanFullLoadTiling::DoBlockTiling()
{
    if (!specialBlockTiling_) {
        fullLoadData.perCoreSegmentNum = inputData.segmentNum / hardwareData.coreNum;
        fullLoadData.resSegmentNum = inputData.segmentNum - fullLoadData.perCoreSegmentNum * hardwareData.coreNum;
        needCoreNum_ = fullLoadData.resSegmentNum;
        if (fullLoadData.perCoreSegmentNum > 0) {
            needCoreNum_ = hardwareData.coreNum;
        }
    }
}

void SparseSegmentMeanFullLoadTiling::PrinttilingData() const
{
    OP_LOGD("SparseSegmentMeanFullLoad", "[SparseSegmentMeanFullLoad] PrinttilingData start running");

    std::ostringstream info;
    SparseSegmentMeanFullLoadTilingData* tilingData = context_->GetTilingData<SparseSegmentMeanFullLoadTilingData>();

    info << "xBufferSize: " << tilingData->xBufferSize << std::endl;
    info << "indicesBufferSize: " << tilingData->indicesBufferSize << std::endl;

    info << "useSimtMode: " << fullLoadData.useSimtMode << std::endl;
    info << "threadNumX_: " << tilingData->threadNumX << std::endl;
    info << "threadNumY_: " << tilingData->threadNumY << std::endl;

    info << "perCoreSegmentNum: " << tilingData->perCoreSegmentNum << std::endl;
    info << "resSegmentNum: " << tilingData->resSegmentNum << std::endl;

    info << "innerSize: " << tilingData->innerSize << std::endl;
    info << "gatherSize: " << tilingData->gatherSize << std::endl;
    info << "segmentNum: " << tilingData->segmentNum << std::endl;
    info << "outterSize: " << tilingData->outterSize << std::endl;

    info << "needCoreNum: " << tilingData->needCoreNum << std::endl;
    info << "normalCoreSegmentNum: " << tilingData->normalCoreSegmentNum << std::endl;
    info << "secondToLastCoreSegmentNum: " << tilingData->secondToLastCoreSegmentNum << std::endl;
    info << "lastCoreSegmentNum: " << tilingData->lastCoreSegmentNum << std::endl;
    info << "specialBlockTiling: " << tilingData->specialBlockTiling << std::endl;

    OP_LOGI("Sparse_Segment_Mean_Full_Load", "%s", info.str().c_str());
}

void SparseSegmentMeanFullLoadTiling::SetTilingData()
{
    SparseSegmentMeanFullLoadTilingData* tilingData = context_->GetTilingData<SparseSegmentMeanFullLoadTilingData>();
    tilingData->innerSize = inputData.innerSize;
    tilingData->gatherSize = inputData.gatherSize;
    tilingData->segmentNum = inputData.segmentNum;
    tilingData->outterSize = inputData.outterSize;

    tilingData->threadNumX = fullLoadData.threadNumX;
    tilingData->threadNumY = fullLoadData.threadNumY;

    tilingData->perCoreSegmentNum = fullLoadData.perCoreSegmentNum;
    tilingData->resSegmentNum = fullLoadData.resSegmentNum;

    tilingData->xBufferSize = fullLoadData.xBufferSize;
    tilingData->indicesBufferSize = fullLoadData.indicesBufferSize;

    tilingData->needCoreNum = needCoreNum_;
    tilingData->normalCoreSegmentNum = normalCoreSegmentNum_;
    tilingData->secondToLastCoreSegmentNum = secondToLastCoreSegmentNum_;
    tilingData->lastCoreSegmentNum = lastCoreSegmentNum_;
    tilingData->specialBlockTiling = specialBlockTiling_;
}

ge::graphStatus SparseSegmentMeanFullLoadTiling::DoOpTiling()
{
    ThreadTiling();
    DoBlockTiling();
    SetTilingData();
    PrinttilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSegmentMeanFullLoadTiling::GetWorkspaceSize()
{
    auto sysWorkspace = WS_SYS_SIZE;
    sysWorkspace += (inputData.segmentNum + 1) * sizeof(int64_t);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSegmentMeanFullLoadTiling::PostTiling()
{
    context_->SetBlockDim(needCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(static_cast<uint32_t>(hardwareData.ubSize - FULL_LOAD_DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SparseSegmentMean, SparseSegmentMeanFullLoadTiling, 0);

} // namespace optiling