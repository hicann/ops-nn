/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_segment_mean_tiling_simt_tiling.cpp
 * \brief
 */

#include "sparse_segment_mean_tiling_simt_tiling.h"

namespace optiling
{

const static uint64_t LARGE_INNER_TILING_KEY = 200;
const static uint64_t SMALL_INNER_TILING_KEY = 300;
const static uint64_t SIMT_LOOP_TILING_KEY = 400;
static constexpr size_t WS_SYS_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr int32_t DCACHE_SIZE = 128 * 1024;
constexpr int32_t INNER_POW2_THRESHOLD = 64;
constexpr int32_t AVG_REDUCE_SIZE_POW2 = 1;
constexpr int32_t SMALL_INNER_THREAD = 64;
static constexpr int64_t MAX_THREAD_NUMS = 2048;
static constexpr int64_t SMALL_THREAD_NUMS = 1024;
static constexpr int64_t SEGMENTNUM_THRES1 = 200;
static constexpr int64_t SEGMENTNUM_THRES2 = 400;
static constexpr int64_t SEGMENTNUM_THRES3 = 800;
static constexpr int64_t SEGMENTNUM_THRES4 = 1600;
static constexpr int64_t SEGMENTNUM_THRES5 = 1000;
static constexpr int64_t INNER_THRES1 = 3300;
static constexpr int64_t INNER_THRES2 = 4000;
static constexpr int64_t INNER_THRES3 = 4096;
static constexpr int64_t INNER_THRES4 = 5500;
static constexpr int64_t INNER_THRES5 = 6000;
static constexpr int64_t INNER_THRES6 = 6500;
static constexpr int64_t INNER_THRES7 = 4500;
static constexpr int64_t SIMD_INNER_THRES1 = 320;
static constexpr int64_t SIMD_INNER_THRES2 = 540;
static constexpr int64_t SIMD_INNER_THRES3 = 1280;
static constexpr int64_t SIMD_INNER_THRES4 = 10000;
static constexpr int64_t SIMD_INNER_THRES5 = 16;
static constexpr int64_t NUM_TWO = 2;
static constexpr int64_t NUM_FOUR = 4;
static constexpr int64_t NUM_SEVEN = 7;
static constexpr int64_t NUM_EIGHT = 8;
static constexpr int64_t NUM_TEN = 10;
static constexpr int64_t NUM_TWENTY_FIVE = 25;
static constexpr int64_t NUM_TWENTY_SEVEN = 27;
static constexpr int64_t MAX_THREAD_BLOCKS = 16;

int64_t SparseSegmentMeanSimtTiling::GetUpPow2(int64_t n)
{
    int64_t res = 1;
    while (res < n) {
        res <<= 1;
    }
    return res;
}


void SparseSegmentMeanSimtTiling::CalcThreadTiling()
{
    int64_t innerPow2 = GetUpPow2(inputData.innerSize);
    int64_t avgReduceSize = (inputData.outterSize + inputData.segmentNum - 1) / inputData.segmentNum;
    int64_t avgReduceSizePow2 = GetUpPow2(avgReduceSize);
    if (innerPow2 < INNER_POW2_THRESHOLD && avgReduceSizePow2 > AVG_REDUCE_SIZE_POW2) {
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
        isSmallInner_ = true;
        threadNum_ = SMALL_INNER_THREAD;
        threadNumX_ = innerPow2;
        threadNumY_ = std::min(avgReduceSizePow2, threadNum_ / innerPow2);
    } else if (inputData.innerSize > MAX_THREAD_NUMS) {
        isSimtLoop_ = true;
        int64_t loopNum = (inputData.innerSize + MAX_THREAD_NUMS - 1) / MAX_THREAD_NUMS;
        threadNumX_ = (inputData.innerSize + loopNum - 1) / loopNum;
        threadNumY_ = threadNum_ / threadNumX_;
    } else {
        threadNumX_ = inputData.innerSize;
        if (threadNumX_ * avgReduceSize <= threadNum_ / NUM_TWO) {
            threadNum_ = threadNum_ / NUM_TWO;
        }
        threadNumY_ = threadNum_ / threadNumX_;
    }
}


void SparseSegmentMeanSimtTiling::CalcBlockTiling()
{
    if (!specialBlockTiling_) {
        perCoreSegmentNum_ = inputData.segmentNum / hardwareData.coreNum;
        resSegmentNum_ = inputData.segmentNum - perCoreSegmentNum_ * hardwareData.coreNum;
        needCoreNum_ = resSegmentNum_;
        if (perCoreSegmentNum_ > 0) {
            needCoreNum_ = hardwareData.coreNum;
        }
    }
}


ge::graphStatus SparseSegmentMeanSimtTiling::DoOpTiling() 
{
    CalcThreadTiling();
    CalcBlockTiling();

    SparseSegmentMeanSimtTilingData *tilingData = context_->GetTilingData<SparseSegmentMeanSimtTilingData>();
    tilingData->needCoreNum = needCoreNum_;
    tilingData->innerSize = inputData.innerSize;
    tilingData->gatherSize = inputData.gatherSize;
    tilingData->segmentNum = inputData.segmentNum;
    tilingData->outterSize = inputData.outterSize;
    tilingData->threadNumX = threadNumX_;
    tilingData->threadNumY = threadNumY_;
    tilingData->perCoreSegmentNum = perCoreSegmentNum_;
    tilingData->resSegmentNum = resSegmentNum_;

    tilingData->normalCoreSegmentNum = normalCoreSegmentNum_;
    tilingData->secondToLastCoreSegmentNum = secondToLastCoreSegmentNum_;
    tilingData->lastCoreSegmentNum = lastCoreSegmentNum_;
    tilingData->specialBlockTiling = specialBlockTiling_;
    
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}


bool SparseSegmentMeanSimtTiling::IsCapable()
{
    // 总体规律：inner轴越大simd越好；segmentNum越大simd越好；平均每组的id数越多simd越好
    bool isSimt = inputData.innerSize <= MAX_THREAD_NUMS ||
                  (inputData.innerSize <= INNER_THRES7 && inputData.segmentNum <= SEGMENTNUM_THRES5 && inputData.outterSize / inputData.segmentNum <= NUM_TWO) ||
                  (inputData.innerSize <= INNER_THRES1 && inputData.segmentNum <= SEGMENTNUM_THRES4 && inputData.outterSize / inputData.segmentNum <= NUM_TWO) ||
                  (inputData.innerSize <= INNER_THRES4 && inputData.segmentNum <= SEGMENTNUM_THRES1 && inputData.outterSize / inputData.segmentNum <= NUM_FOUR) ||
                  (inputData.innerSize <= INNER_THRES5 && inputData.segmentNum <= SEGMENTNUM_THRES2 && inputData.outterSize / inputData.segmentNum <= NUM_FOUR) ||
                  (inputData.innerSize <= INNER_THRES2 && inputData.segmentNum <= SEGMENTNUM_THRES3 && inputData.outterSize / inputData.segmentNum <= NUM_FOUR) ||
                  (inputData.innerSize <= INNER_THRES6 && inputData.segmentNum <= SEGMENTNUM_THRES1 && inputData.outterSize / inputData.segmentNum <= NUM_EIGHT) ||
                  (inputData.innerSize <= INNER_THRES3 && inputData.segmentNum <= SEGMENTNUM_THRES2 && inputData.outterSize / inputData.segmentNum <= NUM_EIGHT);

    // 尾轴小但是simd更好的场景：(1)尾轴小但是segmentNum和平均每组的id数很大；(2)segmentNum足够小导致simt一半核都开不满
    bool isSimd = (inputData.innerSize >= SMALL_THREAD_NUMS && inputData.innerSize <= MAX_THREAD_NUMS && inputData.segmentNum >= SIMD_INNER_THRES1 && inputData.outterSize / inputData.segmentNum >= NUM_TWENTY_SEVEN) ||
                  (inputData.innerSize >= SMALL_THREAD_NUMS && inputData.innerSize <= MAX_THREAD_NUMS && inputData.segmentNum >= SIMD_INNER_THRES2 && inputData.outterSize / inputData.segmentNum >= NUM_TEN) ||
                  (inputData.innerSize >= SMALL_THREAD_NUMS && inputData.innerSize <= MAX_THREAD_NUMS && inputData.segmentNum >= SIMD_INNER_THRES3 && inputData.outterSize / inputData.segmentNum >= NUM_SEVEN) ||
                  (inputData.innerSize >= SMALL_THREAD_NUMS && inputData.innerSize <= MAX_THREAD_NUMS && inputData.segmentNum <= SIMD_INNER_THRES5 && inputData.outterSize / inputData.segmentNum >= NUM_TWENTY_FIVE) ||
                  (inputData.innerSize >= SMALL_THREAD_NUMS && inputData.innerSize <= MAX_THREAD_NUMS && inputData.segmentNum >= SIMD_INNER_THRES4);
    
    return isSimt && (!isSimd);
}


uint64_t SparseSegmentMeanSimtTiling::GetTilingKey() const {
    if (isSmallInner_) {
        return SMALL_INNER_TILING_KEY;
    } else if (isSimtLoop_) {
        return SIMT_LOOP_TILING_KEY;
    }
    return LARGE_INNER_TILING_KEY;
}


ge::graphStatus SparseSegmentMeanSimtTiling::GetWorkspaceSize()
{
    auto sysWorkspace = WS_SYS_SIZE; 
    sysWorkspace += (inputData.segmentNum + 1) * sizeof(int64_t);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus SparseSegmentMeanSimtTiling::PostTiling()
{
    OP_LOGD(context_->GetNodeName(), "SparseSegmentMeanSimtTiling PostTiling enter.");
    context_->SetBlockDim(needCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(static_cast<uint32_t>(hardwareData.ubSize - DCACHE_SIZE));

    return ge::GRAPH_SUCCESS;
}


void SparseSegmentMeanSimtTiling::PrintTilingData() const
{
    OP_LOGD("SparseSegmentMeanSimt", "[SparseSegmentMeanSimt] PrintTilingData start running");
    std::ostringstream info;
    SparseSegmentMeanSimtTilingData *tilingData = context_->GetTilingData<SparseSegmentMeanSimtTilingData>();

    info << "needCoreNum: " << tilingData->needCoreNum << ", ";
    info << "innerSize: " << tilingData->innerSize << ", ";
    info << "gatherSize: " << tilingData->gatherSize << ", ";
    info << "segmentNum: " << tilingData->segmentNum << ", ";
    info << "outterSize: " << tilingData->outterSize << ", ";
    info << "threadNumX: " << tilingData->threadNumX << ", ";
    info << "threadNumY: " << tilingData->threadNumY << ", ";
    info << "perCoreSegmentNum: " << tilingData->perCoreSegmentNum << ", ";
    info << "resSegmentNum: " << tilingData->resSegmentNum << ", ";

    info << "normalCoreSegmentNum: " << tilingData->normalCoreSegmentNum << ", ";
    info << "secondToLastCoreSegmentNum: " << tilingData->secondToLastCoreSegmentNum << ", ";
    info << "lastCoreSegmentNum: " << tilingData->lastCoreSegmentNum << ", ";
    info << "specialBlockTiling: " << tilingData->specialBlockTiling;

    OP_LOGI("SparseSegmentMeanSimt", "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(SparseSegmentMean, SparseSegmentMeanSimtTiling, 1);

}  // namespace optiling