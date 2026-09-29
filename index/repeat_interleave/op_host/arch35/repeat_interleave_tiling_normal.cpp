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
 * \file repeat_interleave_tiling_normal.cpp
 * \brief
 */

#include "repeat_interleave_tiling_normal.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t ALIGN_COUNT = 32;
static constexpr int64_t MIN_CP_THRESHOLD = 2048;
static constexpr int64_t MAX_THREAD_NUM = 2048;
static constexpr int64_t CUMSUMUB_REPEATS_THRESHOLD = 8192;
static constexpr int64_t MIN_SHAPE_THRESHOLD = 64;
static constexpr int64_t MIN_SUM_REPEATS = 256;
static constexpr int64_t SIMT_CP_THRESHOLD = 128;
static constexpr int64_t SIMT_SPLIT_REPEAT_CP_THRESHOLD = 64;
static constexpr int64_t SIMT_SPLIT_REPEAT_CP_THRESHOLD_SMALL = 32;
static constexpr int64_t SIMT_SPLIT_REPEAT_REPEATCORE_THRESHOLD = 1562; // batchNum * mergedDim_[1] > 100000
static constexpr int64_t BATCH_TILING_REPEAT_SCALAR = 101;
static constexpr int64_t BATCH_TILING_REPEAT_TENSOR = 102;
static constexpr int64_t BATCH_TILING_SPLIT_REPEATS_SHAPE_INT32 = 201;
static constexpr int64_t BATCH_TILING_SPLIT_REPEATS_SHAPE_INT64 = 202;
static constexpr int64_t MIN_TOTAL_REPEATS_SUM = 1024;
static constexpr int64_t REDUCE_SUM_BUFFER = 32;
static constexpr int64_t INT32_MAX_LIM = 2147483647;
constexpr int32_t DCACHE_SIZE = 128 * 1024;
const static uint32_t SIMT_ADDR_KEY = 100U;
const static uint32_t SIMT_CUMSUMINUB_KEY = 10000U;
static constexpr uint32_t CACHELINE_SIZE = 128;

template <typename T1, typename T2>
inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

bool RepeatInterleaveTilingKernelNorm::IsCapable() { return true; }
void RepeatInterleaveTilingKernelNorm::SplitRepeatsShape()
{
    int64_t ratio = totalCoreNum_ / usedCoreNumBefore_;
    int64_t repeatsShape = repeatShape_.GetDim(0);
    normalRepeatsCount_ = CeilDiv(repeatsShape, ratio);
    repeatsSlice_ = CeilDiv(repeatsShape, normalRepeatsCount_);
    usedCoreNum_ = usedCoreNumBefore_ * repeatsSlice_;
    tailRepeatsCount_ = repeatsShape - normalRepeatsCount_ * (repeatsSlice_ - 1);
    isSplitShape_ = true;
    return;
}
void RepeatInterleaveTilingKernelNorm::SplitCP()
{
    int64_t ratio = totalCoreNum_ / usedCoreNum_;
    int64_t cpAxis = 2;
    int64_t cpDim = mergedDim_[cpAxis];
    int64_t cpMin = MIN_CP_THRESHOLD / ge::GetSizeByDataType(inputDtype_);
    normalCP_ = std::max(CeilDiv(cpDim, ratio), cpMin);
    cpSlice_ = CeilDiv(cpDim, normalCP_);
    usedCoreNum_ = usedCoreNum_ * cpSlice_;
    tailCP_ = cpDim - normalCP_ * (cpSlice_ - 1);
    isSplitCP_ = 1;
    return;
}
void RepeatInterleaveTilingKernelNorm::GetUbFactor()
{
    int64_t batchNum = mergedDim_[0];
    eachCoreBatchCount_ = CeilDiv(batchNum, totalCoreNum_);
    usedCoreNum_ = CeilDiv(batchNum, eachCoreBatchCount_);
    usedCoreNumBefore_ = usedCoreNum_;
    tailCoreBatchCount_ = batchNum - eachCoreBatchCount_ * (usedCoreNum_ - 1);
    totalRepeatSum_ = yShape_.GetDim(axis_);
    // 核数不满一半，继续切分CP轴
    int64_t cpAxis = 2;
    if ((usedCoreNum_ < totalCoreNum_ / DOUBLE) &&
        (mergedDim_[cpAxis] * ge::GetSizeByDataType(inputDtype_) > MIN_CP_THRESHOLD)) {
        SplitCP();
    }
    // CP轴切分后，核数还是不满，则切分repeats shape
    if ((usedCoreNum_ < totalCoreNum_ / DOUBLE) &&
        (repeatShape_.GetDim(0) > MIN_SHAPE_THRESHOLD || (mergedDim_[cpAxis] == 1 && repeatShape_.GetDim(0) != 1)) &&
        (totalRepeatSum_ > MIN_TOTAL_REPEATS_SUM)) {
        SplitRepeatsShape();
    }

    if (repeatShape_.GetDimNum() == 0 || repeatShape_.GetDim(0) == 1) {
        repeatsCount_ = yShape_.GetDim(axis_) / inputShape_.GetDim(axis_);
    }
    averageRepeatTime_ = yShape_.GetDim(axis_) / inputShape_.GetDim(axis_);
    int64_t ubFactor = 0;
    ge::DataType computeDtype;
    if (isUseInt64_ == 0) {
        computeDtype = ge::DataType::DT_INT32;
    } else if (isUseInt64_ == 1) {
        computeDtype = ge::DataType::DT_INT64;
    }
    if (isSplitShape_) {
        ubSize_ -= (totalCoreNum_ * ALIGN_COUNT * DOUBLE);
        ubFactor = (ubSize_ - REDUCE_SUM_BUFFER * ge::GetSizeByDataType(computeDtype)) /
                   (ge::GetSizeByDataType(inputDtype_) * DOUBLE + ge::GetSizeByDataType(repeatDtype_));
    } else {
        ubFactor = ubSize_ / DOUBLE /
                   (ge::GetSizeByDataType(inputDtype_) * DOUBLE + ge::GetSizeByDataType(repeatDtype_));
    }
    ubFactor_ = ubFactor / ALIGN_COUNT * ALIGN_COUNT;
    return;
}

bool RepeatInterleaveTilingKernelNorm::isSplitBatchSimt()
{
    // repeats为scalar返回false
    if (repeatShape_.GetDimNum() == 0 || repeatShape_.GetDim(0) == 1) {
        return false;
    }

    MergDim();

    int64_t batchNum = mergedDim_[0];
    eachCoreBatchCount_ = CeilDiv(batchNum, totalCoreNum_);
    usedCoreNum_ = CeilDiv(batchNum, eachCoreBatchCount_);
    usedCoreNumBefore_ = usedCoreNum_; // batchCoreNum
    tailCoreBatchCount_ = batchNum - eachCoreBatchCount_ * (usedCoreNum_ - 1);
    int64_t cpAxis = 2;

    // batch 轴够切且尾轴小于128B走simtCumSum模板
    if ((usedCoreNum_ >= totalCoreNum_ / DOUBLE) &&
        (mergedDim_[cpAxis] * ge::GetSizeByDataType(inputDtype_) <= SIMT_CP_THRESHOLD)) {
        isSimtSplitBatch_ = true;
        return true;
    }
    return false;
}

bool RepeatInterleaveTilingKernelNorm::isSplitRepeatSumSimt()
{
    // repeats为scalar返回false
    if (repeatShape_.GetDimNum() == 0 || repeatShape_.GetDim(0) == 1) {
        return false;
    }

    MergDim();

    int64_t batchNum = mergedDim_[0];
    eachCoreBatchCount_ = CeilDiv(batchNum, totalCoreNum_);
    usedCoreNum_ = CeilDiv(batchNum, eachCoreBatchCount_);
    usedCoreNumBefore_ = usedCoreNum_; // batchCoreNum
    // tailCoreBatchCount_ = batchNum - eachCoreBatchCount_ * (usedCoreNum_ - 1);
    totalRepeatSum_ = yShape_.GetDim(axis_);
    int64_t cpAxis = 2;
    // batch轴切分后，核数不满，则需要切分repeats sum
    if (usedCoreNum_ < totalCoreNum_ / DOUBLE) {
        int64_t cpByte = mergedDim_[cpAxis] * ge::GetSizeByDataType(inputDtype_);
        int64_t ratio = totalCoreNum_ / usedCoreNum_;
        int64_t normalRepeatCoreRepeatsCount = CeilDiv(mergedDim_[1], ratio);
        if ((totalRepeatSum_ > MIN_SUM_REPEATS) && (cpByte <= SIMT_SPLIT_REPEAT_CP_THRESHOLD) &&
            (normalRepeatCoreRepeatsCount > SIMT_SPLIT_REPEAT_REPEATCORE_THRESHOLD ||
             cpByte <= SIMT_SPLIT_REPEAT_CP_THRESHOLD_SMALL)) {
            return true;
        }
    }
    return false;
}

void RepeatInterleaveTilingKernelNorm::CalcSimtAddr()
{
    uint64_t yShapeNum = 1;
    for (size_t i = 0; i < yShape_.GetDimNum(); i++) {
        yShapeNum *= yShape_.GetDim(i);
    }
    uint64_t xShapeNum = 1;
    for (size_t i = 0; i < inputShape_.GetDimNum(); i++) {
        xShapeNum *= inputShape_.GetDim(i);
    }

    // simt地址偏移用int64
    if (xShapeNum > INT32_MAX_LIM || yShapeNum > INT32_MAX_LIM) {
        isUseInt64_ = 1;
    }
}

ge::graphStatus RepeatInterleaveTilingKernelNorm::splitBatchSimtWithCumSumTiling()
{
    ubSize_ -= DCACHE_SIZE;
    CumSumTiling();
    CalcSimtAddr();

    int64_t repeatsShape = repeatShape_.GetDim(0);
    /* repeats小于8k时，simt核内自己算前缀和，放ub*/
    if (repeatsShape * ge::GetSizeByDataType(repeatDtype_) < CUMSUMUB_REPEATS_THRESHOLD) {
        isCumSumInUb_ = true;
    }

    averageRepeatTime_ = yShape_.GetDim(axis_) / inputShape_.GetDim(axis_);
    int64_t cpAxis = 2;
    threadNumX_ = mergedDim_[cpAxis];
    threadNumY_ = std::min(MAX_THREAD_NUM / threadNumX_, mergedDim_[1]);
    threadNumZ_ = std::min(MAX_THREAD_NUM / threadNumX_ / threadNumY_, eachCoreBatchCount_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RepeatInterleaveTilingKernelNorm::SplitRepeatSimtWithCumSumTiling()
{
    isSimtSplitRepeatSum_ = true;
    ubSize_ -= DCACHE_SIZE;

    int64_t cpAxis = 2;
    threadNumX_ = mergedDim_[cpAxis];
    threadNumY_ = std::min(MAX_THREAD_NUM / threadNumX_, mergedDim_[1]);
    CalcSimtAddr();

    CumSumTiling();

    // 剩余分不满的核按 totalRepeatSum_ 分核
    int64_t ratio = totalCoreNum_ / usedCoreNum_;
    normalRepeatsCount_ = CeilDiv(totalRepeatSum_, ratio);
    repeatsSlice_ = CeilDiv(totalRepeatSum_, normalRepeatsCount_); // repeatsCoreNum
    usedCoreNum_ = usedCoreNum_ * repeatsSlice_;
    tailRepeatsCount_ = totalRepeatSum_ - normalRepeatsCount_ * (repeatsSlice_ - 1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RepeatInterleaveTilingKernelNorm::DoOpTiling()
{
    if (isSplitBatchSimt()) {
        return splitBatchSimtWithCumSumTiling();
    }

    if (isSplitRepeatSumSimt()) {
        return SplitRepeatSimtWithCumSumTiling();
    }
    MergDim();
    UseInt64();
    GetUbFactor();
    return ge::GRAPH_SUCCESS;
}

uint64_t RepeatInterleaveTilingKernelNorm::GetTilingKey() const
{
    if (isSimtSplitBatch_) {
        uint64_t tilingKey = 5000UL;
        if (isCumSumInUb_) {
            tilingKey += SIMT_CUMSUMINUB_KEY;
        }
        if (isUseInt64_ == 1) {
            tilingKey += SIMT_ADDR_KEY;
        }
        tilingKey += isCumSumCast_ ? 1 : 0;
        return tilingKey;
    }
    if (isSimtSplitRepeatSum_) {
        uint64_t tilingKey = 4000UL;
        if (isUseInt64_ == 1) {
            tilingKey += SIMT_ADDR_KEY;
        }
        tilingKey += isCumSumCast_ ? 1 : 0;
        return tilingKey;
    }
    if (isSplitShape_ && isUseInt64_ == 0) {
        return BATCH_TILING_SPLIT_REPEATS_SHAPE_INT32;
    } else if (isSplitShape_) {
        return BATCH_TILING_SPLIT_REPEATS_SHAPE_INT64;
    }
    if (repeatsCount_ > -1) {
        return BATCH_TILING_REPEAT_SCALAR;
    }
    return BATCH_TILING_REPEAT_TENSOR;
}

void RepeatInterleaveTilingKernelNorm::setCumSumTilingData()
{
    cumSumTilingData_.set_cumSumCoreNum(cumSumCoreNum_);
    cumSumTilingData_.set_cumSumNormalCoreRepeatsCount(cumSumNormalCoreRepeatsCount_);
    cumSumTilingData_.set_cumSumTailCoreRepeatsCount(cumSumTailCoreRepeatsCount_);
    cumSumTilingData_.set_cumSumNormalCoreLoops(cumSumNormalCoreLoops_);
    cumSumTilingData_.set_cumSumTailCoreLoops(cumSumTailCoreLoops_);
    cumSumTilingData_.set_cumSumNormalUbFactors(cumSumNormalUbFactors_);
    cumSumTilingData_.set_cumSumNormalCoreTailUbFactors(cumSumNormalCoreTailUbFactors_);
    cumSumTilingData_.set_cumSumTailCoreTailUbFactors(cumSumTailCoreTailUbFactors_);

    cumSumTilingData_.set_eachCoreBatchCount(eachCoreBatchCount_);
    cumSumTilingData_.set_tailCoreBatchCount(tailCoreBatchCount_);
    cumSumTilingData_.set_threadNumX(threadNumX_);
    cumSumTilingData_.set_threadNumY(threadNumY_);
    cumSumTilingData_.set_threadNumZ(threadNumZ_);
    cumSumTilingData_.set_totalRepeatSum(totalRepeatSum_);

    cumSumTilingData_.set_usedCoreNum(usedCoreNum_);
    cumSumTilingData_.set_batchCoreNum(usedCoreNumBefore_);
    // cumSumTilingData_.set_cpCoreNum(cpSlice_);
    cumSumTilingData_.set_repeatsCoreNum(repeatsSlice_);
    cumSumTilingData_.set_normalCoreOutputRepeats(normalRepeatsCount_);
    cumSumTilingData_.set_tailCoreOutputRepeats(tailRepeatsCount_);
    cumSumTilingData_.set_mergedDims(mergedDim_);

    cumSumTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                   context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(cumSumTilingData_.GetDataSize());
}

ge::graphStatus RepeatInterleaveTilingKernelNorm::PostTiling()
{
    if (isSimtSplitRepeatSum_ || isSimtSplitBatch_) {
        int64_t coreNum = std::max(usedCoreNum_, cumSumCoreNum_);
        context_->SetBlockDim(coreNum);
        context_->SetScheduleMode(1);
        context_->SetLocalMemorySize(ubSize_);
        setCumSumTilingData();
        return ge::GRAPH_SUCCESS;
    }
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_);
    if (isSplitShape_) {
        kernelTilingDataSmall_.set_totalCoreNum(totalCoreNum_);
        kernelTilingDataSmall_.set_usedCoreNum(usedCoreNum_);
        kernelTilingDataSmall_.set_eachCoreBatchCount(eachCoreBatchCount_);
        kernelTilingDataSmall_.set_tailCoreBatchCount(tailCoreBatchCount_);
        kernelTilingDataSmall_.set_normalRepeatsCount(normalRepeatsCount_);
        kernelTilingDataSmall_.set_tailRepeatsCount(tailRepeatsCount_);
        kernelTilingDataSmall_.set_repeatsSlice(repeatsSlice_);
        kernelTilingDataSmall_.set_ubFactor(ubFactor_);
        kernelTilingDataSmall_.set_totalRepeatSum(totalRepeatSum_);
        kernelTilingDataSmall_.set_averageRepeatTime(averageRepeatTime_);
        kernelTilingDataSmall_.set_mergedDims(mergedDim_);
        kernelTilingDataSmall_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                            context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(kernelTilingDataSmall_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
    kernelTilingData_.set_totalCoreNum(totalCoreNum_);
    kernelTilingData_.set_usedCoreNum(usedCoreNum_);
    kernelTilingData_.set_eachCoreBatchCount(eachCoreBatchCount_);
    kernelTilingData_.set_tailCoreBatchCount(tailCoreBatchCount_);
    kernelTilingData_.set_isSplitCP(isSplitCP_);
    kernelTilingData_.set_normalCP(normalCP_);
    kernelTilingData_.set_tailCP(tailCP_);
    kernelTilingData_.set_cpSlice(cpSlice_);
    kernelTilingData_.set_ubFactor(ubFactor_);
    kernelTilingData_.set_repeatsCount(repeatsCount_);
    kernelTilingData_.set_totalRepeatSum(totalRepeatSum_);
    kernelTilingData_.set_mergedDims(mergedDim_);

    kernelTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                   context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(kernelTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void RepeatInterleaveTilingKernelNorm::DumpCumSumTilingInfo()
{
    std::ostringstream info;
    info << "cumSumCoreNum: " << cumSumCoreNum_;
    info << ", cumSumNormalCoreRepeatsCount: " << cumSumNormalCoreRepeatsCount_;
    info << ", cumSumTailCoreRepeatsCount: " << cumSumTailCoreRepeatsCount_;
    info << ", cumSumNormalCoreLoops: " << cumSumNormalCoreLoops_;
    info << ", cumSumTailCoreLoops: " << cumSumTailCoreLoops_;
    info << ", cumSumNormalUbFactors: " << cumSumNormalUbFactors_;
    info << ", cumSumNormalCoreTailUbFactors: " << cumSumNormalCoreTailUbFactors_;
    info << ", cumSumTailCoreTailUbFactors: " << cumSumTailCoreTailUbFactors_;
    info << ", eachCoreBatchCount: " << eachCoreBatchCount_;
    info << ", tailCoreBatchCount: " << tailCoreBatchCount_;
    info << ", threadNumX: " << threadNumX_;
    info << ", threadNumY: " << threadNumY_;
    info << ", threadNumZ: " << threadNumX_;
    info << ", totalRepeatSum: " << totalRepeatSum_;
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", batchCoreNum: " << usedCoreNumBefore_;
    info << ", repeatsCoreNum: " << repeatsSlice_;
    info << ", normalCoreOutputRepeats: " << normalRepeatsCount_;
    info << ", tailCoreOutputRepeats: " << tailRepeatsCount_;
    info << ", tilingKey: " << GetTilingKey();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

void RepeatInterleaveTilingKernelNorm::DumpTilingInfo()
{
    if (isSimtSplitRepeatSum_ || isSimtSplitBatch_) {
        DumpCumSumTilingInfo();
    }
    std::ostringstream info;
    info << "totalCoreNum: " << totalCoreNum_;
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", eachCoreBatchCount: " << eachCoreBatchCount_;
    info << ", tailCoreBatchCount: " << tailCoreBatchCount_;
    info << ", isSplitCP: " << isSplitCP_;
    info << ", normalCP: " << normalCP_;
    info << ", tailCP: " << tailCP_;
    info << ", cpSlice: " << cpSlice_;
    info << ", ubFactor: " << ubFactor_;
    info << ", repeatsCount: " << repeatsCount_;
    info << ", totalRepeatSum: " << totalRepeatSum_;
    info << ", isSplitShape: " << isSplitShape_;
    info << ", normalRepeatsCount: " << normalRepeatsCount_;
    info << ", tailRepeatsCount: " << tailRepeatsCount_;
    info << ", repeatsSlice: " << repeatsSlice_;
    info << ", tilingKey: " << GetTilingKey();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
    return;
}

ge::graphStatus RepeatInterleaveTilingKernelNorm::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (isSimtSplitRepeatSum_ || isSimtSplitBatch_) {
        int64_t repeatsShape = repeatShape_.GetDim(0);
        int64_t cumSumDataTypeSize = isCumSumCast_ ? static_cast<int64_t>(sizeof(int64_t)) :
                                                     static_cast<int64_t>(ge::GetSizeByDataType(repeatDtype_));
        sysWorkspace += repeatsShape * cumSumDataTypeSize + cumSumCoreNum_ * cumSumDataTypeSize;
    } else {
        sysWorkspace += totalCoreNum_ * CACHELINE_SIZE;
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("RepeatInterleave", RepeatInterleaveTilingKernelNorm, 2);
} // namespace optiling
