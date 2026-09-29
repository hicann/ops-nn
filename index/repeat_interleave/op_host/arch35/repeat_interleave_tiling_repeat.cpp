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
 * \file repeat_interleave_tiling_repeat.cpp
 * \brief
 */

#include "repeat_interleave_tiling_repeat.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
static constexpr int64_t DOUBLE = 2;
static constexpr int64_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
static constexpr int64_t SPLIT_REPEAT_REPEAT_INT32_CAST_TRUE_SHAPE_INT64 = 3011;
static constexpr int64_t SPLIT_REPEAT_REPEAT_INT32_CAST_FALSE_SHAPE_INT64 = 3001;
static constexpr int64_t SPLIT_REPEAT_REPEAT_INT32_CAST_FALSE_SHAPE_INT32 = 3000;
static constexpr int64_t SPLIT_REPEAT_REPEAT_INT64_CAST_FALSE_SHAPE_INT64 = 3101;
static constexpr int64_t MIN_TOTAL_REPEATS_SUM = 1024;
static constexpr int64_t MIN_CP_THRESHOLD = 2048;
static constexpr int64_t MIN_CP_SPLIT_FACTOR = 2048;
static constexpr int64_t REMAIN_TO_REPEAT_SIZE = 512;     // 2 * 256，预留空间在ub内使用非对齐复制
static constexpr int64_t REMAIN_TO_INPUT_REPEAT_IDX = 32; // 预留空间在ub内存储计算的输入repeat
static constexpr int64_t CP_AXIS = 2;
static constexpr int64_t REPEAT_UB_BUFFER = 16384;  // 16 * 1024
static constexpr int64_t UB_FOR_SIMT_CACHE = 32768; // 32 * 1024
static constexpr int64_t MIN_CP_AXIS_LIMIT = 512;

bool RepeatInterleaveTilingKernelRepeatTiling::IsCapable()
{
    // 101场景：repeats为scalar/广播（shape为空或为1），由Norm的scalar路径（101）处理
    if (repeatShape_.GetDimNum() == 0 || repeatShape_.GetDim(0) == 1) {
        return false;
    }
    MergDim();
    // 102场景：模拟Norm的batch/CP切核，若切分后核数能满足要求（过半），由Norm的tensor路径（102）处理
    int64_t batchNum = mergedDim_[0];
    int64_t eachCoreBatchCount = Ops::Base::CeilDiv(batchNum, totalCoreNum_);
    int64_t usedCore = Ops::Base::CeilDiv(batchNum, eachCoreBatchCount);
    if ((usedCore < totalCoreNum_ / DOUBLE) &&
        (mergedDim_[CP_AXIS] * ge::GetSizeByDataType(inputDtype_) > MIN_CP_THRESHOLD)) {
        int64_t ratio = totalCoreNum_ / usedCore;
        int64_t cpDim = mergedDim_[CP_AXIS];
        int64_t cpMin = MIN_CP_THRESHOLD / ge::GetSizeByDataType(inputDtype_);
        int64_t normalCP = std::max(Ops::Base::CeilDiv(cpDim, ratio), cpMin);
        int64_t cpSlice = Ops::Base::CeilDiv(cpDim, normalCP);
        usedCore = usedCore * cpSlice;
    }
    if (usedCore >= totalCoreNum_ / DOUBLE) {
        return false;
    }
    if (mergedDim_[CP_AXIS] < MIN_CP_AXIS_LIMIT / ge::GetSizeByDataType(inputDtype_)) {
        return false;
    }
    // 其余场景（batch/CP切分无法满足核数要求）优先在本模板（301/311/302）处理
    return true;
}

void RepeatInterleaveTilingKernelRepeatTiling::SplitCPForKernelUbFactor()
{
    cpSplitBlocks_ = totalCoreNum_ / usedCoreNum_;
    for (; cpSplitBlocks_ > 1; cpSplitBlocks_--) {
        if (mergedDim_[CP_AXIS] / cpSplitBlocks_ < MIN_CP_SPLIT_FACTOR) {
            continue;
        } else {
            eachCoreCpCount_ = mergedDim_[CP_AXIS] / cpSplitBlocks_;
            tailCoreCpCount_ = mergedDim_[CP_AXIS] - eachCoreCpCount_ * cpSplitBlocks_;
            isSplitCPForKernel_ = 1;
            usedCoreNum_ = usedCoreNum_ * cpSplitBlocks_;
            break;
        }
    }
}

void RepeatInterleaveTilingKernelRepeatTiling::SplitCPUbFactor()
{
    // cp 轴很大，单个 buffer 放不下一行，需要将 cp 轴按片切分
    uint32_t ubBlock = Ops::Base::GetUbBlockSize(context_);
    uint32_t ubBlockFactor = ubBlock / ge::GetSizeByDataType(inputDtype_);
    isSplitCP_ = 1;
    singleBufSize_ = (ubSize_ - REPEAT_UB_BUFFER - REMAIN_TO_INPUT_REPEAT_IDX) / DOUBLE;
    cpSliceFactorAlign_ = Ops::Base::FloorAlign(singleBufSize_ / ge::GetSizeByDataType(inputDtype_),
                                                static_cast<int64_t>(ubBlockFactor));
    cpSliceNum_ = Ops::Base::CeilDiv(mergedDim_[CP_AXIS], cpSliceFactorAlign_);
    cpSliceFactorTail_ = mergedDim_[CP_AXIS] - (cpSliceNum_ - 1) * cpSliceFactorAlign_;
    cpCountInUb_ = 1;
}

void RepeatInterleaveTilingKernelRepeatTiling::GetUbFactor()
{
    int64_t batchNum = mergedDim_[0];
    // 使用 outputshape 进行分核
    totalRepeatSum_ = yShape_.GetDim(axis_);
    int64_t totalCps = batchNum * totalRepeatSum_;
    eachCoreCpCount_ = Ops::Base::CeilDiv(totalCps, totalCoreNum_);
    usedCoreNum_ = totalCoreNum_;
    if (eachCoreCpCount_ == 1) {
        usedCoreNum_ = Ops::Base::CeilDiv(totalCps, eachCoreCpCount_);
    }
    eachCoreCpCount_ = totalCps / usedCoreNum_;
    tailCoreCpCount_ = totalCps - eachCoreCpCount_ * usedCoreNum_;
    singleBufSize_ = (ubSize_ - REPEAT_UB_BUFFER - REMAIN_TO_INPUT_REPEAT_IDX - UB_FOR_SIMT_CACHE) / DOUBLE / DOUBLE -
                     REMAIN_TO_REPEAT_SIZE;
    cpCountInUb_ = singleBufSize_ / (mergedDim_[CP_AXIS] * ge::GetSizeByDataType(inputDtype_));
    if (usedCoreNum_ < totalCoreNum_ / DOUBLE) {
        SplitCPForKernelUbFactor();
    }
    if (cpCountInUb_ < 1) {
        SplitCPUbFactor();
    }
}

ge::graphStatus RepeatInterleaveTilingKernelRepeatTiling::DoOpTiling()
{
    UseInt64();
    CumSumTiling();
    GetUbFactor();
    return ge::GRAPH_SUCCESS;
}

uint64_t RepeatInterleaveTilingKernelRepeatTiling::GetTilingKey() const
{
    if (repeatDtype_ == ge::DT_INT32 && isCumSumCast_ && isUseInt64_) {
        return SPLIT_REPEAT_REPEAT_INT32_CAST_TRUE_SHAPE_INT64;
    } else if (repeatDtype_ == ge::DT_INT32 && !isCumSumCast_ && isUseInt64_) {
        return SPLIT_REPEAT_REPEAT_INT32_CAST_FALSE_SHAPE_INT64;
    } else if (repeatDtype_ == ge::DT_INT32 && !isCumSumCast_ && !isUseInt64_) {
        return SPLIT_REPEAT_REPEAT_INT32_CAST_FALSE_SHAPE_INT32;
    }
    return SPLIT_REPEAT_REPEAT_INT64_CAST_FALSE_SHAPE_INT64;
}

void RepeatInterleaveTilingKernelRepeatTiling::SetRepeatKernelTilingData()
{
    kernelTilingRepeatData_.set_cumSumCoreNum(cumSumCoreNum_);
    kernelTilingRepeatData_.set_cumSumNormalCoreRepeatsCount(cumSumNormalCoreRepeatsCount_);
    kernelTilingRepeatData_.set_cumSumTailCoreRepeatsCount(cumSumTailCoreRepeatsCount_);
    kernelTilingRepeatData_.set_cumSumNormalCoreLoops(cumSumNormalCoreLoops_);
    kernelTilingRepeatData_.set_cumSumTailCoreLoops(cumSumTailCoreLoops_);
    kernelTilingRepeatData_.set_cumSumNormalUbFactors(cumSumNormalUbFactors_);
    kernelTilingRepeatData_.set_cumSumNormalCoreTailUbFactors(cumSumNormalCoreTailUbFactors_);
    kernelTilingRepeatData_.set_cumSumTailCoreTailUbFactors(cumSumTailCoreTailUbFactors_);

    kernelTilingRepeatData_.set_usedCoreNum(usedCoreNum_);
    kernelTilingRepeatData_.set_eachCoreCpCount(eachCoreCpCount_);
    kernelTilingRepeatData_.set_tailCoreCpCount(tailCoreCpCount_);
    kernelTilingRepeatData_.set_cpCountInUb(cpCountInUb_);
    kernelTilingRepeatData_.set_isSplitCP(isSplitCP_);
    kernelTilingRepeatData_.set_cpSliceNum(cpSliceNum_);
    kernelTilingRepeatData_.set_cpSliceFactorAlign(cpSliceFactorAlign_);
    kernelTilingRepeatData_.set_cpSliceFactorTail(cpSliceFactorTail_);
    kernelTilingRepeatData_.set_isSplitCPForKernel(isSplitCPForKernel_);
    kernelTilingRepeatData_.set_cpSplitBlocks(cpSplitBlocks_);
    kernelTilingRepeatData_.set_totalRepeatSum(totalRepeatSum_);
    kernelTilingRepeatData_.set_mergedDims(mergedDim_);
    kernelTilingRepeatData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                         context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(kernelTilingRepeatData_.GetDataSize());
}

ge::graphStatus RepeatInterleaveTilingKernelRepeatTiling::PostTiling()
{
    context_->SetBlockDim(std::max(cumSumCoreNum_, usedCoreNum_));
    context_->SetScheduleMode(1);
    context_->SetLocalMemorySize(ubSize_ - UB_FOR_SIMT_CACHE);
    SetRepeatKernelTilingData();
    return ge::GRAPH_SUCCESS;
}

void RepeatInterleaveTilingKernelRepeatTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "usedCoreNum: " << usedCoreNum_;
    info << ", eachCoreCpCount: " << eachCoreCpCount_;
    info << ", tailCoreCpCount: " << tailCoreCpCount_;
    info << ", cpCountInUb: " << cpCountInUb_;
    info << ", isSplitCP: " << isSplitCP_;
    info << ", cpSliceNum: " << cpSliceNum_;
    info << ", cpSliceFactorAlign: " << cpSliceFactorAlign_;
    info << ", cpSliceFactorTail: " << cpSliceFactorTail_;
    info << ", isSplitCPForKernel: " << isSplitCPForKernel_;
    info << ", cpSplitBlocks: " << cpSplitBlocks_;
    info << ", totalRepeatSum: " << totalRepeatSum_;
    info << ", tilingKey: " << GetTilingKey();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
    return;
}

ge::graphStatus RepeatInterleaveTilingKernelRepeatTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();
    int64_t repeatsShape = repeatShape_.GetDim(0);
    int64_t cumSumDataTypeSize = isCumSumCast_ ? static_cast<int64_t>(sizeof(int64_t)) :
                                                 static_cast<int64_t>(ge::GetSizeByDataType(repeatDtype_));
    sysWorkspace += repeatsShape * cumSumDataTypeSize + cumSumCoreNum_ * cumSumDataTypeSize;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("RepeatInterleave", RepeatInterleaveTilingKernelRepeatTiling, 1);
} // namespace optiling
