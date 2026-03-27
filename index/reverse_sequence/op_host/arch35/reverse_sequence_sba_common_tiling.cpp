/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file reverse_sequence_a1sba_tiling.cpp
 * \brief
 */

#include "reverse_sequence_a1sba_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "error_util.h"
#include "index/reverse_sequence/op_kernel/arch35/reverse_sequence_struct.h"
#include "index/reverse_sequence/op_kernel/arch35/reverse_sequence_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
using namespace ReverseSequence;

static constexpr int64_t SIMT_THRESHOLD = 128;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t SPLIT_DIM_A = 1;
static constexpr int64_t SPLIT_DIM_S = 2;
static constexpr int64_t SPLIT_DIM_B = 3;
static constexpr int64_t SPLIT_DIM_A1 = 4;
static constexpr int64_t BLOCK_SPLIT_THREHOLD = 4096;
static constexpr int64_t DIM_A1 = 0;
static constexpr int64_t DIM_S = 1;
static constexpr int64_t DIM_B = 2;
static constexpr int64_t DIM_A = 3;
static constexpr int64_t TEMPLATE_MODE = 4;
static constexpr int64_t SBA_RESERVED_SIZE = 131072; // 128k
static constexpr int64_t TYPE_A1SBA = 9;
static constexpr int64_t TYPE_SBA = 4;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

bool ReverseSequenceSBACommonTiling::IsCapable()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::IsCapable begin");
    if (inputData_.comBineType != TYPE_A1SBA || inputData_.comBineType != TYPE_SBA) {
        return false;
    }
    InitializationVars();
    DoUBTiling();
    if (inputData_.inputDim[DIM_A] * inputData_.xDtypeSize < SIMT_THRESHOLD) {
        // A轴太小的离散场景不处理；SBA能全载的除外
        if (splitMode_ == SPLIT_DIM_A1) {
            return true;
        }
        return false;
    }
    return true;
}

void ReverseSequenceSBACommonTiling::InitializationVars()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::InitializationVars begin");
}

void ReverseSequenceSBACommonTiling::DoBlockTiling()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::DoBlockTiling begin");
    int64_t totalLoop = a1Loop_ * sLoop_ * bLoop_ * aLoop_;
    blockFactor_ = totalLoop / static_cast<int64_t>(coreNum_);
    blockTail_ = totalLoop - blockFactor_ * static_cast<int64_t>(coreNum_);
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : static_cast<int64_t>(coreNum_);
    int64_t ubFactorA = splitMode_ == SPLIT_DIM_A || splitMode_ == SPLIT_DIM_A1 ? ubFactorA_ : Ops::Base::CeilAlign(ubFactorA_, oneBlockNum_);
    int64_t ubFactorSBA = ubFactorS_ * ubFactorB_ * ubFactorA;
    if (splitMode_ == SPLIT_DIM_A1) {
        ubFactorSBA = Ops::Base::CeilAlign(ubFactorSBA, oneBlockNum_);
    }
    inUbSize_ = ubFactorA1_ * ubFactorSBA;
}

int64_t ReverseSequenceSBACommonTiling::CalcBufferSize(int64_t inA1, int64_t inS, int64_t inB, int64_t inA, int64_t splitMode)
{
    OP_LOGD("ReverseSequenceSBACommonTiling::CalcBufferSize begin");
    int64_t tmpInDataBufferSize = inA1 * inS * inB * Ops::Base::CeilAlign(inA, oneBlockNum_);
    if (splitMode == SPLIT_DIM_A1) {
        tmpInDataBufferSize = Ops::Base::CeilAlign(inA1 * inS * inB * inA, oneBlockNum_) * 2;
    }
    
    tmpInDataBufferSize *= DOUBLE_BUFFER;
    if (splitMode == SPLIT_DIM_A1) {
        tmpInDataBufferSize += sbaResvervedNum_;
    }
    return tmpInDataBufferSize;
}

void ReverseSequenceSBACommonTiling::CalcSplitDimB()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::CalcSplitDimB begin");
    int64_t inDimBLower = 1;
    int64_t inDimBUpper = inputData_.inputDim[DIM_B];
    while (inDimBLower < inDimBUpper) {
        int64_t inDimBMid = (inDimBLower + inDimBUpper + 1) / DIGIT_TWO;
        int64_t midBuffer = CalcBufferSize(1, 1, inDimBMid, inputData_.inputDim[DIM_A], SPLIT_DIM_B);
        if (midBuffer <= availableUb_) {
            inDimBLower = inDimBMid;
        } else {
            inDimBUpper = inDimBMid - 1;
        }
    }
    ubFactorA_ = inputData_.inputDim[DIM_A];
    ubFactorB_ = inDimBLower;
    ubFactorS_ = 1;
    ubFactorA1_ = 1;
    a1Loop_ = inputData_.inputDim[DIM_A1];
    sLoop_ = inputData_.inputDim[DIM_S];
    bLoop_ = (inputData_.inputDim[DIM_B] + ubFactorB_ - 1) / ubFactorB_;
    aLoop_ = 1;
    splitMode_ = SPLIT_DIM_B;
}

void ReverseSequenceSBACommonTiling::CalcSplitDimA()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::CalcSplitDimA begin");
    int64_t inDimA = std::min(inputData_.inputDim[DIM_A], (availableUb_ / DIGIT_TWO));
    ubFactorA_ = inDimA;

    if (ubFactorA_ <= 0) {
        OP_LOGE(context_, "ReverseSequence ubFactorA_ is %ld.", ubFactorA_);
        return;
    }
    ubFactorB_ = 1;
    ubFactorS_ = 1;
    ubFactorA1_ = 1;
    a1Loop_ = inputData_.inputDim[DIM_A1];
    sLoop_ = inputData_.inputDim[DIM_S];
    bLoop_ = inputData_.inputDim[DIM_B];
    aLoop_ = (inputData_.inputDim[DIM_A] + ubFactorA_ - 1) / ubFactorA_;
    splitMode_ = SPLIT_DIM_A;
}

void ReverseSequenceSBACommonTiling::CalcSplitDimS()
{    
    OP_LOGD("ReverseSequenceSBACommonTiling::CalcSplitDimS begin");
    int64_t inDimSLower = 1;
    int64_t inDimSUpper = inputData_.inputDim[DIM_S];
    while (inDimSLower < inDimSUpper) {
        int64_t inDimSMid = (inDimSLower + inDimSUpper + 1) / DIGIT_TWO;
        int64_t midBuffer = CalcBufferSize(1, inDimSMid, inputData_.inputDim[DIM_B], inputData_.inputDim[DIM_A], SPLIT_DIM_S);
        if (midBuffer <= availableUb_) {
            inDimSLower = inDimSMid;
        } else {
            inDimSUpper = inDimSMid - 1;
        }
    }
    ubFactorB_ = inputData_.inputDim[DIM_B];
    ubFactorA_ = inputData_.inputDim[DIM_A];
    ubFactorS_ = inDimSLower;
    ubFactorA1_ = 1;
    a1Loop_ = inputData_.inputDim[DIM_A1];
    sLoop_ = (inputData_.inputDim[DIM_S] + ubFactorS_ - 1) / ubFactorS_;
    bLoop_ = 1;
    aLoop_ = 1;
    splitMode_ = SPLIT_DIM_S;
}

void ReverseSequenceSBACommonTiling::CalcSplitDimA1() 
{
    OP_LOGD("ReverseSequenceSBACommonTiling::CalcSplitDimA1 begin");
    int64_t inDimA1Lower = 1;
    int64_t inDimA1Upper = inputData_.inputDim[DIM_A1];
    while (inDimA1Lower < inDimA1Upper) {
        int64_t inDimA1Mid = (inDimA1Lower + inDimA1Upper + 1) / DIGIT_TWO;
        int64_t midBuffer = CalcBufferSize(inDimA1Mid, inputData_.inputDim[DIM_S], inputData_.inputDim[DIM_B], inputData_.inputDim[DIM_A], SPLIT_DIM_A1);
        if (midBuffer <= availableUb_) {
            inDimA1Lower = inDimA1Mid;
        } else {
            inDimA1Upper = inDimA1Mid - 1;
        }
    }
    ubFactorB_ = inputData_.inputDim[DIM_B];
    ubFactorA_ = inputData_.inputDim[DIM_A];
    ubFactorS_ = inputData_.inputDim[DIM_S];
    ubFactorA1_ = inDimA1Lower;
    a1Loop_ = (inputData_.inputDim[DIM_A1] + ubFactorA1_ - 1) / ubFactorA1_;
    sLoop_ = 1;
    bLoop_ = 1;
    aLoop_ = 1;
    splitMode_ = SPLIT_DIM_A1;
}

void ReverseSequenceSBACommonTiling::DoUBTilingSingle()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::DoUBTilingSingle begin");
    int64_t oneSBABuffer = CalcBufferSize(1, inputData_.inputDim[DIM_S], inputData_.inputDim[DIM_B], inputData_.inputDim[DIM_A], SPLIT_DIM_A1);
    int64_t oneSeqBuffer = CalcBufferSize(1, 1,  inputData_.inputDim[DIM_B], inputData_.inputDim[DIM_A], SPLIT_DIM_S);
    int64_t oneBatchBuffer = CalcBufferSize(1, 1, 1, inputData_.inputDim[DIM_A], SPLIT_DIM_B);
    // SBA 全载
    if (oneSBABuffer <= availableUb_) {
        CalcSplitDimA1();
        return;
    }
    // BA 全载
    if (oneSeqBuffer <= availableUb_) {
        CalcSplitDimS();
        return;
    }
    // A 全载
    if (oneBatchBuffer <= availableUb_) {
        CalcSplitDimB();
        return;
    }
    // A 不全载
    CalcSplitDimA();
}

void ReverseSequenceSBACommonTiling::DoUBTiling()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::DoUBTiling begin");
    int64_t ubStep = BLOCK_SPLIT_THREHOLD / inputData_.xDtypeSize;
    do {
        DoUBTilingSingle();
        if (a1Loop_ * sLoop_ * bLoop_ * aLoop_ >= static_cast<int64_t>(coreNum_)) {
            break;
        }
        availableUb_ -= ubStep;
    } while (availableUb_ > ubStep);
}

ge::graphStatus ReverseSequenceSBACommonTiling::DoOpTiling()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::DoOpTiling begin");
    DoBlockTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceSBACommonTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}


uint64_t ReverseSequenceSBACommonTiling::GetTilingKey() const
{
    OP_LOGD("ReverseSequenceSBACommonTiling::GetTilingKey begin");
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, inputData_.xDtypeSize, addrRange_);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

ge::graphStatus ReverseSequenceSBACommonTiling::GetWorkspaceSize()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::GetWorkspaceSize begin");
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceSBACommonTiling::PostTiling()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::PostTiling begin");
    context_->SetBlockDim(usedCoreNum_);
    if (inputData_.xShapeSize >= INT32_MAX) {
        addrRange_ = 1;
    }
    if (splitMode_ == SPLIT_DIM_A1) {
        auto res = context_->SetLocalMemorySize(ubSize_ - SBA_RESERVED_SIZE);
        OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
            OP_LOGE(context_->GetNodeName(), "SetLocalMemorySize ubSize = %lu failed.", (ubSize_ - SBA_RESERVED_SIZE)), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void ReverseSequenceSBACommonTiling::SetTilingData()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::SetTilingData begin");
    ReverseSequence::ReverseSequenceA1SBATilingData* tilingData =
        context_->GetTilingData<ReverseSequence::ReverseSequenceA1SBATilingData>();

    tilingData->a1Dim = inputData_.inputDim[DIM_A1];
    tilingData->sDim = inputData_.inputDim[DIM_S];
    tilingData->bDim = inputData_.inputDim[DIM_B];
    tilingData->aDim = inputData_.inputDim[DIM_A];
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->ubFactorA1 = ubFactorA1_;
    tilingData->ubFactorS = ubFactorS_;
    tilingData->ubFactorB = ubFactorB_;
    tilingData->ubFactorA = ubFactorA_;
    tilingData->a1Loop = a1Loop_;
    tilingData->sLoop = sLoop_;
    tilingData->bLoop = bLoop_;
    tilingData->aLoop = aLoop_;
    tilingData->inUbSize = inUbSize_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->splitMode = splitMode_;
    tilingData->dtypeSize = inputData_.xDtypeSize;
    tilingData->reverseSize = inputData_.reverseSize;
    tilingData->batchSize = inputData_.batchSize;
}

void ReverseSequenceSBACommonTiling::DumpTilingInfo()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::DumpTilingInfo begin");
    ReverseSequence::ReverseSequenceA1SBATilingData* tilingData =
        context_->GetTilingData<ReverseSequence::ReverseSequenceA1SBATilingData>();
    std::ostringstream str;
    str << " a1Dim:" << tilingData->a1Dim;
    str << " sDim:" << tilingData->sDim;
    str << " bDim:" << tilingData->bDim;
    str << " aDim:" << tilingData->aDim;
    str << " blockFactor:" << tilingData->blockFactor;
    str << " blockTail:" << tilingData->blockTail;
    str << " ubFactorA1:" << tilingData->ubFactorA1;
    str << " ubFactorS:" << tilingData->ubFactorS;
    str << " ubFactorB:" << tilingData->ubFactorB;
    str << " ubFactorA:" << tilingData->ubFactorA;
    str << " a1Loop:" << tilingData->a1Loop;
    str << " sLoop:" << tilingData->sLoop;
    str << " bLoop:" << tilingData->bLoop;
    str << " aLoop:" << tilingData->aLoop;
    str << " inUbSize:" << tilingData->inUbSize;
    str << " usedCoreNum:" << tilingData->usedCoreNum;
    str << " splitMode:" << tilingData->splitMode;
    str << " dtypeSize:" << tilingData->dtypeSize;
    str << " reverseSize:" << tilingData->reverseSize;
    str << " batchSize:" << tilingData->batchSize;
    OP_LOGI("SBATiling", "ReverseSequenceSBA tilingInfo is :%s", str.str().c_str());
}

ge::graphStatus ReverseSequenceSBACommonTiling::GetPlatformInfo()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::GetPlatformInfo begin");
    return GetReverseSequencePlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus ReverseSequenceSBACommonTiling::GetShapeAttrsInfo()
{
    OP_LOGD("ReverseSequenceSBACommonTiling::GetShapeAttrsInfo begin");
    return GetReverseSequenceShapeAttrsInfo(context_, inputData_);
}
}  // namespace optiling