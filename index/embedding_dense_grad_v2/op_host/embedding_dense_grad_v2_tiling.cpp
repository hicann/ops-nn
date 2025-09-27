/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_v2.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "embedding_dense_grad_v2_regbase_tiling.h"
#include "embedding_dense_grad_v2_tiling.h"

namespace optiling {
constexpr uint64_t VEC_PROCESS_SIZE = 256;
constexpr uint64_t SIZE_OF_FP32 = 4;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t RESERVED_UB_SIZE = 20480;
constexpr uint64_t USE_IDX_NUM_IN_UB = 3;
constexpr uint64_t USE_GRAD_NUM_IN_UB = 3;

constexpr uint64_t SMALL_DIM_THRESHOLD = 512;
constexpr uint64_t CAST_MAX_NUM = 16777216;
constexpr uint64_t ALIGN_32_NUM_SPCAE = 300;
constexpr uint64_t UB_SORT_PART = 10;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ = 2;

class EmbeddingDenseGradV2Tiling
{
public:
    explicit EmbeddingDenseGradV2Tiling(gert::TilingContext* context) : tilingContext_(context)
    {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline void SetTilingKeyMode() const;
    inline void BaseTiling(const int64_t gradRow);
    inline void Tiling4Scale();
    inline void Tiling4Deterministic(const int64_t gradRow);
    inline void CalMaxFormerNum(uint64_t ubSizeLeft);
    inline void Tiling4SmallDim(const int64_t gradRow);
    inline bool CheckIsSmallDim(const uint64_t gradLastDim) const;

    EmbeddingDenseGradV2TilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;
    uint64_t coreNum_ = 0;
    uint64_t numWeights_ = 0;
    uint64_t embeddingDim_ = 0;
    uint64_t paddingIdx_ = 0;
    uint64_t ubSize_ = 0;
    bool scaleGrad_ = false;

private:
    uint64_t formerRow_ = 0;
    uint64_t tailRow_ = 0;
    uint64_t formerRowRepTime_ = 0;
    uint64_t computeMask_ = 0;
    uint64_t formerComputeRepTime_ = 0;
    uint64_t formerComputeFormerNum_ = 0;
    uint64_t formerComputeTailNum_ = 0;
    uint64_t tailComputeRepTime_ = 0;
    uint64_t tailComputeFormerNum_ = 0;
    uint64_t tailComputeTailNum_ = 0;

private:
    // small dim
    uint64_t partNum_ = 0;
    uint64_t formerCopyRow_ = 0;
    uint64_t tailCopyRow_ = 0;
    uint64_t formerCopyTime_ = 0;
    uint64_t tailCopyTime_ = 0;
    uint64_t maxRowInUb_ = 0;
    uint64_t formerLastRow_ = 0;
    uint64_t tailLastRow_ = 0;

private:
    // scale
    uint64_t tailCoreRowNum_ = 0;
    uint64_t formerCoreRowNum_ = 0;
    uint64_t scaleMask_ = 0;
    uint64_t scaleRepTime_ = 0;
    uint64_t scaleFormerNum_ = 0;
    uint64_t scaleTailNum_ = 0;
    uint64_t formerCoreRowRepTime_ = 0;

private:
    // determin
    bool isDeterministMode_ = false;
    uint64_t tailRowNum_ = 0;
    uint64_t formerRowNum_ = 0;
    uint64_t formerRowNumRepTime_ = 0;
    uint64_t determinMask_ = 0;
    uint64_t formerDeterminRepTime_ = 0;
    uint64_t formerDeterminFormerNum_ = 0;
    uint64_t formerDeterminTailNum_ = 0;
    uint64_t tailDeterminRepTime_ = 0;
    uint64_t tailDeterminFormerNum_ = 0;
    uint64_t tailDeterminTailNum_ = 0;
    uint64_t gradRow_ = 0;

private:
    // big shape
    uint64_t tailEmbeddingDim_ = 0;
    uint64_t formerEmbeddingDim_ = 0;
    uint64_t formerDimRepTime_ = 0;
    uint64_t maxFormerNum = 0; // max formerEmbeddingDim_ that ub can calculate
};

inline void EmbeddingDenseGradV2Tiling::SetTilingKeyMode() const
{
    bool isSmallDim = CheckIsSmallDim(embeddingDim_);
    uint64_t tilingKey = Ops::NN::Optiling::RecursiveSum(scaleGrad_, isDeterministMode_, isSmallDim);
    tilingContext_->SetTilingKey(tilingKey);
}

inline void EmbeddingDenseGradV2Tiling::Tiling4Scale()
{
    OP_LOGD(tilingContext_, "scaleTiling start");
    tailCoreRowNum_ = numWeights_ / coreNum_;
    formerCoreRowNum_ = tailCoreRowNum_ + 1UL;
    formerCoreRowRepTime_ = numWeights_ % coreNum_;
    scaleMask_ = VEC_PROCESS_SIZE / SIZE_OF_FP32;

    // formerDim compute params
    formerComputeRepTime_ = formerEmbeddingDim_ / scaleMask_;
    formerComputeFormerNum_ = formerComputeRepTime_ * scaleMask_;
    formerComputeTailNum_ = formerEmbeddingDim_ - formerComputeFormerNum_;

    // tailDim compute params
    tailComputeRepTime_ = tailEmbeddingDim_ / scaleMask_;
    tailComputeFormerNum_ = tailComputeRepTime_ * scaleMask_;
    tailComputeTailNum_ = tailEmbeddingDim_ - tailComputeFormerNum_;
    OP_LOGD(tilingContext_, "scaleTiling finish");
}

inline void EmbeddingDenseGradV2Tiling::BaseTiling(const int64_t gradRow)
{
    OP_LOGD(tilingContext_, "BaseTiling start");
    gradRow_ = static_cast<uint64_t>(gradRow);
    tailRow_ = static_cast<uint64_t>(gradRow) / coreNum_;
    formerRow_ = tailRow_ + 1UL;
    formerRowRepTime_ = static_cast<uint64_t>(gradRow) % coreNum_;

    formerEmbeddingDim_ = embeddingDim_ <= maxFormerNum ? embeddingDim_ : maxFormerNum;
    formerDimRepTime_ = embeddingDim_ / formerEmbeddingDim_;
    tailEmbeddingDim_ = embeddingDim_ - formerEmbeddingDim_ * formerDimRepTime_;

    // formerDim compute params
    computeMask_ = VEC_PROCESS_SIZE / SIZE_OF_FP32;
    formerComputeRepTime_ = formerEmbeddingDim_ / computeMask_;
    formerComputeFormerNum_ = formerComputeRepTime_ * computeMask_;
    formerComputeTailNum_ = formerEmbeddingDim_ - formerComputeFormerNum_;
    // tailDim compute params
    tailComputeRepTime_ = tailEmbeddingDim_ / computeMask_;
    tailComputeFormerNum_ = tailComputeRepTime_ * computeMask_;
    tailComputeTailNum_ = tailEmbeddingDim_ - tailComputeFormerNum_;

    OP_LOGD(tilingContext_, "BaseTiling finish");
    if (scaleGrad_) {
        Tiling4Scale();
    }
}

inline void EmbeddingDenseGradV2Tiling::Tiling4Deterministic(const int64_t gradRow)
{
    OP_LOGD(tilingContext_, "determinist tiling start");
    gradRow_ = static_cast<uint64_t>(gradRow);
    tailRowNum_ = static_cast<uint64_t>(gradRow) / coreNum_;
    formerRowNum_ = tailRowNum_ + 1UL;
    formerRowNumRepTime_ = static_cast<uint64_t>(gradRow) % coreNum_;

    formerEmbeddingDim_ = embeddingDim_ <= maxFormerNum ? embeddingDim_ : maxFormerNum;
    formerDimRepTime_ = embeddingDim_ / formerEmbeddingDim_;
    tailEmbeddingDim_ = embeddingDim_ - formerEmbeddingDim_ * formerDimRepTime_;

    // formerDim determin compute params
    determinMask_ = VEC_PROCESS_SIZE / SIZE_OF_FP32;
    formerDeterminRepTime_ = formerEmbeddingDim_ / determinMask_;
    formerDeterminFormerNum_ = formerDeterminRepTime_ * determinMask_;
    formerDeterminTailNum_ = formerEmbeddingDim_ - formerDeterminFormerNum_;
    // tailDim determin compute params
    tailDeterminRepTime_ = tailEmbeddingDim_ / determinMask_;
    tailDeterminFormerNum_ = tailDeterminRepTime_ * determinMask_;
    tailDeterminTailNum_ = tailEmbeddingDim_ - tailDeterminFormerNum_;

    OP_LOGD(tilingContext_, "determinist tiling finish");
    if (scaleGrad_) {
        Tiling4Scale();
    }
}

inline void EmbeddingDenseGradV2Tiling::CalMaxFormerNum(uint64_t ubSizeLeft)
{
    auto const gradDtype = tilingContext_->GetInputDesc(0)->GetDataType();
    uint64_t idxAlignNum = BLOCK_SIZE / sizeof(int);
    uint64_t gradAlignNum = BLOCK_SIZE / sizeof(gradDtype);
    ubSizeLeft -= RESERVED_UB_SIZE + idxAlignNum * sizeof(int) * USE_IDX_NUM_IN_UB;
    uint64_t availableUbForGrad = ubSizeLeft > 0UL ? ubSizeLeft : 0UL;
    maxFormerNum = (availableUbForGrad / (gradAlignNum * sizeof(gradDtype) * USE_GRAD_NUM_IN_UB)) * gradAlignNum;
}

inline void EmbeddingDenseGradV2Tiling::Tiling4SmallDim(const int64_t gradRow)
{
    OP_LOGD(tilingContext_, "small dim tiling start");

    formerEmbeddingDim_ = embeddingDim_ <= maxFormerNum ? embeddingDim_ : maxFormerNum;
    formerDimRepTime_ = embeddingDim_ / formerEmbeddingDim_;
    tailEmbeddingDim_ = embeddingDim_ - formerEmbeddingDim_ * formerDimRepTime_;
    partNum_ = 1UL;
    formerCopyRow_ = static_cast<uint64_t>(gradRow) / coreNum_;
    tailCopyRow_ = static_cast<uint64_t>(gradRow) - formerCopyRow_ * (coreNum_ - 1UL);
    // ubSize4Data = x * embeddingDim_ + (x + 31) * 10 + embeddingDim_
    uint64_t dataInBlock = BLOCK_SIZE / sizeof(float);
    uint64_t divNum = UB_SORT_PART + (embeddingDim_ + dataInBlock - 1) / dataInBlock * dataInBlock;
    maxRowInUb_ = ((ubSize_ - RESERVED_UB_SIZE) / SIZE_OF_FP32 - embeddingDim_ - VEC_PROCESS_SIZE) / divNum;
    formerCopyTime_ = (formerCopyRow_ + maxRowInUb_ - 1UL) / maxRowInUb_;
    tailCopyTime_ = (tailCopyRow_ + maxRowInUb_ - 1UL) / maxRowInUb_;
    formerLastRow_ = formerCopyRow_ - maxRowInUb_ * (formerCopyTime_ - 1UL);
    tailLastRow_ = tailCopyRow_ - maxRowInUb_ * (tailCopyTime_ - 1UL);
    OP_LOGD(tilingContext_, "small dim tiling end");
    if (scaleGrad_) {
        Tiling4Scale();
    }
}

ge::graphStatus EmbeddingDenseGradV2Tiling::Init()
{
    OP_LOGD(tilingContext_, "Tiling initing");
    isDeterministMode_ = (tilingContext_->GetDeterministic() == 1);
    auto compileInfo = static_cast<const EmbeddingDenseGradV2CompileInfo*>(tilingContext_->GetCompileInfo());
    auto attrs = tilingContext_->GetAttrs();
    auto selfShape = tilingContext_->GetInputShape(0)->GetStorageShape();
    int64_t gradRow = 1;
    for (size_t i = 0; i < selfShape.GetDimNum() - 1; i++) {
        gradRow *= selfShape.GetDim(i);
    }
    embeddingDim_ = selfShape.GetDim(selfShape.GetDimNum() - 1);
    numWeights_ = *(attrs->GetAttrPointer<uint64_t>)(EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS);
    paddingIdx_ = *(attrs->GetAttrPointer<uint64_t>)(EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX);
    scaleGrad_ = *(attrs->GetAttrPointer<bool>)(EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ);
    size_t alignNum = BLOCK_SIZE / sizeof(int64_t);
    size_t scaleWorkspaceSize = ((numWeights_ + alignNum - 1UL) / alignNum) * alignNum * sizeof(int64_t);
    size_t sysWorkspaceSize = 16UL * 1024UL * 1024UL;
    sysWorkspaceSize = scaleGrad_ ? sysWorkspaceSize + scaleWorkspaceSize : sysWorkspaceSize;
    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;

    coreNum_ = scaleGrad_ ? std::min(
                                compileInfo->totalCoreNum,
                                std::min(static_cast<uint64_t>(numWeights_), static_cast<uint64_t>(gradRow))) :
                            std::min(compileInfo->totalCoreNum, static_cast<uint64_t>(gradRow));
    ubSize_ = compileInfo->ubSizePlatForm;
    if (coreNum_ == 0UL || embeddingDim_ == 0UL) {
        OP_LOGE(tilingContext_, "coreNum %lu, embeddingDim %lu", coreNum_, embeddingDim_);
        return ge::GRAPH_FAILED;
    }

    CalMaxFormerNum(ubSize_);
    OP_CHECK_IF((maxFormerNum == 0), OP_LOGE(tilingContext_, "Do not have enough ub size."), return ge::GRAPH_FAILED);
    SetTilingKeyMode();
    tilingContext_->SetNeedAtomic(true);
    if (isDeterministMode_) {
        Tiling4Deterministic(gradRow);
    } else if (CheckIsSmallDim(embeddingDim_)) {
        Tiling4SmallDim(gradRow);
    } else {
        BaseTiling(gradRow);
    }
    OP_LOGD(tilingContext_, "Tiling inited");
    return ge::GRAPH_SUCCESS;
}

bool EmbeddingDenseGradV2Tiling::CheckIsSmallDim(const uint64_t gradLastDim) const
{
    return !isDeterministMode_ && gradLastDim <= SMALL_DIM_THRESHOLD && numWeights_ <= CAST_MAX_NUM;
}

ge::graphStatus EmbeddingDenseGradV2Tiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(coreNum_);
    tilingData_.params.set_tailRowNum(tailRow_);
    tilingData_.params.set_formerRowNum(formerRow_);
    tilingData_.params.set_formerRowRepTime(formerRowRepTime_);
    tilingData_.params.set_computeMask(computeMask_);
    tilingData_.params.set_formerComputeRepTime(formerComputeRepTime_);
    tilingData_.params.set_formerComputeFormerNum(formerComputeFormerNum_);
    tilingData_.params.set_formerComputeTailNum(formerComputeTailNum_);
    tilingData_.params.set_tailComputeRepTime(tailComputeRepTime_);
    tilingData_.params.set_tailComputeFormerNum(tailComputeFormerNum_);
    tilingData_.params.set_tailComputeTailNum(tailComputeTailNum_);
    tilingData_.params.set_embeddingDim(embeddingDim_);
    tilingData_.params.set_numWeights(numWeights_);
    tilingData_.params.set_paddingIdx(paddingIdx_);
    tilingData_.params.set_scaleGradByFreq(scaleGrad_);
    tilingData_.params.set_formerDimRepTime(formerDimRepTime_);
    tilingData_.params.set_formerEmbeddingDim(formerEmbeddingDim_);
    tilingData_.params.set_tailEmbeddingDim(tailEmbeddingDim_);

    tilingData_.scaleTiling.set_tailCoreRowNum(tailCoreRowNum_);
    tilingData_.scaleTiling.set_formerCoreRowNum(formerCoreRowNum_);
    tilingData_.scaleTiling.set_formerCoreRowRepTime(formerCoreRowRepTime_);
    tilingData_.scaleTiling.set_mask(scaleMask_);
    tilingData_.scaleTiling.set_formerComputeRepTime(formerComputeRepTime_);
    tilingData_.scaleTiling.set_formerComputeFormerNum(formerComputeFormerNum_);
    tilingData_.scaleTiling.set_formerComputeTailNum(formerComputeTailNum_);
    tilingData_.scaleTiling.set_tailComputeRepTime(tailComputeRepTime_);
    tilingData_.scaleTiling.set_tailComputeFormerNum(tailComputeFormerNum_);
    tilingData_.scaleTiling.set_tailComputeTailNum(tailComputeTailNum_);

    tilingData_.determinTiling.set_gradRow(gradRow_);
    tilingData_.determinTiling.set_tailRowNum(tailRowNum_);
    tilingData_.determinTiling.set_formerRowNum(formerRowNum_);
    tilingData_.determinTiling.set_formerRowNumRepTime(formerRowNumRepTime_);
    tilingData_.determinTiling.set_computeMask(determinMask_);
    tilingData_.determinTiling.set_formerComputeRepTime(formerDeterminRepTime_);
    tilingData_.determinTiling.set_formerComputeFormerNum(formerDeterminFormerNum_);
    tilingData_.determinTiling.set_formerComputeTailNum(formerDeterminTailNum_);
    tilingData_.determinTiling.set_tailComputeRepTime(tailDeterminRepTime_);
    tilingData_.determinTiling.set_tailComputeFormerNum(tailDeterminFormerNum_);
    tilingData_.determinTiling.set_tailComputeTailNum(tailDeterminTailNum_);

    tilingData_.smallDimTiling.set_partNum(partNum_);
    tilingData_.smallDimTiling.set_formerCopyRow(formerCopyRow_);
    tilingData_.smallDimTiling.set_tailCopyRow(tailCopyRow_);
    tilingData_.smallDimTiling.set_formerCopyTime(formerCopyTime_);
    tilingData_.smallDimTiling.set_tailCopyTime(tailCopyTime_);
    tilingData_.smallDimTiling.set_maxRowInUb(maxRowInUb_);
    tilingData_.smallDimTiling.set_formerLastRow(formerLastRow_);
    tilingData_.smallDimTiling.set_tailLastRow(tailLastRow_);

    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void EmbeddingDenseGradV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "maxFormerNum:            %lu", maxFormerNum);
    OP_LOGD(tilingContext_, "coreNum:                 %lu", coreNum_);
    OP_LOGD(tilingContext_, "numWeights:              %lu", numWeights_);
    OP_LOGD(tilingContext_, "embeddingDim:            %lu", embeddingDim_);
    OP_LOGD(tilingContext_, "paddingIdx:              %lu", paddingIdx_);
    OP_LOGD(tilingContext_, "scaleGrad:               %d", scaleGrad_);
    OP_LOGD(tilingContext_, "gradRow:                 %lu", gradRow_);
    OP_LOGD(tilingContext_, "tailRow:                 %lu", tailRow_);
    OP_LOGD(tilingContext_, "formerRow:               %lu", formerRow_);
    OP_LOGD(tilingContext_, "formerRowRepTime:        %lu", formerRowRepTime_);

    OP_LOGD(tilingContext_, "formerDimRepTime:        %lu", formerDimRepTime_);
    OP_LOGD(tilingContext_, "formerEmbeddingDim:      %lu", formerEmbeddingDim_);
    OP_LOGD(tilingContext_, "tailEmbeddingDim:        %lu", tailEmbeddingDim_);

    OP_LOGD(tilingContext_, "computeMask:             %lu", computeMask_);
    OP_LOGD(tilingContext_, "formerComputeRepTime:    %lu", formerComputeRepTime_);
    OP_LOGD(tilingContext_, "formerComputeFormerNum:  %lu", formerComputeFormerNum_);
    OP_LOGD(tilingContext_, "formerComputeTailNum:    %lu", formerComputeTailNum_);

    OP_LOGD(tilingContext_, "tailComputeRepTime:      %lu", tailComputeRepTime_);
    OP_LOGD(tilingContext_, "tailComputeFormerNum:    %lu", tailComputeFormerNum_);
    OP_LOGD(tilingContext_, "tailComputeTailNum:      %lu", tailComputeTailNum_);

    OP_LOGD(tilingContext_, "tailCoreRow:             %lu", tailCoreRowNum_);
    OP_LOGD(tilingContext_, "formerCoreRow:           %lu", formerCoreRowNum_);
    OP_LOGD(tilingContext_, "formerCoreRowRepTime:    %lu", formerCoreRowRepTime_);
    OP_LOGD(tilingContext_, "scaleMask:               %lu", scaleMask_);
    OP_LOGD(tilingContext_, "scaleRepTime:            %lu", scaleRepTime_);
    OP_LOGD(tilingContext_, "scaleFormerNum:          %lu", scaleFormerNum_);
    OP_LOGD(tilingContext_, "scaleTailNum:            %lu", scaleTailNum_);

    OP_LOGD(tilingContext_, "tailRowNum:              %lu", tailRowNum_);
    OP_LOGD(tilingContext_, "formerRowNum:            %lu", formerRowNum_);
    OP_LOGD(tilingContext_, "formerRowNumRepTime:     %lu", formerRowNumRepTime_);
    OP_LOGD(tilingContext_, "determinMask:            %lu", determinMask_);
    OP_LOGD(tilingContext_, "formerDeterminRepTime:   %lu", formerDeterminRepTime_);
    OP_LOGD(tilingContext_, "formerDeterminFormerNum: %lu", formerDeterminFormerNum_);
    OP_LOGD(tilingContext_, "formerDeterminTailNum:   %lu", formerDeterminTailNum_);

    OP_LOGD(tilingContext_, "tailDeterminRepTime:     %lu", tailDeterminRepTime_);
    OP_LOGD(tilingContext_, "tailDeterminFormerNum:   %lu", tailDeterminFormerNum_);
    OP_LOGD(tilingContext_, "tailDeterminTailNum:     %lu", tailDeterminTailNum_);

    OP_LOGD(tilingContext_, "partNum:                 %lu", partNum_);
    OP_LOGD(tilingContext_, "formerCopyRow:           %lu", formerCopyRow_);
    OP_LOGD(tilingContext_, "tailCopyRow:             %lu", tailCopyRow_);
    OP_LOGD(tilingContext_, "formerCopyTime:          %lu", formerCopyTime_);
    OP_LOGD(tilingContext_, "tailCopyTime:            %lu", tailCopyTime_);
    OP_LOGD(tilingContext_, "maxRowInUb:              %lu", maxRowInUb_);
    OP_LOGD(tilingContext_, "formerLastRow:           %lu", formerLastRow_);
    OP_LOGD(tilingContext_, "tailLastRow:             %lu", tailLastRow_);
}

ge::graphStatus TilingEmbeddingDenseGradV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("EmbeddingDenseGradV2", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Tiling4EmbeddingDenseGradV2 enter.");

    const EmbeddingDenseGradV2CompileInfo* compile_info =
        static_cast<const EmbeddingDenseGradV2CompileInfo*>(context->GetCompileInfo());

    if (compile_info->isRegBase) {
        OP_LOGD(context->GetNodeName(), "Tiling EmbeddingDenseGradV2 RegBase start");
        EmbeddingDenseGradV2ForRegBase tilingObj(context);
        return tilingObj.DoTiling();
    }

    EmbeddingDenseGradV2Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetKernelTiling();
}

static bool IsRegbaseSocVersion(platform_ascendc::SocVersion version)
{
    const static std::set<platform_ascendc::SocVersion> regbaseSocVersions = {
        platform_ascendc::SocVersion::ASCEND910_95
    };

    return regbaseSocVersions.find(version) != regbaseSocVersions.end();
}

static bool IsRegbaseSocVersion(const gert::TilingParseContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto socVersion = ascendcPlatform.GetSocVersion();
    return IsRegbaseSocVersion(socVersion);
}

ge::graphStatus TilingPrepareForEmbeddingDenseGradV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For EmbeddingDenseGradV2 start");
    auto compileInfo = context->GetCompiledInfo<EmbeddingDenseGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->isRegBase = IsRegbaseSocVersion(context);
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
        return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
    OP_LOGD(context->GetNodeName(), "Tiling prepare for embedding dense grad v2 end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EmbeddingDenseGradV2)
    .Tiling(TilingEmbeddingDenseGradV2)
    .TilingParse<EmbeddingDenseGradV2CompileInfo>(TilingPrepareForEmbeddingDenseGradV2);
} // namespace optiling