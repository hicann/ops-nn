/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file embedding_dense_grad_v2_regbase_tiling.cc
 * \brief
 */
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "embedding_dense_grad_v2_tiling.h"
#include "embedding_dense_grad_v2_regbase_tiling.h"

namespace optiling {

const static size_t EMBEDDING_DENSE_GRAD_IN_GRAD = 0;
const static size_t EMBEDDING_DENSE_GRAD_IN_INDICES = 1;
const static size_t EMBEDDING_DENSE_GRAD_IN_POSIDX = 2;
const static size_t EMBEDDING_DENSE_GRAD_OUT_Y = 0;
const static size_t EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS = 0;
const static size_t EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX = 1;
const static size_t EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ = 2;
const static uint32_t SCATTER_FACTOR_LIMIT = 10;
const static int64_t BLOCK_SPLIT_THRESHOLD = 4096;
const static int64_t MAX_ELEWISE_AXIS_LIMIT = 8192;
const static uint64_t CACHE_LINE = 128;
const static uint64_t RESERVED_WORKSPACE = 16777216;
const static int64_t DOUBLE = 2;
const static uint32_t FOUR_BUFFER = 4;
const static int64_t EXTRA_BYTE_FOR_COUNT = 64;
const static uint32_t MIN_CACHE_LINE_SIZE = 256;
const static uint32_t MAX_CACHE_LINE_LIMIT = 512;

static uint64_t ComputeTilingKey(const std::vector<uint32_t>& args)
{
    uint64_t startValue = 1;
    uint64_t result = 1000000UL;
    constexpr uint64_t increCoef = 10;
    for (auto it = args.rbegin(); it != args.rend(); ++it) {
        result += *it * startValue;
        startValue *= increCoef;
    }

    return result;
}

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus EmbeddingDenseGradV2ForRegBase::GetPlatformInfo()
{
    OP_LOGD(opName_, "EmbeddingDenseGradV2ForRegBase GetPlatformInfo.");
    auto compileInfo = static_cast<const EmbeddingDenseGradV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_CHECK_IF((ubSize_ <= 0), OP_LOGE(opName_, "ub size is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingDenseGradV2ForRegBase::VerifyIndicesAndPosIdx()
{
    auto indicesDesc = context_->GetInputDesc(EMBEDDING_DENSE_GRAD_IN_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);

    auto posIdxDesc = context_->GetInputDesc(EMBEDDING_DENSE_GRAD_IN_POSIDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, posIdxDesc);
    auto posIdxDtype = posIdxDesc->GetDataType();

    indicesDtype_ = indicesDesc->GetDataType();
    OP_CHECK_IF(
        (indicesDtype_ != ge::DT_INT32 && indicesDtype_ != ge::DT_INT64) || posIdxDtype != ge::DT_INT32,
        OP_LOGE(opName_, "Indices dtype only support int32 or int64, and pos idx dtype only support int32."),
        return ge::GRAPH_FAILED);
    indicesDtypeSize_ = ge::GetSizeByDataType(indicesDtype_);
    OP_CHECK_IF(
        indicesDtypeSize_ <= 0,
        OP_LOGE(
            opName_, "Get invalid dtype size. indices dtype [%s], size: %u.",
            Ops::Base::ToString(indicesDtype_).c_str(), indicesDtypeSize_),
        return ge::GRAPH_FAILED);

    auto indicesTensor = context_->GetInputShape(EMBEDDING_DENSE_GRAD_IN_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesTensor);
    const gert::Shape& indicesShape = indicesTensor->GetStorageShape();

    auto posIdxTensor = context_->GetInputShape(EMBEDDING_DENSE_GRAD_IN_POSIDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, posIdxTensor);
    const gert::Shape& posIdxShape = posIdxTensor->GetStorageShape();
    OP_CHECK_IF(
        indicesShape.GetShapeSize() != posIdxShape.GetShapeSize(),
        OP_LOGE(
            opName_, "sort indices shape (%s) size is no equal to pos idx shape (%s) size.",
            Ops::Base::ToString(indicesShape).c_str(), Ops::Base::ToString(posIdxShape).c_str()),
        return ge::GRAPH_FAILED);

    scatterAxis_ = indicesShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus EmbeddingDenseGradV2ForRegBase::GetShapeAttrsInfo()
{
    OP_LOGD(opName_, "EmbeddingDenseGradV2ForRegBase GetShapeAttrsInfo.");
    // 获取grad_out的shape和datatype，获取scatter轴大小和A轴大小
    auto gradOutTensor = context_->GetInputShape(EMBEDDING_DENSE_GRAD_IN_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradOutTensor);

    const gert::Shape& gradOutShape = gradOutTensor->GetStorageShape();
    int64_t gradOutDimNum = gradOutShape.GetDimNum();
    OP_CHECK_IF(
        (gradOutDimNum < DOUBLE), OP_LOGE(opName_, "grad_out dim num (%ld) is invalid.", gradOutDimNum),
        return ge::GRAPH_FAILED);

    elewiseAxis_ = gradOutShape.GetDim(gradOutDimNum - 1);

    auto gradOutDesc = context_->GetInputDesc(EMBEDDING_DENSE_GRAD_IN_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradOutDesc);
    gradDType_ = gradOutDesc->GetDataType();
    gradDtypeSize_ = ge::GetSizeByDataType(gradDType_);
    OP_CHECK_IF(
        gradDtypeSize_ <= 0,
        OP_LOGE(
            opName_, "Get invalid dtype size. grad dtype [%s], size: %u.", Ops::Base::ToString(gradDType_).c_str(),
            gradDtypeSize_),
        return ge::GRAPH_FAILED);

    if (VerifyIndicesAndPosIdx() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t gradScatterShapeSize = 1;
    for (int64_t i = 0; i < gradOutDimNum - 1; i++) {
        gradScatterShapeSize *= gradOutShape.GetDim(i);
    }
    OP_CHECK_IF(
        gradScatterShapeSize != scatterAxis_, OP_LOGE(opName_, "indices shape is not equal to grad scatter dims."),
        return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    numWeights_ = *(attrs->GetAttrPointer<int64_t>)(EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS);
    paddingIdx_ = *(attrs->GetAttrPointer<int64_t>)(EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX);
    scaleGrad_ = *(attrs->GetAttrPointer<bool>)(EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ);

    auto gradWeightTensor = context_->GetOutputShape(EMBEDDING_DENSE_GRAD_IN_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context_, gradWeightTensor);
    const gert::Shape& gradWeightShape = gradWeightTensor->GetStorageShape();
    int64_t gradWeightDimNum = gradWeightShape.GetDimNum();
    OP_CHECK_IF(
        (gradWeightDimNum != DOUBLE || static_cast<int64_t>(gradWeightShape.GetDim(0)) != numWeights_),
        OP_LOGE(opName_, "grad_weight shape is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool EmbeddingDenseGradV2ForRegBase::IsCapable()
{
    return true;
}

void EmbeddingDenseGradV2ForRegBase::SetTilingData(
    int64_t scatterDimLoopNum, int64_t elewiseDimLoopNum, int64_t elewiseDimOuterLoopNum, int64_t normalBlockLoopNum,
    int64_t tailBlockLoopNum)
{
    OP_LOGD(opName_, "EmbeddingDenseGradV2ForRegBase SetTilingData.");
    EmbeddingDenseGradV2TilingData4RegBase* tilingData =
        context_->GetTilingData<EmbeddingDenseGradV2TilingData4RegBase>();
    tilingData->scatterDimSize = scatterAxis_;
    tilingData->elewiseDimSize = elewiseAxis_;
    tilingData->scatterDimLoopNum = scatterDimLoopNum;
    tilingData->elewiseDimLoopNum = elewiseDimLoopNum;
    tilingData->elewiseDimOuterLoopNum = elewiseDimOuterLoopNum;
    tilingData->normalBlockLoopNum = normalBlockLoopNum;
    tilingData->tailBlockLoopNum = tailBlockLoopNum;
    tilingData->numWeights = static_cast<int64_t>(numWeights_);
    tilingData->paddingIdx = paddingIdx_;
    tilingData->scatterFactor = static_cast<uint32_t>(scatterFactor_);
    tilingData->elewiseFactor = static_cast<uint32_t>(elewiseFactor_);
    tilingData->usedCoreNum = static_cast<uint32_t>(blockDim_);
    tilingData->blockNumForLastProcess = static_cast<uint32_t>(blockLastNum_);
    tilingData->elewiseDimLastPerUbProcessNum = elewiseNormalCount_;
    tilingData->elewiseDimLastProcessTailNum = elewiseTailCount_;
    tilingData->elewiseDimLastNormalLoopNum = normalBlockLastLoop_;
    tilingData->elewiseDimLastTailLoopNum = tailBlockLastLoop_;
}

bool EmbeddingDenseGradV2ForRegBase::DoUBTilingSingle(
    int64_t avaliableUbSize, uint32_t elewiseAligned, int64_t resBufSize)
{
    // 1、切UB
    int64_t lastResBuf = Ops::Base::CeilAlign(resBufSize, static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_)));
    avaliableUbSize = avaliableUbSize - lastResBuf - EXTRA_BYTE_FOR_COUNT - EXTRA_BYTE_FOR_COUNT;
    if (avaliableUbSize <= 0) {
        return false;
    }

    // elewise轴大也无法开double buffer，因为scalar阻塞
    uint32_t indicesBuff = 1;
    uint32_t posIdexBuff = 1;
    uint32_t noDupCountBuff = 1;
    uint32_t inputBuff = elewiseAligned;
    uint32_t outputBuff = inputBuff;
    uint32_t scatterPerUb =
        static_cast<uint32_t>(avaliableUbSize) /
        ((indicesBuff + noDupCountBuff) * indicesDtypeSize_ + posIdexBuff * static_cast<uint32_t>(sizeof(int32_t)) +
         (inputBuff + outputBuff) * gradDtypeSize_);

    uint32_t sFactor = Ops::Base::FloorAlign(scatterPerUb, Ops::Base::GetUbBlockSize(context_)); // 一次ub处理的行数
    if (sFactor <= SCATTER_FACTOR_LIMIT) {
        return false;
    }

    scatterFactor_ = sFactor;
    return true;
}

void EmbeddingDenseGradV2ForRegBase::FinalizeProcessOpTiling()
{
    uint32_t totalIndicesNumber = blockDim_ * DOUBLE - 1;
    uint32_t oneBlkSize = Ops::Base::GetUbBlockSize(context_);
    uint32_t indexCountInBlock = oneBlkSize / indicesDtypeSize_;
    uint32_t indicesBufSize = Ops::Base::CeilAlign(totalIndicesNumber + 1, indexCountInBlock) * indicesDtypeSize_;
    uint32_t indexBufferCount = FOUR_BUFFER;
    uint32_t freqBufferSize =
        Ops::Base::CeilAlign(totalIndicesNumber + 1, indexCountInBlock) * oneBlkSize; // 包含Begin和End
    uint32_t avaliableUb = static_cast<uint32_t>(
        ubSize_ - static_cast<uint64_t>(EXTRA_BYTE_FOR_COUNT * DOUBLE) -
        static_cast<uint64_t>(indicesBufSize * indexBufferCount + freqBufferSize));
    if (oneBlkSize == 0U) {
        OP_LOGD(opName_, "EmbeddingDenseGradV2ForRegBase get ub blocksize failed.");
    } else {
        avaliableUb = avaliableUb / oneBlkSize * oneBlkSize;
    }
    elewiseNormalCount_ = avaliableUb / ((static_cast<uint32_t>(sizeof(float)) + gradDtypeSize_) * totalIndicesNumber);
    uint32_t perBlkNum = oneBlkSize / static_cast<uint32_t>(sizeof(float));
    elewiseNormalCount_ = elewiseNormalCount_ / perBlkNum * perBlkNum;

    uint64_t elewiseLoopNum = Ops::Base::CeilDiv(elewiseAxis_, static_cast<int64_t>(elewiseNormalCount_));
    normalBlockLastLoop_ = Ops::Base::CeilDiv(elewiseLoopNum, static_cast<uint64_t>(blockDim_));
    blockLastNum_ = Ops::Base::CeilDiv(elewiseLoopNum, normalBlockLastLoop_);
    tailBlockLastLoop_ = elewiseLoopNum - (blockLastNum_ - 1UL) * normalBlockLastLoop_;
    elewiseTailCount_ =
        static_cast<uint32_t>(elewiseAxis_) - (static_cast<uint32_t>(elewiseLoopNum) - 1U) * elewiseNormalCount_;
}

// 3、计算数据切分TilingData
ge::graphStatus EmbeddingDenseGradV2ForRegBase::DoOpTiling()
{
    OP_LOGD(opName_, "EmbeddingDenseGradV2ForRegBase DoOpTiling.");
    elewiseFactor_ = static_cast<uint32_t>(elewiseAxis_);
    if (elewiseAxis_ * static_cast<int64_t>(gradDtypeSize_) > static_cast<int64_t>(MAX_CACHE_LINE_LIMIT)) {
        elewiseFactor_ = MAX_CACHE_LINE_LIMIT / gradDtypeSize_;
    }

    uint32_t elewiseAligned = Ops::Base::CeilAlign(
        elewiseFactor_,
        static_cast<uint32_t>(Ops::Base::GetUbBlockSize(context_) / gradDtypeSize_)); // ub内列方向基本块大小
    int64_t avaliableUbSize = static_cast<int64_t>(ubSize_);
    int64_t scatterDimLoopNum = 0;

    int64_t elewiseTotalSize = elewiseAxis_ * static_cast<int64_t>(sizeof(float));
    int64_t resBufSize = elewiseTotalSize > MAX_ELEWISE_AXIS_LIMIT ? MAX_ELEWISE_AXIS_LIMIT : elewiseTotalSize;

    do {
        if (!DoUBTilingSingle(avaliableUbSize, elewiseAligned, resBufSize)) {
            break;
        }

        scatterDimLoopNum = Ops::Base::CeilDiv(scatterAxis_, static_cast<int64_t>(scatterFactor_));
        if (scatterDimLoopNum >= totalCoreNum_) {
            break;
        }

        avaliableUbSize = avaliableUbSize - BLOCK_SPLIT_THRESHOLD;
    } while (avaliableUbSize > BLOCK_SPLIT_THRESHOLD);

    int64_t elewiseInnerNum =
        elewiseTotalSize > MAX_ELEWISE_AXIS_LIMIT ? MAX_ELEWISE_AXIS_LIMIT / sizeof(float) : elewiseAxis_;
    int64_t elewiseDimLoopNum = Ops::Base::CeilDiv(elewiseInnerNum, static_cast<int64_t>(elewiseAligned));
    int64_t elewiseDimOuterLoopNum = Ops::Base::CeilDiv(elewiseTotalSize, MAX_ELEWISE_AXIS_LIMIT);

    // 仅在scatter轴分核
    int64_t normalBlockLoopNum = Ops::Base::CeilDiv(scatterDimLoopNum, totalCoreNum_);
    blockDim_ = Ops::Base::CeilDiv(scatterDimLoopNum, normalBlockLoopNum);
    int64_t tailBlockLoopNum = scatterDimLoopNum - (blockDim_ - 1) * normalBlockLoopNum;

    FinalizeProcessOpTiling();

    SetTilingData(scatterDimLoopNum, elewiseDimLoopNum, elewiseDimOuterLoopNum, normalBlockLoopNum, tailBlockLoopNum);
    return ge::GRAPH_SUCCESS;
}

// 4、计算高阶API的TilingData
ge::graphStatus EmbeddingDenseGradV2ForRegBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t EmbeddingDenseGradV2ForRegBase::GetTilingKey() const
{
    OP_LOGD(opName_, "EmbeddingDenseGradV2ForRegBase GetTilingKey.");

    uint32_t isSplitCol = static_cast<uint32_t>(elewiseAxis_ > static_cast<int64_t>(elewiseFactor_));
    uint64_t tilingKey = ComputeTilingKey({static_cast<uint32_t>(scaleGrad_), isSplitCol});

    return tilingKey;
}

// 6、计算Workspace 大小
ge::graphStatus EmbeddingDenseGradV2ForRegBase::GetWorkspaceSize()
{
    workspaceSize_ = RESERVED_WORKSPACE + CACHE_LINE;
    workspaceSize_ +=
        blockDim_ * DOUBLE * (CACHE_LINE * DOUBLE + Ops::Base::CeilAlign(elewiseAxis_ * sizeof(float), CACHE_LINE));

    return ge::GRAPH_SUCCESS;
}

// 7、保存Tiling数据
ge::graphStatus EmbeddingDenseGradV2ForRegBase::PostTiling()
{
    OP_LOGD(opName_, "EmbeddingDenseGradV2ForRegBase PostTiling.");

    // 设置workspace大小
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    auto res = context_->SetBlockDim(static_cast<uint32_t>(blockDim_));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetBlockDim failed."), return ge::GRAPH_FAILED);

    context_->SetScheduleMode(1);
    res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(opName_, "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void EmbeddingDenseGradV2ForRegBase::DumpTilingInfo()
{
    std::ostringstream info;
    EmbeddingDenseGradV2TilingData4RegBase* tilingData =
        context_->GetTilingData<EmbeddingDenseGradV2TilingData4RegBase>();

    info << "scatterAxis: " << scatterAxis_ << std::endl;
    info << "elewiseAxis: " << elewiseAxis_ << std::endl;
    info << "scatterDimLoopNum: " << tilingData->scatterDimLoopNum << std::endl;
    info << "elewiseDimLoopNum: " << tilingData->elewiseDimLoopNum << std::endl;
    info << "normalBlockLoopNum: " << tilingData->normalBlockLoopNum << std::endl;
    info << "tailBlockLoopNum: " << tilingData->tailBlockLoopNum << std::endl;
    info << "numWeights: " << numWeights_ << std::endl;
    info << "paddingIdx: " << paddingIdx_ << std::endl;
    info << "scatterFactor: " << scatterFactor_ << std::endl;
    info << "elewiseFactor: " << elewiseFactor_ << std::endl;
    info << "usedCoreNum: " << blockDim_ << std::endl;

    OP_LOGI(opName_, "%s", info.str().c_str());
}

} // namespace optiling