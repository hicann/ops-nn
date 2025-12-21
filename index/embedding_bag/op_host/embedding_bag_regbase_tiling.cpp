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
 * \file embedding_bag_regbase_tiling.cc
 * \brief
 */

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "embedding_bag_regbase_tiling.h"

namespace optiling {

constexpr int64_t WEIGHT_IDX = 0;
constexpr int64_t INDICES_IDX = 1;
constexpr int64_t OFFSETS_IDX = 2;
constexpr int64_t PER_SAMPLE_WEIGHT_IDX = 3;
constexpr int64_t OUTPUT_IDX = 0;
constexpr int64_t OFSET2BAG_IDX = 1;
constexpr int64_t BAGSIZE_IDX = 2;
constexpr int64_t MAXINDICES_IDX = 3;
constexpr int64_t MODE_INDEX = 0;
constexpr int64_t ATTR_INCLUDE_LAST_OFFSET = 3;
constexpr int64_t ATTR_PADD_INDEX = 4;
constexpr int64_t DIM_TWO = 2;

constexpr int64_t MDOE_SUM = 0;
constexpr int64_t MDOE_MEAN = 1;
constexpr int64_t MDOE_MAX = 2;
constexpr int64_t QUE_NUM_THREE = 3;
constexpr int64_t INDICE_QUE_NUM = 3;
constexpr int64_t OFFSET_QUE_NUM = 3;
constexpr uint64_t WEIGHT_DIM_QUE_NUM_SUM_MEAN = 2;
constexpr uint64_t WEIGHT_DIM_QUE_NUM_MAX = 5;
constexpr uint64_t MAX_SIMT_EMBDDING_BYTES = 256;
constexpr uint64_t OFFSET_DIFF_TYPE_SMALL = 2;
constexpr uint64_t OFFSET_DIFF_TYPE_LARGE = 3;
constexpr int64_t INIDCES_LIMIT_BYTE = 1024 * 32;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;
constexpr int64_t BASE_BLOCK_SIZE = 2048;
constexpr int64_t DCACHE_SIZE = 32U * 1024U;

constexpr int64_t UB_RESERVED_BUFF = 256;
constexpr int64_t DOUBLE_BUF = 2;
constexpr int64_t INDICE_QUE_NUM_2D = 3;
constexpr int64_t INDICES_LIMIT_BYTE = 4L * 1024L;
constexpr int64_t PERSMAPLE_LIMIT_BYTE = 4L * 1024L;
constexpr int64_t OFFSETS_LIMIT_BYTE = 2L * 1024L;

constexpr int64_t BASE_BLOCK_COPY_ALIGN = 512;
constexpr uint64_t TILING_KEY_INDICES_1D = 100;
constexpr uint64_t TILING_KEY_INDICES_2D = 200;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t WEIGHT_DIM_LIMIT = 128L;
constexpr uint64_t TILING_KEY_SIMT_1D = 300;
constexpr uint64_t TILING_KEY_SIMT_2D = 400;

bool EmbeddingBagRegBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus EmbeddingBagRegBaseTiling::GetPlatformInfo()
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    ubSize_ -= DCACHE_SIZE;
    OP_CHECK_IF(
        (aivNum <= 0), OP_LOGE(opName, "fail to get coreNum."),
        return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingBagRegBaseTiling::GetShapeAttrsInfo()
{
    auto weight = context_->GetInputShape(WEIGHT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weight);
    auto weightShape = weight->GetStorageShape();

    OP_CHECK_IF(weightShape.GetDimNum() != DIM_TWO,
        OP_LOGE(opName, "weight shape check failed."), return ge::GRAPH_FAILED);
    auto numEmbeddings_ = weightShape.GetDim(0);
    embeddingDim_ = weightShape.GetDim(1);
    weightShapeSize_ = static_cast<uint64_t>(embeddingDim_ * numEmbeddings_);
    auto weightDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightDesc);
    auto weightType = weightDesc->GetDataType();
    weightTypeSize_ = ge::GetSizeByDataType(weightType);

    auto indices = context_->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    auto indiceShape = indices->GetStorageShape();
    indicesNumel_ = indiceShape.GetShapeSize();

    auto indicesDesc = context_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    indicesType_ = indicesDesc->GetDataType();
    indicesTypeSize_ = ge::GetSizeByDataType(indicesType_);

    auto offsets = context_->GetInputShape(OFFSETS_IDX);
    if (offsets != nullptr) {
        auto offsetsShape = offsets->GetStorageShape();
        offsetsNum_ = offsetsShape.GetDim(0);

        auto offsetsDesc = context_->GetInputDesc(OFFSETS_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, offsetsDesc);
        offsetsType_ = offsetsDesc->GetDataType();
        offsetsTypeSize_ = ge::GetSizeByDataType(offsetsType_);
        if (indicesType_ != offsetsType_) {
            promoteTypeSize_ = ge::GetSizeByDataType(ge::DT_INT64);
        } else {
            promoteTypeSize_ = indicesTypeSize_;
        }
    }

    auto sampleWeight = context_->GetInputShape(PER_SAMPLE_WEIGHT_IDX);
    if (sampleWeight != nullptr) {
        auto sampleWeightShape = sampleWeight->GetStorageShape();
        sampleWeightNum_ = sampleWeightShape.GetShapeSize();
        isNeedSampleWeight_ = 1;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const std::string modeStr = std::string(attrs->GetAttrPointer<char>(MODE_INDEX));
    std::map<std::string, int64_t> modeMap = {{"sum", MDOE_SUM}, {"mean", MDOE_MEAN}, {"max", MDOE_MAX}};
    mode_ = modeMap[modeStr];
    if (mode_ != MDOE_SUM){
        isNeedSampleWeight_ = 0;
    }
    
    inclueLastOfst_ = *(attrs->GetAttrPointer<bool>)(ATTR_INCLUDE_LAST_OFFSET);
    paddingIdx_ = *(attrs->GetAttrPointer<int64_t>)(ATTR_PADD_INDEX);
    paddingIdx_ = paddingIdx_ < 0 ? paddingIdx_ + numEmbeddings_ : paddingIdx_;

    if (embeddingDim_ * weightTypeSize_ <= MAX_SIMT_EMBDDING_BYTES) {
        usedCoreNum_ = totalCoreNum_;
        isSimt_ = 1;
    }
    if (indiceShape.GetDimNum() == 1) {
        numBags_ = inclueLastOfst_ ? offsetsNum_ - 1 : offsetsNum_;
        isNeedOffset_ = 1;
    } else if (indiceShape.GetDimNum() == DIM_TWO) {
        numBags_ = indiceShape.GetDim(0);
        indiceSize_ = indiceShape.GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

std::set<int64_t> EmbeddingBagRegBaseTiling::FindUniqueCut()
{
    std::set<int64_t> result;
    int64_t upbound = std::ceil(std::sqrt(usedCoreNum_) + 1);

    for (int64_t m = 1; m < upbound; m++) {
        int64_t y = usedCoreNum_ / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

void EmbeddingBagRegBaseTiling::AutoTiling()
{
    int64_t baseNum = BASE_BLOCK_COPY_ALIGN / weightTypeSize_;
    int64_t colAlignNum = Ops::Base::CeilDiv(embeddingDim_, baseNum);
    /*
    **给定一个Shape[M,N]和block core num，找到尽可能均匀且尽量用满核的切分方式
    */
    usedCoreNum_ = std::min(totalCoreNum_, static_cast<uint64_t>(numBags_ * colAlignNum * baseNum / BASE_BLOCK_SIZE));
    usedCoreNum_ = (usedCoreNum_ == 0 ? 1 : usedCoreNum_);

    /* 给定Shape[M, N] 和 block core num
    ** M切分成m块，N切分成n块，找到m*n <= usedCoreNum, 且m*n尽可能接近usedCoreNum的所有m和n的可能
    */
    std::set<int64_t> cutSet = FindUniqueCut();
    std::vector<std::vector<int64_t>> allTiling;

    // 核先按照行方向切分，枚举m的取值
    for (int64_t m : cutSet) {
        // 行方向分核超过行方向数据量，则说明有空闲核
        if (m > numBags_) {
            continue;
        }

        int64_t n = usedCoreNum_ / m;
        n = (n < 1 ? 1 : n);
        // 列方向分核超过列方向数据量，则说明有空闲核
        if (n > colAlignNum) {
            continue;
        }

        int64_t rowNormalBlock = Ops::Base::CeilDiv(numBags_, m);
        // 列方向按照512基本块切分
        int64_t colNormalBlock = Ops::Base::CeilDiv(colAlignNum, n);

        // 正常切分块和尾块的差值计算
        int64_t delta = rowNormalBlock * colNormalBlock;
        if (m * n == static_cast<int64_t>(usedCoreNum_)) {
            if (numBags_ % m == 0 && colAlignNum % n == 0) {
                rowTileNum_ = m;
                colTileNum_ = n;
            } else if (numBags_ % m == 0) {
                delta = delta - rowNormalBlock * (colAlignNum % colNormalBlock);
            } else if (colAlignNum % n == 0) {
                delta = delta - (numBags_ % rowNormalBlock) * n;
            } else {
                delta = delta - (numBags_ % rowNormalBlock) * (colAlignNum % colNormalBlock);
            }
        }

        allTiling.push_back({m, n, m * n, delta});
    }

    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int nIndex = 1;
        constexpr int deltaIndex = 3;
        return std::make_pair(a[deltaIndex], a[nIndex]) < std::make_pair(b[deltaIndex], b[nIndex]);
    });

    rowTileNum_ = allTiling[0][0];
    colTileNum_ = allTiling[0][1];
    rowNormalNum_ = Ops::Base::FloorDiv(numBags_, rowTileNum_);
    colNormalNum_ = Ops::Base::FloorDiv(embeddingDim_, colTileNum_);
    rowTailNum_ = numBags_ - rowNormalNum_ * rowTileNum_ + rowNormalNum_;
    colTailNum_ = embeddingDim_ - colNormalNum_ * colTileNum_ + colNormalNum_;
    usedCoreNum_ = rowTileNum_ * colTileNum_;
}

int64_t EmbeddingBagRegBaseTiling::GetOffsetAlignSize1D(int64_t offsetFactor)
{
    /**
     * inQueueOffsets_ -> (offsetFactor + 1) * offsetsTypeSize_
     * bagSizeBuf -> (offsetFactor + 1) * promoteTypeSize_
     */
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t occupy = Ops::Base::CeilAlign((offsetFactor + 1) * offsetsTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign((offsetFactor + 1) * promoteTypeSize_, ubBlock);
    return occupy;
}

int64_t EmbeddingBagRegBaseTiling::GetWeightAlignSize1D(int64_t weightRowFactor, int64_t weightDimFactor)
{
    /**
    * inQueueWeight_ -> weightRowFactor * weightDimFactor * weightTypeSize_
    * outQueueY_ -> weightDimFactor * weightTypeSize_
    * maxIndicesOutBuf -> weightDimFactor * promoteTypeSize_
    * maxIndicesCalcBuf_ -> weightRowFactor * promoteTypeSize_
    * offset2BagBuf_ -> weightRowFactor * promoteTypeSize_
    * preSampleWeightBuf -> weightRowFactor * weightTypeSize_
    */
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t occupy = weightRowFactor * Ops::Base::CeilAlign(weightDimFactor * weightTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(weightDimFactor * weightTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(weightDimFactor * promoteTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(weightRowFactor * promoteTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(weightRowFactor * promoteTypeSize_, ubBlock);
    if (isNeedSampleWeight_ == 1) {
        occupy += Ops::Base::CeilAlign(weightRowFactor * ge::GetSizeByDataType(ge::DT_FLOAT), ubBlock);
    }
    return occupy;
}

void EmbeddingBagRegBaseTiling::Compute1DFactor()
{
    auto halfUbSize = ubSize_ / DOUBLE_BUF - UB_RESERVED_BUFF;
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));

    /* indices 分配4K */
    int64_t resetSize = static_cast<int64_t>(-1);
    indicesFactor_ = INDICES_LIMIT_BYTE / indicesTypeSize_;

    offsetsFactor_ = rowNormalNum_;
    offsetsFactor_ += 1;
    while (resetSize <= 0) {
        --offsetsFactor_;
        resetSize = OFFSETS_LIMIT_BYTE - GetOffsetAlignSize1D(offsetsFactor_);
    }
    offsetsFactor_ = offsetsFactor_ > rowNormalNum_ ? rowNormalNum_ : offsetsFactor_;

    int64_t aviableSizeForWeight = halfUbSize - OFFSETS_LIMIT_BYTE - INDICES_LIMIT_BYTE;
    if (isNeedSampleWeight_ == 1) {
        // 和 indices 类型 倍数关系
        aviableSizeForWeight -= PERSMAPLE_LIMIT_BYTE;
    }
    if (GetWeightAlignSize1D(1, colNormalNum_) > aviableSizeForWeight) {
        weightDimFactor_ = std::min(colNormalNum_, WEIGHT_DIM_LIMIT);
    } else {
        weightDimFactor_ = Ops::Base::CeilAlign(colNormalNum_, ubBlock) / weightTypeSize_;
    }
    weightRowFactor_ = indicesFactor_;
    resetSize = static_cast<int64_t>(-1);
    while (resetSize <= 0) {
        --weightRowFactor_;
        resetSize = aviableSizeForWeight - GetWeightAlignSize1D(weightRowFactor_, weightDimFactor_);
    }
    weightRowFactor_ = weightRowFactor_ == 0 ? 1 : weightRowFactor_;
}

int64_t EmbeddingBagRegBaseTiling::GetInidcesAlignSize2D(int64_t indicesFactor)
{
    /**
    * indicesQue -> indicesFactor_ * indiceSize_ * indicesTypeSize_
    * perSampleWeightQue -> indicesFactor_ * indiceSize_ * weightTypeSize_
    * bagSizeBuf -> indicesFactor_ * promoteTypeSize_
    * offset2BagBuf -> indiceSize_ * promoteTypeSize_
    */
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t occupy = Ops::Base::CeilAlign(indicesFactor * indiceSize_ * indicesTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(indicesFactor_ * promoteTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(indiceSize_ * promoteTypeSize_, ubBlock);
    if (isNeedSampleWeight_ == 1) {
        occupy += Ops::Base::CeilAlign(indicesFactor * indiceSize_ * weightTypeSize_, ubBlock);
    }
    return occupy;
}

int64_t EmbeddingBagRegBaseTiling::GetWeightAlignSize2D(int64_t weightRowFactor, int64_t weightDimFactor)
{
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    /**
    * sum/mean:
    * weightInQue -> weightRowFactor * weightDimFactor * weightTypeSize_
    * weightOutQue -> weightDimFactor * weightTypeSize_
    */
    if (mode_ == MDOE_SUM && isNeedSampleWeight_ == 1) {
        /* persampleWeight时weightRowFactor为1 */
        int64_t occupy = Ops::Base::CeilAlign(weightDimFactor * weightTypeSize_, ubBlock) +
                         Ops::Base::CeilAlign(weightDimFactor * weightTypeSize_, ubBlock);
        return occupy;
    }
    if (mode_ == MDOE_SUM || mode_ == MDOE_MEAN) {
        int64_t occupy = weightRowFactor * Ops::Base::CeilAlign(weightDimFactor * weightTypeSize_, ubBlock) +
                        Ops::Base::CeilAlign(weightDimFactor * weightTypeSize_, ubBlock);
        return occupy;
    }

    /**
    * max:
    * weightInQue -> weightRowFactor * weightDimFactor * weightTypeSize_
    * weightOutQue -> weightDimFactor * weightTypeSize_
    * maxIndicesOutBuf + maxIndicesCalcBuf -> weightDimFactor * promoteTypeSize_
    * compareMaskBuf -> weightDimFactor * uint8
    */
    int64_t occupy = weightRowFactor * Ops::Base::CeilAlign(weightDimFactor * weightTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(weightDimFactor * weightTypeSize_, ubBlock) +
                     Ops::Base::CeilAlign(weightDimFactor * promoteTypeSize_, ubBlock) * NUM_TWO + 
                     Ops::Base::CeilAlign(weightDimFactor * ge::GetSizeByDataType(ge::DT_UINT8), ubBlock);
    return occupy;
}

void EmbeddingBagRegBaseTiling::Compute2DFactor()
{
    auto halfUbSize = ubSize_ / DOUBLE_BUF - UB_RESERVED_BUFF;
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));

    int64_t weightBlockSize = GetWeightAlignSize2D(indiceSize_, colNormalNum_);
    int64_t indicesBlockSize = GetInidcesAlignSize2D(rowNormalNum_);

    if ((indiceSize_ * indicesTypeSize_ * NUM_TWO) > INDICES_LIMIT_BYTE) {
        isSimt_ = 1;
        return;
    }
    /* indices 分配4K */
    int64_t resetSize = static_cast<int64_t>(-1);
    int64_t aviableSizeForIndices = halfUbSize - weightBlockSize;

    if (aviableSizeForIndices >= INDICES_LIMIT_BYTE) {
        weightRowFactor_ = indiceSize_;
        if (mode_ == MDOE_SUM && isNeedSampleWeight_ == 1) {
            weightRowFactor_ = 1;
        }
        weightDimFactor_ = Ops::Base::CeilAlign(colNormalNum_ * weightTypeSize_, ubBlock) / weightTypeSize_;

        /* 确定weightDimFactor_/weightRowFactor_后更新indices的可用ub空间 */
        aviableSizeForIndices = halfUbSize - GetWeightAlignSize2D(weightRowFactor_, weightDimFactor_);
        /* indicesFactor_初始化为可能的最大值 */
        indicesFactor_ = rowNormalNum_;
        indicesFactor_ += 1;
        while (resetSize <= 0) {
            --indicesFactor_;
            resetSize = aviableSizeForIndices - GetInidcesAlignSize2D(indicesFactor_);
        }
        indicesFactor_ = indicesFactor_ == 0 ? 1 : indicesFactor_;
    } else {
        aviableSizeForIndices = std::min(INDICES_LIMIT_BYTE, indicesBlockSize);
        indicesFactor_ = rowNormalNum_;
        indicesFactor_ += 1;
        while (resetSize <= 0) {
            --indicesFactor_;
            resetSize = aviableSizeForIndices - GetInidcesAlignSize2D(indicesFactor_);
        }
        indicesFactor_ = indicesFactor_ == 0 ? 1 : indicesFactor_;
    
        /* 确定indicesFactor_后更新weight的可用ub空间 */
        int64_t aviableSizeForWeight = halfUbSize - GetInidcesAlignSize2D(indicesFactor_);
        if (GetWeightAlignSize2D(1, colNormalNum_) > aviableSizeForWeight) {
            weightDimFactor_ = std::min(colNormalNum_, 128L);
        } else {
            weightDimFactor_ = Ops::Base::CeilAlign(colNormalNum_, ubBlock) / weightTypeSize_;
        }
        weightRowFactor_ = indiceSize_;
        resetSize = static_cast<int64_t>(-1);
        while (resetSize <= 0) {
            --weightRowFactor_;
            resetSize = aviableSizeForWeight - GetWeightAlignSize2D(weightRowFactor_, weightDimFactor_);
        }
        weightRowFactor_ = weightRowFactor_ == 0 ? 1 : weightRowFactor_;
    }
}

void EmbeddingBagRegBaseTiling::SetTilingData()
{
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_nBags(numBags_);
    tilingData_.set_embeddingDim(embeddingDim_);
    tilingData_.set_paddingIdx(paddingIdx_);
    tilingData_.set_indiceSize(indiceSize_);
    tilingData_.set_rowTileNum(rowTileNum_);
    tilingData_.set_colTileNum(colTileNum_);
    tilingData_.set_rowNormalNum(rowNormalNum_);
    tilingData_.set_colNormalNum(colNormalNum_);
    tilingData_.set_rowTailNum(rowTailNum_);
    tilingData_.set_colTailNum(colTailNum_);
    tilingData_.set_mode(mode_);
    tilingData_.set_indicesFactor(indicesFactor_);
    tilingData_.set_offsetsFactor(offsetsFactor_);
    tilingData_.set_weightDimFactor(weightDimFactor_);
    tilingData_.set_weightRowFactor(weightRowFactor_);
    tilingData_.set_isNeedSampleWeight(isNeedSampleWeight_);
    tilingData_.set_indicesNumel(indicesNumel_);
    tilingData_.set_indicesLimit(indicesLimit_);
    tilingData_.set_sampleWeightNum(sampleWeightNum_);

    TilingDataPrint();
}

void EmbeddingBagRegBaseTiling::TilingDataPrint() const
{
    OP_LOGD(context_->GetNodeName(), "start print             ");
    OP_LOGD(context_->GetNodeName(), "usedCoreNum_:              %ld", usedCoreNum_);
    OP_LOGD(context_->GetNodeName(), "embeddingDim_:             %ld", embeddingDim_);
    OP_LOGD(context_->GetNodeName(), "indiceSize_:               %ld", indiceSize_);
    OP_LOGD(context_->GetNodeName(), "rowNormalNum_:             %ld", rowNormalNum_);
    OP_LOGD(context_->GetNodeName(), "colNormalNum_:             %ld", colNormalNum_);
    OP_LOGD(context_->GetNodeName(), "rowTailNum_:               %ld", rowTailNum_);
    OP_LOGD(context_->GetNodeName(), "colTailNum_:               %ld", colTailNum_);
    OP_LOGD(context_->GetNodeName(), "indicesFactor_:            %ld", indicesFactor_);
    OP_LOGD(context_->GetNodeName(), "offsetsFactor_:            %ld", offsetsFactor_);
    OP_LOGD(context_->GetNodeName(), "weightDimFactor_:          %ld", weightDimFactor_);
    OP_LOGD(context_->GetNodeName(), "isNeedSampleWeight_:       %ld", isNeedSampleWeight_);
    OP_LOGD(context_->GetNodeName(), "tilingKey_:                %ld", tilingKey_);
}

ge::graphStatus EmbeddingBagRegBaseTiling::DoOpTiling()
{
    OP_LOGD(opName, "EmbeddingBag DoOpTiling Enter.");
    if (numBags_ <= 0){
        usedCoreNum_ = 1;
        return ge::GRAPH_SUCCESS;
    }
    
    AutoTiling();
    if (isNeedOffset_ == 1) {
        Compute1DFactor();
    } else {
        Compute2DFactor();
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingBagRegBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}


uint64_t EmbeddingBagRegBaseTiling::GetTilingKey() const
{
    uint64_t tilingKey = 0;
    if (isSimt_ == 1) {
        tilingKey = isNeedOffset_ ? TILING_KEY_SIMT_1D : TILING_KEY_SIMT_2D;

        if (weightShapeSize_ <= UINT32_MAX) {
            tilingKey += (offsetsType_ == indicesType_) ? 0 : OFFSET_DIFF_TYPE_SMALL;
        } else {
            tilingKey += (offsetsType_ == indicesType_) ? 1 : OFFSET_DIFF_TYPE_LARGE;
        }
        return tilingKey;
    }

    tilingKey = isNeedOffset_ ? TILING_KEY_INDICES_1D : TILING_KEY_INDICES_2D;
    if (offsetsType_ != indicesType_) {
        tilingKey += 1;
    }

    return tilingKey;
}

ge::graphStatus EmbeddingBagRegBaseTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingBagRegBaseTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingKey_ = GetTilingKey();
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF(
        (res != ge::GRAPH_SUCCESS), OP_LOGE(opName, "SetLocalMemorySize ubSize = %lu failed.", ubSize_),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingBagTilingForRegBase(gert::TilingContext* context)
{
    EmbeddingBagRegBaseTiling tiling(context);
    return tiling.DoTiling();
}

} // namespace optiling
