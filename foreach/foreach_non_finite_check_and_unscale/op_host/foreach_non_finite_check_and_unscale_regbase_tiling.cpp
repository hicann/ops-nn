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
 * \file foreach_non_finite_check_and_unscale_regbase_tiling.cpp
 * \brief
 */
#include "foreach_non_finite_check_and_unscale_tiling.h"

namespace optiling {
constexpr int32_t SCALE_GRADS_INDEX = 0;
constexpr int32_t INV_SCALE_INDEX_OFFSET = 1;

constexpr uint32_t NON_DYN_CNT = 2;
constexpr uint32_t BYTE_REPEAT = 256; // The amount of data that can be processed by a repeat.
constexpr size_t WORKSPACE_SIZE = 32;
constexpr uint8_t DTYPE_SIZE_FLOAT = 4;

constexpr uint64_t TILING_KEY_REGBASE = 100;
constexpr uint32_t COEFFICIENT_OF_FLOAT = 2;
constexpr uint32_t TEMP_BUFFER_NUM_FOR_MIN_MAX = 2;
constexpr uint32_t TEM_BUFFER_REMAIN = 32;
constexpr int64_t DOUBLE_BUFFER = 2;

class ForeachNonFiniteCheckAndUnscaleRegbaseTiling : public ForeachNonFiniteCheckAndUnscaleBaseClass
{
public:
    explicit ForeachNonFiniteCheckAndUnscaleRegbaseTiling(gert::TilingContext* context)
        : ForeachNonFiniteCheckAndUnscaleBaseClass(context) {};
    ~ForeachNonFiniteCheckAndUnscaleRegbaseTiling() override = default;
    void Reset(gert::TilingContext* context) override
    {
        ForeachNonFiniteCheckAndUnscaleBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        if (socVersion != platform_ascendc::SocVersion::ASCEND910_95) {
            return false;
        }
        return true;
    }

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    uint64_t GetTilingKey() const override;

private:
    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();
    void InitTilingDataItems();
    ge::graphStatus CheckParams() const;
    uint64_t GetTilingKeyVal();
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    void AssignDataToEachCore(int64_t needCoreNum);
    bool DivideUbMemory(uint64_t ubSizePlatForm);
    uint32_t GetReduceRetValSize(uint32_t srcDataSize);
    void FillTilingData();

private:
    std::string nodeName_ = "ForeachNonFiniteCheckAndUnscale";
    ForeachNonFiniteCheckAndUnscaleRegbaseTilingData tilingData_;

    uint32_t scaledGradsUbSize_ = 0;
    uint32_t reduceTempValUbSize_ = 0;
    int64_t tensorDataCountAlignedList_[MAX_TENSOR_COUNT] = {0};
    int64_t* tensorDataCountList_ = nullptr;
    int64_t* tensorStartOffsetList_ = nullptr;
    int64_t* tensorEndOffsetList_ = nullptr;
    uint16_t* tensorStartList_ = nullptr;
    uint16_t* tensorEndList_ = nullptr;
    int64_t totalDataCount_ = 0;
    int64_t totalDataCountAligned_ = 0;
    ge::DataType dataType_ = ge::DT_UNDEFINED;
    int32_t dataTypeSize_ = 0;
    int32_t elementsPerBlock_ = 0;
    int16_t totalTensorCount_ = 0;
    int32_t blockSize_ = 0;
};

ge::graphStatus ForeachNonFiniteCheckAndUnscaleRegbaseTiling::DoOpTiling()
{
    if (RunBigKernelTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Run big kernel tiling return failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ForeachNonFiniteCheckAndUnscaleRegbaseTiling::GetShapeAttrsInfo()
{
    InitTilingDataItems();
    blockSize_ = Ops::Base::GetUbBlockSize(context_);
    int16_t allTensorCount = int16_t(context_->GetComputeNodeInputNum());
    totalTensorCount_ = allTensorCount - NON_DYN_CNT;
    OP_CHECK_IF(
        totalTensorCount_ > MAX_TENSOR_COUNT || totalTensorCount_ <= 0,
        OP_LOGE(nodeName_, "The number of input tensors [%hd] not in (0, %hu].", totalTensorCount_, MAX_TENSOR_COUNT),
        return ge::GRAPH_FAILED);

    // Get shape, dtype information, and the total number of data.
    totalDataCount_ = 0;
    totalDataCountAligned_ = 0;
    for (int16_t i = 0; i < totalTensorCount_; i++) {
        auto descPtr = context_->GetDynamicInputDesc(SCALE_GRADS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, descPtr);
        auto tempDtype = descPtr->GetDataType();
        // Determine whether all data types are consistent.
        if (i == 0) {
            dataType_ = tempDtype;
            dataTypeSize_ = ge::GetSizeByDataType(dataType_);
            OP_CHECK_IF(
                dataTypeSize_ <= 0, OP_LOGE(nodeName_, "dataTypeSize[%d] error.", dataTypeSize_),
                return ge::GRAPH_FAILED);
            elementsPerBlock_ = blockSize_ / dataTypeSize_;
        } else if (tempDtype != dataType_) {
            OP_LOGE(nodeName_, "All tensor dtype must be consistent.");
            return ge::GRAPH_FAILED;
        }
        auto shapePtr = context_->GetDynamicInputShape(SCALE_GRADS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
        tensorDataCountList_[i] = shapePtr->GetStorageShape().GetShapeSize();
        totalDataCount_ += tensorDataCountList_[i];
        // Make a 32-byte alignment for each Tensor
        tensorDataCountAlignedList_[i] =
            Ops::Base::CeilDiv(tensorDataCountList_[i], static_cast<int64_t>(elementsPerBlock_)) * elementsPerBlock_;
        totalDataCountAligned_ += tensorDataCountAlignedList_[i];
    }

    for (int16_t j = totalTensorCount_; j < allTensorCount; j++) {
        auto inputShape = context_->GetInputShape(j);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
        auto inputLength = inputShape->GetStorageShape().GetShapeSize();
        if (inputLength != 1 && j == totalTensorCount_) {
            OP_LOGE(context_->GetNodeName(), "element num of input found_inf is %ld, which should be 1.", inputLength);
            return ge::GRAPH_FAILED;
        }

        if (inputLength != 1 && j == allTensorCount - 1) {
            OP_LOGE(context_->GetNodeName(), "element num of input inv_scale is %ld, which should be 1.", inputLength);
            return ge::GRAPH_FAILED;
        }
    }
    return CheckParams();
}

uint64_t ForeachNonFiniteCheckAndUnscaleRegbaseTiling::GetTilingKey() const
{
    return context_->GetTilingKey();
}

ge::graphStatus ForeachNonFiniteCheckAndUnscaleRegbaseTiling::RunBigKernelTiling()
{
    OP_LOGD(nodeName_, "Start.");
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    uint64_t ubSizePlatForm = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, ubSizePlatForm);
    context_->SetTilingKey(GetTilingKeyVal());

    uint32_t needCoreNum = std::min<uint32_t>(GetNeedCoreNum(platformInfo->GetCoreNum()), MAX_CORE_COUNT_REGBASE);
    OP_LOGD(nodeName_, "ubSizePlatForm:%lu, needCoreNum:%u", ubSizePlatForm, needCoreNum);
    OP_CHECK_IF(
        ubSizePlatForm == 0, OP_LOGE(nodeName_, "Param needCoreNum or ubSizePlatForm is zero."),
        return ge::GRAPH_FAILED);

    if (totalDataCount_ == 0) {
        needCoreNum = 1;
        tensorStartList_[0] = 0;
        tensorEndList_[0] = 0;
        tensorStartOffsetList_[0] = 0;
        tensorEndOffsetList_[0] = 0;
    } else {
        AssignDataToEachCore(needCoreNum);
    }

    OP_CHECK_IF(
        DivideUbMemory(ubSizePlatForm) == false, OP_LOGE(nodeName_, "DivideUbMemory failed."), return ge::GRAPH_FAILED);
    FillTilingData();
    context_->SetBlockDim(needCoreNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = WORKSPACE_SIZE;
    OP_LOGD(nodeName_, "Success.");
    return ge::GRAPH_SUCCESS;
}

void ForeachNonFiniteCheckAndUnscaleRegbaseTiling::InitTilingDataItems()
{
    tensorDataCountList_ = tilingData_.get_tensorDataCountList();
    tensorStartList_ = tilingData_.get_tensorStartList();
    tensorEndList_ = tilingData_.get_tensorEndList();
    tensorStartOffsetList_ = tilingData_.get_tensorStartOffsetList();
    tensorEndOffsetList_ = tilingData_.get_tensorEndOffsetList();
}

ge::graphStatus ForeachNonFiniteCheckAndUnscaleRegbaseTiling::CheckParams() const
{
    OP_LOGD(
        nodeName_, "dataType:%d, totalTensorCount:%hd, totalDataCountAligned:%ld.", static_cast<int32_t>(dataType_),
        totalTensorCount_, totalDataCountAligned_);
    OP_CHECK_IF(
        dataType_ != ge::DT_FLOAT16 && dataType_ != ge::DT_BF16 && dataType_ != ge::DT_FLOAT,
        OP_LOGE(nodeName_, "The input dtype not in [float16, bfloat16, float]."), return ge::GRAPH_FAILED);
    auto flagDescPtr = context_->GetInputDesc(totalTensorCount_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, flagDescPtr);
    auto scaleDescPtr = context_->GetInputDesc(totalTensorCount_ + INV_SCALE_INDEX_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleDescPtr);
    OP_CHECK_IF(
        flagDescPtr->GetDataType() != ge::DT_FLOAT || scaleDescPtr->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(nodeName_, "The input found_inf and inv_scale dtype must be float."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

uint64_t ForeachNonFiniteCheckAndUnscaleRegbaseTiling::GetTilingKeyVal()
{
    return TILING_KEY_REGBASE;
}

uint32_t ForeachNonFiniteCheckAndUnscaleRegbaseTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    uint32_t tempCoreNum(totalDataCountAligned_ / elementsPerBlock_);
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void ForeachNonFiniteCheckAndUnscaleRegbaseTiling::AssignDataToEachCore(int64_t needCoreNum)
{
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }

    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = totalDataCountAligned_ / elementsPerBlock_;
    // Divisible, representing the amount of data each core needs to process.
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock_;
    int64_t remainderCount = blockCount % needCoreNum; // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList_[coreIndex] = 0;
    tensorStartOffsetList_[coreIndex] = 0;
    for (int16_t i = 0; i < totalTensorCount_; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock_;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountAlignedList_[i] - cursorPos;
        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList_[coreIndex] = i;
        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList_[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountAlignedList_[i]) {
            tensorStartList_[coreIndex] = i;
            tensorStartOffsetList_[coreIndex] = cursorPos;
            --i; // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList_[coreIndex] = i + 1;
            tensorStartOffsetList_[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount) {
        tensorEndList_[coreIndex] = totalTensorCount_ - 1;
        tensorEndOffsetList_[coreIndex] = tensorDataCountAlignedList_[totalTensorCount_ - 1] - 1;
    }
}

bool ForeachNonFiniteCheckAndUnscaleRegbaseTiling::DivideUbMemory(uint64_t ubSizePlatForm)
{
    // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
    uint32_t canUseUbSize =
        uint32_t(ubSizePlatForm - tilingData_.GetDataSize() - TEM_BUFFER_REMAIN * TEMP_BUFFER_NUM_FOR_MIN_MAX) /
        DOUBLE_BUFFER;
    canUseUbSize = canUseUbSize / blockSize_ * blockSize_;
    uint32_t coefficient = COEFFICIENT_OF_FLOAT;
    uint32_t predictSGUbSize = uint32_t(
        BYTE_REPEAT * dataTypeSize_ / (coefficient * BYTE_REPEAT * dataTypeSize_ + blockSize_ * 1.0) * canUseUbSize);
    scaledGradsUbSize_ = predictSGUbSize / blockSize_ * blockSize_;
    reduceTempValUbSize_ = GetReduceRetValSize(scaledGradsUbSize_);
    if ((coefficient * scaledGradsUbSize_ + reduceTempValUbSize_) > canUseUbSize) {
        return false;
    } else {
        return true;
    }
}

uint32_t ForeachNonFiniteCheckAndUnscaleRegbaseTiling::GetReduceRetValSize(uint32_t srcDataSize)
{
    /* Calculate the space size of the intermediate variable workLocal and
        the result variable dstLocal of ReduceMax and ReduceMin. */
    uint32_t srcDataCount = srcDataSize / dataTypeSize_;
    uint8_t perRepeatCount = BYTE_REPEAT / DTYPE_SIZE_FLOAT;
    uint8_t perBlockCount = blockSize_ / DTYPE_SIZE_FLOAT;
    uint32_t iter1OutputCount = uint32_t(std::ceil(2.0 * srcDataCount / perRepeatCount));
    uint32_t iter1AlignEnd =
        Ops::Base::CeilDiv(static_cast<int64_t>(iter1OutputCount), static_cast<int64_t>(perBlockCount)) * perBlockCount;
    return iter1AlignEnd * DTYPE_SIZE_FLOAT;
}

void ForeachNonFiniteCheckAndUnscaleRegbaseTiling::FillTilingData()
{
    std::ostringstream tempOSS;
    tempOSS << "scaledGradsUbSize: " << scaledGradsUbSize_ << ", reduceTempValUbSize: " << reduceTempValUbSize_ << "."
            << std::endl
            << "tensorDataCountList: ";
    for (int16_t i = 0; i < totalTensorCount_; i++) {
        tempOSS << "[" << i << "]: " << tensorDataCountList_[i] << ", ";
    }
    tempOSS << std::endl << "Per core info: " << std::endl;
    for (uint16_t i = 0; i < MAX_CORE_COUNT; i++) {
        tempOSS << "index:[" << i << "], tensorStartList: " << tensorStartList_[i]
                << ", tensorEndList:" << tensorEndList_[i] << ", tensorStartOffsetList: " << tensorStartOffsetList_[i]
                << ", tensorEndOffsetList: " << tensorEndOffsetList_[i] << "." << std::endl;
    }

    OP_LOGD(nodeName_, "Tiling info, %s", tempOSS.str().c_str());

    tilingData_.set_scaledGradsUbSize(scaledGradsUbSize_);
    tilingData_.set_reduceTempValUbSize(reduceTempValUbSize_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

REGISTER_OPS_TILING_TEMPLATE(ForeachNonFiniteCheckAndUnscale, ForeachNonFiniteCheckAndUnscaleRegbaseTiling, 30000);
} // namespace optiling
