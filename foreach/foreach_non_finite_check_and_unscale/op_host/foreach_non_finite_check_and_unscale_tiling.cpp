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
 * \file foreach_non_finite_check_and_unscale_tiling.cpp
 * \brief
 */
#include "foreach_non_finite_check_and_unscale_tiling.h"

namespace optiling {

constexpr int32_t SCALE_GRADS_INDEX = 0;
constexpr int32_t INV_SCALE_INDEX_OFFSET = 1;

constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t NON_DYN_CNT = 2;
constexpr uint32_t BYTE_REPEAT = 256; // The amount of data that can be processed by a repeat.
constexpr size_t WORKSPACE_SIZE = 32;
constexpr uint8_t DTYPE_SIZE_FLOAT = 4;
constexpr uint8_t DTYPE_SIZE_HALF = 2;

constexpr uint64_t TILING_KEY_FLOAT = 1;
constexpr uint64_t TILING_KEY_HALF = 2;
constexpr uint64_t TILING_KEY_BFLOAT16 = 3;

constexpr uint32_t COEFFICIENT_OF_FLOAT = 2;
constexpr uint32_t COEFFICIENT_OF_NON_FLOAT = COEFFICIENT_OF_FLOAT * 3;

class ForeachNonFiniteCheckAndUnscaleTiling
{
public:
    explicit ForeachNonFiniteCheckAndUnscaleTiling(gert::TilingContext* context)
        : tilingContext(context), nodeName(context->GetNodeName()) {};

    ge::graphStatus Init();
    ge::graphStatus RunBigKernelTiling();

private:
    template <typename T1, typename T2>
    inline T1 CeilDiv(T1 a, T2 b) const
    {
        T1 bTemp(b);
        return bTemp == 0 ? a : (a + bTemp - 1) / bTemp;
    };

    void InitTilingDataItems();
    ge::graphStatus CheckParams() const;
    uint64_t GetTilingKeyVal();
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    void AssignDataToEachCore(int64_t needCoreNum);
    bool DivideUbMemory(uint64_t ubSizePlatForm);
    uint32_t GetReduceRetValSize(uint32_t srcDataSize);
    void FillTilingData();

private:
    gert::TilingContext* tilingContext = nullptr;
    std::string nodeName = "ForeachNonFiniteCheckAndUnscale";
    ForeachNonFiniteCheckAndUnscaleTilingData tilingData;

    uint32_t scaledGradsUbSize = 0;
    uint32_t reduceTempValUbSize = 0;
    int64_t tensorDataCountAlignedList[MAX_TENSOR_COUNT] = {0};
    int64_t* tensorDataCountList = nullptr;
    int64_t* tensorStartOffsetList = nullptr;
    int64_t* tensorEndOffsetList = nullptr;
    uint16_t* tensorStartList = nullptr;
    uint16_t* tensorEndList = nullptr;
    int64_t totalDataCountAligned = 0;
    ge::DataType dataType = ge::DT_UNDEFINED;
    int32_t dataTypeSize = 0;
    int32_t elementsPerBlock = 0;
    int16_t totalTensorCount = 0;
};

ge::graphStatus ForeachNonFiniteCheckAndUnscaleTiling::Init()
{
    InitTilingDataItems();
    totalTensorCount = int16_t(tilingContext->GetComputeNodeInputNum() - NON_DYN_CNT);
    OP_CHECK_IF(
        totalTensorCount > MAX_TENSOR_COUNT || totalTensorCount <= 0,
        OP_LOGE(nodeName, "The number of input tensors [%hd] not in (0, %hu].", totalTensorCount, MAX_TENSOR_COUNT),
        return ge::GRAPH_FAILED);
    // Get shape, dtype information, and the total number of data.
    for (int16_t i = 0; i < totalTensorCount; i++) {
        auto descPtr = tilingContext->GetDynamicInputDesc(SCALE_GRADS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, descPtr);
        auto tempDtype = descPtr->GetDataType();
        // Determine whether all data types are consistent.
        if (i == 0) {
            dataType = tempDtype;
            dataTypeSize = ge::GetSizeByDataType(dataType);
            OP_CHECK_IF(
                dataTypeSize <= 0, OP_LOGE(nodeName, "dataTypeSize[%d] error.", dataTypeSize), return ge::GRAPH_FAILED);
            elementsPerBlock = BYTE_BLOCK / dataTypeSize;
        } else if (tempDtype != dataType) {
            OP_LOGE(nodeName, "All tensor dtype must be consistent.");
            return ge::GRAPH_FAILED;
        }
        auto shapePtr = tilingContext->GetDynamicInputShape(SCALE_GRADS_INDEX, i);
        OP_CHECK_NULL_WITH_CONTEXT(tilingContext, shapePtr);
        tensorDataCountList[i] = shapePtr->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(
            tensorDataCountList[i] == 0, OP_LOGE(nodeName, "Input shape not support empty tensor."),
            return ge::GRAPH_FAILED);
        // Make a 32-byte alignment for each Tensor
        tensorDataCountAlignedList[i] = CeilDiv(tensorDataCountList[i], elementsPerBlock) * elementsPerBlock;
        totalDataCountAligned += tensorDataCountAlignedList[i];
    }
    return CheckParams();
}

ge::graphStatus ForeachNonFiniteCheckAndUnscaleTiling::RunBigKernelTiling()
{
    OP_LOGD(nodeName, "Start.");
    auto platformInfo = tilingContext->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, platformInfo);

    uint64_t ubSizePlatForm = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, ubSizePlatForm);
    tilingContext->SetTilingKey(GetTilingKeyVal());

    uint32_t needCoreNum = GetNeedCoreNum(platformInfo->GetCoreNum());
    OP_LOGD(nodeName, "ubSizePlatForm:%lu, needCoreNum:%u", ubSizePlatForm, needCoreNum);
    OP_CHECK_IF(
        needCoreNum == 0 || ubSizePlatForm == 0, OP_LOGE(nodeName, "Param needCoreNum or ubSizePlatForm is zero."),
        return ge::GRAPH_FAILED);
    AssignDataToEachCore(needCoreNum);
    OP_CHECK_IF(
        DivideUbMemory(ubSizePlatForm) == false, OP_LOGE(nodeName, "DivideUbMemory failed."), return ge::GRAPH_FAILED);
    FillTilingData();
    tilingContext->SetBlockDim(needCoreNum);
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;
    OP_LOGD(nodeName, "Success.");
    return ge::GRAPH_SUCCESS;
}

void ForeachNonFiniteCheckAndUnscaleTiling::InitTilingDataItems()
{
    tensorDataCountList = tilingData.get_tensorDataCountList();
    tensorStartList = tilingData.get_tensorStartList();
    tensorEndList = tilingData.get_tensorEndList();
    tensorStartOffsetList = tilingData.get_tensorStartOffsetList();
    tensorEndOffsetList = tilingData.get_tensorEndOffsetList();
}

ge::graphStatus ForeachNonFiniteCheckAndUnscaleTiling::CheckParams() const
{
    OP_LOGD(
        nodeName, "dataType:%d, totalTensorCount:%hd, totalDataCountAligned:%ld.", static_cast<int32_t>(dataType),
        totalTensorCount, totalDataCountAligned);
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16 && dataType != ge::DT_FLOAT,
        OP_LOGE(nodeName, "The input dtype not in [float16, bfloat16, float]."), return ge::GRAPH_FAILED);
    auto flagDescPtr = tilingContext->GetInputDesc(totalTensorCount);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, flagDescPtr);
    auto scaleDescPtr = tilingContext->GetInputDesc(totalTensorCount + INV_SCALE_INDEX_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, scaleDescPtr);
    OP_CHECK_IF(
        flagDescPtr->GetDataType() != ge::DT_FLOAT || scaleDescPtr->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(nodeName, "The input found_inf and inv_scale dtype must be float."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

uint64_t ForeachNonFiniteCheckAndUnscaleTiling::GetTilingKeyVal()
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return TILING_KEY_FLOAT;
        case ge::DT_FLOAT16:
            return TILING_KEY_HALF;
        case ge::DT_BF16:
            return TILING_KEY_BFLOAT16;
        default:
            return 0;
    }
}

uint32_t ForeachNonFiniteCheckAndUnscaleTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    uint32_t tempCoreNum(totalDataCountAligned / elementsPerBlock);
    if (tempCoreNum < coreNumPlatform) {
        return tempCoreNum;
    } else {
        return coreNumPlatform;
    }
}

void ForeachNonFiniteCheckAndUnscaleTiling::AssignDataToEachCore(int64_t needCoreNum)
{
    // Kernel the input data according to 32 byte alignment.
    int64_t blockCount = totalDataCountAligned / elementsPerBlock;
    // Divisible, representing the amount of data each core needs to process.
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t tempPerCoreCount = blockCount / needCoreNum * elementsPerBlock;
    int64_t remainderCount = blockCount % needCoreNum; // remainder.
    uint16_t coreIndex = 0;
    int64_t dataCount = 0;
    int64_t curCmpCount = 0;
    int64_t cursorPos = 0;
    tensorStartList[coreIndex] = 0;
    tensorStartOffsetList[coreIndex] = 0;
    for (int16_t i = 0; i < totalTensorCount; i++) {
        // When the remainder is not 0, each kernel index with less than the remainder processes one more block of data.
        if (remainderCount && coreIndex < remainderCount) {
            curCmpCount = tempPerCoreCount + elementsPerBlock;
        } else {
            curCmpCount = tempPerCoreCount;
        }
        int64_t tempCount = tensorDataCountAlignedList[i] - cursorPos;
        if (dataCount + tempCount < curCmpCount) {
            dataCount += tempCount;
            cursorPos = 0;
            continue;
        }
        // dataCount >= curCmpCount, Calculate the offset
        tensorEndList[coreIndex] = i;
        cursorPos = cursorPos + curCmpCount - dataCount;
        tensorEndOffsetList[coreIndex] = cursorPos - 1;
        dataCount = 0;
        coreIndex++;
        if (cursorPos < tensorDataCountAlignedList[i]) {
            tensorStartList[coreIndex] = i;
            tensorStartOffsetList[coreIndex] = cursorPos;
            --i; // The next loop continues to allocate the current tensor
        } else if (coreIndex != needCoreNum) {
            tensorStartList[coreIndex] = i + 1;
            tensorStartOffsetList[coreIndex] = 0;
            cursorPos = 0;
        }
    }
    /* The temporary count variable is not 0, which means that the last tensor is truncated,
        and you need to manually set the offset of the last core. */
    if (dataCount) {
        tensorEndList[coreIndex] = totalTensorCount - 1;
        tensorEndOffsetList[coreIndex] = tensorDataCountAlignedList[totalTensorCount - 1] - 1;
    }
}

bool ForeachNonFiniteCheckAndUnscaleTiling::DivideUbMemory(uint64_t ubSizePlatForm)
{
    // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
    uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / 2;
    canUseUbSize = canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    uint32_t coefficient = COEFFICIENT_OF_NON_FLOAT;
    if (dataType == ge::DT_FLOAT) {
        coefficient = COEFFICIENT_OF_FLOAT;
    }
    uint32_t predictSGUbSize = uint32_t(
        BYTE_REPEAT * dataTypeSize / (coefficient * BYTE_REPEAT * dataTypeSize + BYTE_BLOCK * 1.0) * canUseUbSize);
    scaledGradsUbSize = predictSGUbSize / BYTE_BLOCK * BYTE_BLOCK;
    reduceTempValUbSize = GetReduceRetValSize(scaledGradsUbSize);
    if ((coefficient * scaledGradsUbSize + reduceTempValUbSize) > canUseUbSize) {
        return false;
    } else {
        return true;
    }
}

uint32_t ForeachNonFiniteCheckAndUnscaleTiling::GetReduceRetValSize(uint32_t srcDataSize)
{
    /* Calculate the space size of the intermediate variable workLocal and
        the result variable dstLocal of ReduceMax and ReduceMin. */
    uint32_t srcDataCount = srcDataSize / dataTypeSize;
    uint8_t perRepeatCount = BYTE_REPEAT / DTYPE_SIZE_FLOAT;
    uint8_t perBlockCount = BYTE_BLOCK / DTYPE_SIZE_FLOAT;
    uint32_t iter1OutputCount = uint32_t(std::ceil(2.0 * srcDataCount / perRepeatCount));
    uint32_t iter1AlignEnd = CeilDiv(iter1OutputCount, perBlockCount) * perBlockCount;
    return iter1AlignEnd * DTYPE_SIZE_FLOAT;
}

void ForeachNonFiniteCheckAndUnscaleTiling::FillTilingData()
{
    std::ostringstream tempOSS;
    tempOSS << "scaledGradsUbSize: " << scaledGradsUbSize << ", reduceTempValUbSize: " << reduceTempValUbSize << "."
            << std::endl
            << "tensorDataCountList: ";
    for (int16_t i = 0; i < totalTensorCount; i++) {
        tempOSS << "[" << i << "]: " << tensorDataCountList[i] << ", ";
    }
    tempOSS << std::endl << "Per core info: " << std::endl;
    for (uint16_t i = 0; i < MAX_CORE_COUNT; i++) {
        tempOSS << "index:[" << i << "], tensorStartList: " << tensorStartList[i]
                << ", tensorEndList:" << tensorEndList[i] << ", tensorStartOffsetList: " << tensorStartOffsetList[i]
                << ", tensorEndOffsetList: " << tensorEndOffsetList[i] << "." << std::endl;
    }

    OP_LOGD(nodeName, "Tiling info, %s", tempOSS.str().c_str());

    tilingData.set_scaledGradsUbSize(scaledGradsUbSize);
    tilingData.set_reduceTempValUbSize(reduceTempValUbSize);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

class ForeachNonFiniteCheckAndUnscaleMembaseTiling : public ForeachNonFiniteCheckAndUnscaleBaseClass
{
public:
    explicit ForeachNonFiniteCheckAndUnscaleMembaseTiling(gert::TilingContext* context)
        : ForeachNonFiniteCheckAndUnscaleBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        ForeachNonFiniteCheckAndUnscaleBaseClass::Reset(context);
    }

protected:
    bool IsCapable() override
    {
        if (socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
            return false;
        }
        return true;
    }

    ge::graphStatus DoOpTiling() override
    {
        ForeachNonFiniteCheckAndUnscaleTiling tilingObject(context_);
        if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "Init tiling object return failed.");
            return ge::GRAPH_FAILED;
        }
        if (tilingObject.RunBigKernelTiling() != ge::GRAPH_SUCCESS) {
            OP_LOGE(context_->GetNodeName(), "Run big kernel tiling return failed.");
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    uint64_t GetTilingKey() const override
    {
        return context_->GetTilingKey();
    }

    ge::graphStatus GetShapeAttrsInfo() override
    {
        return ge::GRAPH_SUCCESS;
    }
};

REGISTER_OPS_TILING_TEMPLATE(ForeachNonFiniteCheckAndUnscale, ForeachNonFiniteCheckAndUnscaleMembaseTiling, 10000);
} // namespace optiling