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
 * \file foreach_reduce_tiling_func.cpp
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_FUNC_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_FUNC_H_

#include <cmath>
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "log/error_code.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info_def.h"
#include "common_dtype.h"
#include "foreach_reduce_tiling_def.h"

namespace optiling {
constexpr uint32_t DEFAULT_SYNCALL_NEED_SIZE = 8;

constexpr uint64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

constexpr int32_t MAX_SUPPORT_DIMS_NUMS = 8;

class ForeachReduceTiling
{
public:
    explicit ForeachReduceTiling(gert::TilingContext* context) : tilingContext(context) {};
    /**
     ** function: Init
     */
    ge::graphStatus Init()
    {
        // Get shape, dtype information, and the total number of data.
        for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
            auto srcTensor = tilingContext->GetDynamicInputTensor(0, i);
            if (srcTensor == nullptr) {
                break;
            }
            auto srcDtype = srcTensor->GetDataType();
            // Determine whether all data types are consistent.
            if (dataType == ge::DT_UNDEFINED) {
                dataType = srcDtype;
                dataTypeSize = GetDataTypeSize(dataType);
                if (dataTypeSize == 0) {
                    dataTypeSize = BYTE_LEN_4;
                }
                elementsPerBlock = BYTE_BLOCK / dataTypeSize;
            } else if (srcDtype != dataType) {
                return ge::GRAPH_FAILED;
            }
            gert::Shape tempShape = srcTensor->GetStorageShape();
            // Make a 32-byte alignment for each Tensor
            tensorDataCountList[i] = (uint64_t)tempShape.GetShapeSize();
            if (tensorDataCountList[i] == 0) {
                isExistEmptyTensor = true;
            }
            totalBlockCount += CeilA2B(tensorDataCountList[i], elementsPerBlock);
            totalTensorCount++;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RunBigKernelTiling()
    {
        if (CheckShapeInfo() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

        uint64_t ubSizePlatForm = 0;

        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

        tilingContext->SetTilingKey(GetTilingKeyVal());

        needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

        AssignDataToEachCore();
        DivideUbMemory(ubSizePlatForm);

        // Reduce Op Addition
        AssignTensorMiddleCountList();

        FillTilingData();

        tilingContext->SetBlockDim(needCoreNum);
        tilingContext->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动

        size_t usrSize = (MAX_CORE_CONT + MAX_TENSOR_CONT) * sizeof(float);
        size_t sysWorkspaceSize = WORK_SPACE_SIZE;
        size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
        if (currentWorkspace == nullptr) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = usrSize + sysWorkspaceSize;

        return ge::GRAPH_SUCCESS;
    }

private:
    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    uint64_t GetTilingKeyVal()
    {
        switch (dataType) {
            case ge::DT_FLOAT:
                return TILING_KEY_FLOAT;
            case ge::DT_FLOAT16:
                return TILING_KEY_HALF;
            case ge::DT_BF16:
                return TILING_KEY_BF16;
            default:
                return 0;
        }
    }

    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform)
    {
        uint32_t tempCoreNum = static_cast<uint32_t>(totalBlockCount);
        if (tempCoreNum == 0) {
            tempCoreNum = 1;
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    void AssignDataToEachCore()
    {
        // Kernel the input data according to 32 byte alignment.
        // Divisible, representing the amount of data each core needs to process.
        uint64_t tempPerCoreCount = totalBlockCount / needCoreNum * elementsPerBlock;
        uint64_t remainderCount = totalBlockCount % needCoreNum; // remainder.
        uint16_t coreIndex = 0;
        uint64_t dataCount = 0;
        uint64_t curCmpCount = 0;
        uint64_t cursorPos = 0;
        tensorStartList[coreIndex] = 0;
        tensorStartOffsetList[coreIndex] = 0;
        for (uint32_t i = 0; i < totalTensorCount; i++) {
            // When the remainder is not 0, each kernel index with less than the remainder processes one more block of
            // data.
            if (remainderCount && coreIndex < remainderCount) {
                curCmpCount = tempPerCoreCount + elementsPerBlock;
            } else {
                curCmpCount = tempPerCoreCount;
            }
            uint64_t tempRealCount = tensorDataCountList[i] - cursorPos;
            uint64_t tempCount = CeilA2B(tempRealCount, elementsPerBlock) * elementsPerBlock;
            if (dataCount + tempCount < curCmpCount) {
                dataCount += tempCount;
                cursorPos = 0;
                continue;
            }
            // dataCount >= curCmpCount, Calculate the offset
            tensorEndList[coreIndex] = i;
            cursorPos = cursorPos + curCmpCount - dataCount;
            // ReduceOp need more currect value
            tensorEndOffsetList[coreIndex] =
                dataCount + tempRealCount < curCmpCount ? tensorDataCountList[i] - 1 : cursorPos - 1;
            dataCount = 0;
            coreIndex++;
            if (cursorPos < tensorDataCountList[i]) {
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
            tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
        }
    }

    void DivideUbMemory(uint64_t ubSizePlatForm)
    {
        uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 16384);
        if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
            totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
        }
        uint32_t canUseUbSize = totalSize / 2;
        inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                         canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
    }

    void AssignTensorMiddleCountList()
    {
        uint16_t preCoreTensorIndex = 0;
        for (uint32_t i = 1; i < needCoreNum; i++) {
            if (preCoreTensorIndex == tensorStartList[i]) {
                tensorMiddleCountList[preCoreTensorIndex] += 1;
            } else {
                if (tensorStartOffsetList[i] > 0) {
                    tensorMiddleCountList[tensorStartList[i]] += 1;
                }
            }
            preCoreTensorIndex = tensorStartList[i];
        }
        uint16_t tensorMiddleStart = 0;
        for (uint32_t j = 0; j < totalTensorCount; j++) {
            tensorMiddleCountList[j]++;
            tensorMiddleStartList[j] = tensorMiddleStart;
            tensorMiddleStart += tensorMiddleCountList[j];
        }
        uint16_t coreMiddleOffset = 0;
        for (uint32_t j = 0; j < needCoreNum; j++) {
            coreMiddleOffsetList[j] = coreMiddleOffset;
            coreMiddleOffset += tensorEndList[j] - tensorStartList[j] + 1;
        }
    }

    /**
     ** function: Check input and out shape info
     */
    ge::graphStatus CheckShapeInfo()
    {
        auto computeNodeInfo = tilingContext->GetComputeNodeInfo();
        OP_CHECK_IF(
            computeNodeInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetComputeNodeInfo failed."),
            return ge::GRAPH_FAILED);
        auto inputTensorsNum = computeNodeInfo->GetIrInputsNum() - 1;
        auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(0);
        OP_CHECK_IF(
            anchorInstanceInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetInputInstanceInfo failed."),
            return ge::GRAPH_FAILED);

        // check tensorlist size
        size_t xSize = anchorInstanceInfo->GetInstanceNum();
        OP_CHECK_IF(
            xSize > MAX_TENSOR_CONT,
            OP_LOGE(
                tilingContext->GetNodeName(), "The number of input tensors [%lu] not in [0, %d].", xSize,
                MAX_TENSOR_CONT),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(
            CheckInputTensorlistShape(static_cast<size_t>(inputTensorsNum)) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "CheckInputTensorlistShape failed."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckOutputShapeAndDtype() != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "CheckOutputShapeAndDtype failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: Check input tensorlist shape  invalid
     */
    ge::graphStatus CheckInputTensorlistShape(size_t inputTensorsNum)
    {
        uint64_t startListIndex = 0;

        for (size_t tensorId = 0; tensorId < totalTensorCount; tensorId++) {
            auto x1Shape = tilingContext->GetDynamicInputShape(startListIndex, tensorId);
            OP_CHECK_IF(
                x1Shape == nullptr,
                OP_LOGE(tilingContext->GetNodeName(), "The input %lu shape is null.", startListIndex),
                return ge::GRAPH_FAILED);

            // check max dim
            OP_CHECK_IF(
                x1Shape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIMS_NUMS,
                OP_LOGE(
                    tilingContext->GetNodeName(),
                    "The input %lu shape is invalid, and it cannot be larger than %zu dimensions.", startListIndex,
                    static_cast<size_t>(MAX_SUPPORT_DIMS_NUMS)),
                return ge::GRAPH_FAILED);

            // checke tensorlist input consistent
            for (size_t listId = startListIndex + 1; listId < static_cast<size_t>(inputTensorsNum); listId++) {
                OP_CHECK_IF(
                    x1Shape->GetStorageShape() !=
                        tilingContext->GetDynamicInputShape(listId, tensorId)->GetStorageShape(),
                    OP_LOGE(
                        tilingContext->GetNodeName(), "The input %lu shape should be same with input %lu.", listId,
                        startListIndex),
                    return ge::GRAPH_FAILED);
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: Check output tensor list shape and dtype invalid
     */
    ge::graphStatus CheckOutputShapeAndDtype()
    {
        size_t outputCount = tilingContext->GetComputeNodeOutputNum();
        OP_CHECK_IF(
            static_cast<size_t>(totalTensorCount) != outputCount,
            OP_LOGE(
                tilingContext->GetNodeName(), "The output num should be same with input, expect %u, actual %lu.",
                totalTensorCount, outputCount),
            return ge::GRAPH_FAILED);
        for (uint32_t i = 0; i < totalTensorCount; i++) {
            auto tempDesc = tilingContext->GetOutputDesc(i);
            OP_CHECK_IF(
                tempDesc == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The output %u desc is null.", i),
                return ge::GRAPH_FAILED);
            auto dstDtype = tempDesc->GetDataType();
            OP_CHECK_IF(
                dstDtype != dataType,
                OP_LOGE(tilingContext->GetNodeName(), "The tensor %u of output datatype should be same with input.", i),
                return ge::GRAPH_FAILED);
            auto dstShape = tilingContext->GetOutputShape(i);
            OP_CHECK_IF(
                dstShape == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The output %u shape is null.", i),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                dstShape->GetStorageShape().GetShapeSize() != 1,
                OP_LOGE(tilingContext->GetNodeName(), "The number of output tensor [%u] must be 1.", i),
                return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    void FillTilingData()
    {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_needCoreNum(needCoreNum);
        tilingData.set_totalTensorCount(totalTensorCount);
        tilingData.set_tensorDataCountList(tensorDataCountList);
        tilingData.set_tensorStartList(tensorStartList);
        tilingData.set_tensorEndList(tensorEndList);
        tilingData.set_tensorStartOffsetList(tensorStartOffsetList);
        tilingData.set_tensorEndOffsetList(tensorEndOffsetList);

        // Reduce Op Addition
        tilingData.set_tensorMiddleCountList(tensorMiddleCountList);
        tilingData.set_tensorMiddleStartList(tensorMiddleStartList);
        tilingData.set_coreMiddleOffsetList(coreMiddleOffsetList);

        tilingData.SaveToBuffer(
            tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

private:
    ForeachReduceTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    uint64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    uint64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    uint64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    uint64_t totalBlockCount = 0;
    uint8_t dataTypeSize = 0;
    uint8_t elementsPerBlock = 0;
    uint32_t totalTensorCount = 0;
    uint32_t needCoreNum = 0;

    bool isExistEmptyTensor = false;

    uint32_t modelCode = 0;

    uint16_t tensorMiddleCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorMiddleStartList[MAX_TENSOR_CONT] = {0};
    uint16_t coreMiddleOffsetList[MAX_CORE_CONT] = {0};
};

static ge::graphStatus Tiling4ForeachNormTiling(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4ForeachTiling(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<ForeachCompileInfo>();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    compileInfoPtr->aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicCoreNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    return ge::GRAPH_SUCCESS;
}

REDUCE_MEMBASE_TILING(ForeachNorm);
REGISTER_OPS_TILING_TEMPLATE(ForeachNorm, ForeachNormMembaseTiling, 10000);

IMPL_OP_OPTILING(ForeachNorm)
    .Tiling(Tiling4ForeachNormTiling)
    .TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_COMMON_REDUCE_FUNC_H_
