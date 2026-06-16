/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * \file foreach_tiling_class.h
 * \brief
 */

#ifndef FOREACH_TILING_CLASS_H_
#define FOREACH_TILING_CLASS_H_

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "log/error_code.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_info_def.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "foreach_tiling_def.h"
#include "common_dtype.h"

namespace optiling {
constexpr uint64_t TILING_HALF_N_SCALAR = 14;
constexpr uint64_t TILING_FLOAT_N_SCALAR = 4;
constexpr uint64_t TILING_INT_N_SCALAR = 4;
constexpr uint64_t TILING_BF16_N_SCALAR = 14;
constexpr uint32_t TILING_FLOAT_ERF = 5;
constexpr uint32_t TILING_HALF_ERF = 12;

constexpr uint64_t WORK_SPACE_SIZE = 32; // foreach(vector) not need workspace

constexpr uint32_t TANH_HALF_CALC_PROC = 5;
constexpr uint32_t TANH_FLOAT_CALC_PROC = 6;
constexpr uint32_t FOREACH_TANH_DIVIDER = 2;
constexpr uint32_t SIN_HALF_CALC_FAC = 6;
constexpr uint32_t SIN_FLOAT_CALC_FAC = 2;
constexpr uint32_t SIN_BASIC_BLOCK = 2048;

constexpr uint32_t COSH_HALF_CALC_PROC = 6;
constexpr uint32_t COSH_FLOAT_CALC_PROC = 2;
constexpr uint32_t COSH_BASIC_BLOCK_SIZE = 1024;

constexpr uint32_t SINH_HALF_CALC_PROC = 4;
constexpr uint32_t SINH_FLOAT_CALC_PROC = 1;
constexpr uint32_t SINH_BASIC_BLOCK_SIZE = 1024;

constexpr uint32_t ATAN_HALF_CALC_PROC = 10;
constexpr uint32_t ATAN_FLOAT_CALC_PROC = 4;
constexpr uint32_t ATAN_BASIC_BLOCK_SIZE = 1024;

constexpr uint32_t TAN_HALF_CALC_PROC = 10;
constexpr uint32_t TAN_FLOAT_CALC_PROC = 4;
constexpr uint32_t TAN_BASIC_BLOCK_SIZE = 1024;

constexpr uint32_t SIGN_CALC_PROC = 3;
constexpr uint32_t SIGN_BASIC_BLOCK_SIZE = 1024;

constexpr uint32_t BINARY_LIST_UB_DIVIDER = 6;
constexpr uint32_t BINARY_SCALAR_UB_DIVIDER = 4;
constexpr uint32_t FOREACH_POINTWISE_DIVIDER = 8;
constexpr uint32_t FOREACH_POW_SCALAR_DIVIDER = 4;
constexpr uint32_t FOREACH_COS_DIVIDER = 4;
constexpr uint32_t FOREACH_POINTWISE_LIST_DIVIDER = 8;
constexpr uint32_t FOREACH_LERP_SCALAR_UB_DIVIDER = 6;
constexpr uint32_t FOREACH_LERP_LIST_UB_DIVIDER = 11;
constexpr uint32_t FOREACH_SIN_DIVIDER = 4;
constexpr uint32_t FOREACH_ERF_BUFFER_DIVIDER = 4;
constexpr uint32_t FOREACH_ERF_FLOAT_DIVIDER = 4;  // erf float 预留 3 倍的输入空间
constexpr uint32_t FOREACH_ERF_HALF_DIVIDER = 9;   // erf half 预留 8 倍的输入空间
constexpr uint32_t FOREACH_ERFC_FLOAT_DIVIDER = 8; // erfc float 预留 7 倍的输入空间
constexpr uint32_t FOREACH_ERFC_HALF_DIVIDER = 17; // erfc half 预留 16 倍的输入空间

constexpr uint8_t ZERO_OP_CODE = 1;
constexpr uint8_t SOLO_LOG_OP_CODE = 2;
constexpr uint8_t BINARY_LIST_OP_CODE = 3;
constexpr uint8_t FOREACH_POINTWISE_OP_CODE = 4;
constexpr uint8_t FOREACH_COS_OP_CODE = 5;
constexpr uint8_t SOLO_LOG2_OP_CODE = 6;
constexpr uint8_t SOLO_NEG_OP_CODE = 7;
constexpr uint8_t FOREACH_POW_TENSOR_OP_CODE = 8;
constexpr uint8_t FOREACH_BINARY_SCALAR_OP_CODE = 9;
constexpr uint8_t FOREACH_POINTWISE_LIST_OP_CODE = 10;
constexpr uint8_t FOREACH_SIGMOID_OP_CODE = 11;
constexpr uint8_t FOREACH_ERF_OP_CODE = 12;
constexpr uint8_t FOREACH_COSH_OP_CODE = 13;
constexpr uint8_t FOREACH_ASIN_OP_CODE = 13;
constexpr uint8_t FOREACH_ACOS_OP_CODE = 13;
constexpr uint8_t FOREACH_SINH_OP_CODE = 14;
constexpr uint8_t FOREACH_TAN_OP_CODE = 15;
constexpr uint8_t FOREACH_ERFC_OP_CODE = 16;
constexpr uint8_t FOREACH_TANH_OP_CODE = 17;
constexpr uint8_t FOREACH_ATAN_OP_CODE = 18;
constexpr uint8_t FOREACH_LERP_SCALAR_OP_CODE = 19;
constexpr uint8_t FOREACH_LERP_LIST_OP_CODE = 20;
constexpr uint8_t FOREACH_POW_SCALAR_OP_CODE = 21;
constexpr uint8_t FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE = 22;
constexpr uint8_t FOREACH_SIN_OP_CODE = 23;
constexpr uint8_t FOREACH_ABS_OP_CODE = 24;
constexpr uint8_t FOREACH_MUL_SCALAR_OP_CODE = 25;
constexpr uint8_t FOREACH_EXP_OP_CODE = 26;
constexpr uint8_t FOREACH_MAXIMUM_LIST_OP_CODE = 27;
constexpr uint8_t FOREACH_ADD_LIST_OP_CODE = 28;
constexpr uint8_t FOREACH_ROUND_OFF_NUM_OP_CODE = 29;
constexpr uint8_t FOREACH_SUB_SCALAR_OP_CODE = 30;
constexpr uint8_t FOREACH_DIV_SCALAR_OP_CODE = 31;
constexpr uint8_t FOREACH_COPY_OP_CODE = 32;
constexpr uint8_t FOREACH_SIGN_OP_CODE = 33;

constexpr uint16_t LOG2_BASIC_FOR_LOG2 = 1024;
constexpr uint32_t LOG2_HALF_FOR_LOG2 = 4;
constexpr uint32_t LOG2_FLOAT_FOR_LOG2 = 0;

constexpr uint8_t BYTE_PER_BLOCK = 32;
constexpr uint32_t BYTE_PER_REPEAT = 256;
constexpr int32_t POW_TENSOR_TENSOR_CALC_PROC[9] = {12, 3, 5, 3, 12, 12, 12, 12, 12};

constexpr uint8_t UB_DIVIDER_FOR_TEMP_CASTING = 10;

constexpr int32_t MAX_SUPPORT_DIMS_NUMS = 8;

constexpr uint32_t BUFFER_ATTENUATION = 2;

enum class ForeachInputType : uint8_t
{
    TYPE_TENSORS = 0,        // only tensor type
    TYPE_SCALARS_TENSOR = 1, // include scalarlist type
    TYPE_SCALAR = 2          // include scalar type
};

enum class ForeachReturnType : uint8_t
{
    TYPE_BASE = 0,    // inplace or return
    TYPE_INPLACE = 1, // only inplace
};
class ForeachCommonTiling
{
public:
    explicit ForeachCommonTiling(gert::TilingContext* context) : tilingContext(context) {};
    /**
     ** function: Init
     */
    ge::graphStatus Init(
        uint8_t theCode = 0, ForeachInputType inputType = ForeachInputType::TYPE_TENSORS,
        ForeachReturnType returnType = ForeachReturnType::TYPE_BASE)
    {
        opCode = theCode;
        opInputType = inputType;
        opReturnType = returnType;
        int dynamicIdx = opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE ? 1 : 0;
        for (uint32_t i = 0; i < MAX_TENSOR_CONT; i++) {
            auto srcTensor = tilingContext->GetDynamicInputTensor(dynamicIdx, i);
            if (srcTensor == nullptr) {
                break;
            }

            auto temp = tilingContext->GetInputDesc(0);
            if (temp == nullptr) {
                return ge::GRAPH_FAILED;
            }

            auto srcDtype = temp->GetDataType();

            if (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
                if (tilingContext->GetInputDesc(1) != nullptr) {
                    srcDtype = tilingContext->GetInputDesc(1)->GetDataType();
                } else {
                    return ge::GRAPH_FAILED;
                }
            }
            // Determine whether all data types are consistent.
            if (dataType == ge::DT_UNDEFINED) {
                dataType = srcDtype;
                dataTypeSize = GetDataTypeSize(dataType);
                if (dataTypeSize == 0) {
                    dataTypeSize = BYTE_LEN_4;
                }
                elementsPerBlock = static_cast<uint8_t>(BYTE_BLOCK / dataTypeSize);
            } else if (srcDtype != dataType) {
                return ge::GRAPH_FAILED;
            }
            if (opCode == FOREACH_COPY_OP_CODE) {
                auto tempDstDesc = tilingContext->GetOutputDesc(i);
                dstDataType = tempDstDesc->GetDataType();
            }
            gert::Shape tempShape = srcTensor->GetStorageShape();
            // Make a 32-byte alignment for each Tensor
            tensorDataCountList[i] = tempShape.GetShapeSize();
            totalDataCount += tensorDataCountList[i];
            totalTensorCount++;
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: RunBigKernelTiling
     */
    ge::graphStatus RunBigKernelTiling()
    {
        // checke op info
        OP_CHECK_IF(
            CheckOpInfo() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "CheckOpInfo failed."),
            return ge::GRAPH_FAILED);

        auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());

        uint64_t ubSizePlatForm = 0;
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        if (opCode == FOREACH_COPY_OP_CODE) {
            tilingContext->SetTilingKey(GetTilingKeyForForeachCopy(dataType, dstDataType));
        } else {
            tilingContext->SetTilingKey(GetTilingKeyByDtypeOnly(dataType));
        }

        uint32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());
        needCoreNum = needCoreNum > MAX_CORE_CONT ? MAX_CORE_CONT : needCoreNum;
        AssignDataToEachCore(static_cast<int64_t>(needCoreNum));
        DivideUbMemory(ubSizePlatForm);
        FillTilingData();
        tilingContext->SetBlockDim(needCoreNum);
        size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspaces[0] = WORK_SPACE_SIZE;

        return ge::GRAPH_SUCCESS;
    }

    /**
    ** function: RunBigScalarKernelTiling
    */
    ge::graphStatus RunBigScalarKernelTiling()
    {
        auto platformInfo = tilingContext->GetPlatformInfo();
        auto compileInfoPtr = reinterpret_cast<const ForeachCompileInfo*>(tilingContext->GetCompileInfo());

        uint64_t ubSizePlatForm = 0;
        uint32_t needCoreNum = 0;
        if (platformInfo != nullptr) {
            // checke op info
            OP_CHECK_IF(
                CheckOpInfo() != ge::GRAPH_SUCCESS, OP_LOGE(tilingContext->GetNodeName(), "CheckOpInfo failed."),
                return ge::GRAPH_FAILED);
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
            needCoreNum = GetNeedCoreNum(ascendcPlatform.GetCoreNumAiv());
        } else {
            ubSizePlatForm = compileInfoPtr->ubSize;
            needCoreNum = GetNeedCoreNum(compileInfoPtr->aivCoreNum);
        }
        needCoreNum = needCoreNum > MAX_CORE_CONT ? MAX_CORE_CONT : needCoreNum;
        tilingContext->SetTilingKey(GetTilingKeyByDtypeOnly(dataType));
        AssignDataToEachCore(static_cast<int64_t>(needCoreNum));
        DivideUbMemory(ubSizePlatForm);
        FillTilingData();
        tilingContext->SetBlockDim(needCoreNum);
        size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
        if (workspaces == nullptr) {
            return ge::GRAPH_FAILED;
        }
        workspaces[0] = WORK_SPACE_SIZE;

        return ge::GRAPH_SUCCESS;
    }

private:

    ge::graphStatus CheckForeachCopyDtype(ge::DataType dstDtype)
    {
        if (dstDtype != dstDataType) {
            OP_LOGE_FOR_INVALID_DTYPE(
                tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(dstDtype).c_str(),
                ge::TypeUtils::DataTypeToSerialString(dstDataType).c_str());
            return ge::GRAPH_FAILED;
        }
        if (dataType == dstDtype) {
            return ge::GRAPH_SUCCESS;
        }
        if (dataType == ge::DT_FLOAT && dstDtype == ge::DT_FLOAT16) {
            return ge::GRAPH_SUCCESS;
        } else if (dataType == ge::DT_FLOAT16 && dstDtype == ge::DT_FLOAT) {
            return ge::GRAPH_SUCCESS;
        } else if (dataType == ge::DT_FLOAT && dstDtype == ge::DT_BF16) {
            return ge::GRAPH_SUCCESS;
        } else if (dataType == ge::DT_BF16 && dstDtype == ge::DT_FLOAT) {
            return ge::GRAPH_SUCCESS;
        } else {
            std::string reasonMsg =
                "The dtype of y must be the same as x "
                "when the dtypes of x and y are not within the supported floating-point combinations: x is float16 or "
                "bf16 and y is float, or x is float and y is float16 or bf16";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                tilingContext->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(dstDtype).c_str(), reasonMsg);
            return ge::GRAPH_FAILED;
        }
    }

    /**
     ** function: Check output tensor list shape and dtype invalid
     */
    ge::graphStatus CheckOutputShapeAndDtype()
    {
        // inplace op does not contains output
        if (opReturnType == ForeachReturnType::TYPE_INPLACE) {
            return ge::GRAPH_SUCCESS;
        }
        uint64_t inputIndexZero = 0;
        if (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
            inputIndexZero = static_cast<uint64_t>(1);
        }

        for (uint32_t i = 0; i < totalTensorCount; i++) {
            auto tempDesc = tilingContext->GetOutputDesc(i);
            OP_CHECK_IF(
                tempDesc == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The output %u desc is null.", i),
                return ge::GRAPH_FAILED);
            auto dstDtype = tempDesc->GetDataType();
            if (opCode == FOREACH_COPY_OP_CODE) {
                OP_CHECK_IF(
                    CheckForeachCopyDtype(dstDtype) != ge::GRAPH_SUCCESS,
                    OP_LOGE(tilingContext->GetNodeName(), "The tensor %u of output datatype should meet the conditions with input.", i),
                    return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(
                    dstDtype != dataType,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        tilingContext->GetNodeName(), "y",
                        ge::TypeUtils::DataTypeToSerialString(dstDtype).c_str(),
                        "The dtype of y must be same as x"),
                    return ge::GRAPH_FAILED);
            }
            auto srcShape = tilingContext->GetDynamicInputShape(inputIndexZero, i);
            OP_CHECK_IF(
                srcShape == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The input %u shape is null.", i),
                return ge::GRAPH_FAILED);
            auto dstShape = tilingContext->GetOutputShape(i);
            OP_CHECK_IF(
                dstShape == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The output %u shape is null.", i),
                return ge::GRAPH_FAILED);
            if (srcShape->GetStorageShape() != dstShape->GetStorageShape() &&
                    srcShape->GetStorageShape().GetShapeSize() > dstShape->GetStorageShape().GetShapeSize()) {
                std::string reasonMsg = "The shape size of " + std::to_string(i) +
                                    "th tensor in tensor list y should be greater than or equal to that of the tensor "
                                    "in the same position of the another tensor list x";
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    tilingContext->GetNodeName(), "x and y",
                    (std::to_string(srcShape->GetStorageShape().GetShapeSize()) + " and " +
                     std::to_string(dstShape->GetStorageShape().GetShapeSize())).c_str(),
                    reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: Check scalar tensor shape  invalid
     */
    ge::graphStatus CheckScalarTenorShapeInfo(size_t inputTensorsNum)
    {
        size_t irIndex = inputTensorsNum;
        if (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
            irIndex = static_cast<size_t>(0);
        }
        auto scalarShape = tilingContext->GetRequiredInputShape(irIndex);
        OP_CHECK_IF(
            scalarShape == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The scalar shape is null."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            scalarShape->GetStorageShape().GetShapeSize() != 1,
            OP_LOGE_FOR_INVALID_SHAPESIZE(
                tilingContext->GetNodeName(), "scalar",
                std::to_string(scalarShape->GetStorageShape().GetShapeSize()).c_str(), "1"),
            return ge::GRAPH_FAILED);
        // check max dim
        OP_CHECK_IF(
            scalarShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIMS_NUMS,
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                tilingContext->GetNodeName(), "scalar",
                std::to_string(scalarShape->GetStorageShape().GetDimNum()).c_str(),
                "less than or equal to 8"),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: Check scalars tensor shape  invalid
     */
    ge::graphStatus CheckScalarsTenorShapeInfo(size_t inputTensorsNum)
    {
        size_t irIndex = inputTensorsNum;
        auto scalarsShape = tilingContext->GetRequiredInputShape(irIndex);
        OP_CHECK_IF(
            scalarsShape == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The scalar shape is null."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            scalarsShape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIMS_NUMS,
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                tilingContext->GetNodeName(), "scalars",
                std::to_string(scalarsShape->GetStorageShape().GetDimNum()).c_str(),
                "less than or equal to 8"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            scalarsShape->GetStorageShape().GetShapeSize() != totalTensorCount,
            OP_LOGE_FOR_INVALID_SHAPESIZE(
                tilingContext->GetNodeName(), "scalars",
                std::to_string(scalarsShape->GetStorageShape().GetShapeSize()).c_str(),
                (std::to_string(totalTensorCount)).c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: Check input tensorlist shape  invalid
     */
    ge::graphStatus CheckInputTensorlistShape(size_t inputTensorsNum)
    {
        uint64_t inputIndexZero = 0;
        if (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
            inputIndexZero = static_cast<uint64_t>(1);
            inputTensorsNum++;
        }

        for (size_t tensorIndex = 0; tensorIndex < totalTensorCount; ++tensorIndex) {
            auto x1Shape = tilingContext->GetDynamicInputShape(inputIndexZero, tensorIndex);
            OP_CHECK_IF(
                x1Shape == nullptr,
                OP_LOGE(tilingContext->GetNodeName(), "The input %lu shape is null.", inputIndexZero),
                return ge::GRAPH_FAILED);

            // check max dim
            OP_CHECK_IF(
                x1Shape->GetStorageShape().GetDimNum() > MAX_SUPPORT_DIMS_NUMS,
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    tilingContext->GetNodeName(), "x/x1",
                    std::to_string(x1Shape->GetStorageShape().GetDimNum()).c_str(),
                    "less than or equal to 8"),
                return ge::GRAPH_FAILED);

            // check tensorlist input shape consistent
            for (size_t listId = static_cast<size_t>(inputIndexZero) + 1U;
                 listId < static_cast<size_t>(inputTensorsNum); ++listId) {
                if (x1Shape->GetStorageShape() !=
                    tilingContext->GetDynamicInputShape(listId, tensorIndex)->GetStorageShape()) {
                    std::string errMsg = Ops::Base::ToString(
                        tilingContext->GetDynamicInputShape(listId, tensorIndex)->GetStorageShape());
                    std::string reasonMsg = "The shapes of " + std::to_string(tensorIndex) +
                                            "th tensor in tensor list input " + std::to_string(listId) + " and input " +
                                            std::to_string(inputIndexZero) + " must be the same";
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        tilingContext->GetNodeName(), ("input " + std::to_string(listId)).c_str(), errMsg.c_str(),
                        reasonMsg.c_str());
                        return ge::GRAPH_FAILED;
                }
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: Check input tensorlist size  invalid
     */
    ge::graphStatus CheckInputTensorlistSize(size_t inputTensorsNum)
    {
        uint64_t startIndex = 0;
        if (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
            startIndex = static_cast<uint64_t>(1);
            inputTensorsNum++;
        }
        auto computeNodeInfo = tilingContext->GetComputeNodeInfo();
        OP_CHECK_IF(
            computeNodeInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetComputeNodeInfo failed."),
            return ge::GRAPH_FAILED);

        auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(startIndex);
        OP_CHECK_IF(
            anchorInstanceInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetInputInstanceInfo failed."),
            return ge::GRAPH_FAILED);
        size_t xSize = anchorInstanceInfo->GetInstanceNum();

        for (uint64_t i = startIndex + 1U; i < static_cast<uint64_t>(inputTensorsNum); i++) {
            auto tempAnchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(i);
            OP_CHECK_IF(
                tempAnchorInstanceInfo == nullptr,
                OP_LOGE(tilingContext->GetNodeName(), "GetInputInstanceInfo failed."), return ge::GRAPH_FAILED);
            size_t otherSize = tempAnchorInstanceInfo->GetInstanceNum();
            if (otherSize != xSize){
                std::string otherName = "x" + std::to_string(i + 1);
                std::string paramName = "x1 and "+ otherName;
                std::string errMsg = std::to_string(xSize) + " and " + std::to_string(otherSize);
                std::string reasonMsg = "The tensorNums of dynamic input tensor lists x1 and " + otherName + " must be the same";
                OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
                    tilingContext->GetNodeName(), paramName,
                    errMsg.c_str(),
                    reasonMsg.c_str());
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: Check input and out shape and dtype info
     */
    ge::graphStatus CheckOpInfo()
    {
        auto computeNodeInfo = tilingContext->GetComputeNodeInfo();
        OP_CHECK_IF(
            computeNodeInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetComputeNodeInfo failed."),
            return ge::GRAPH_FAILED);

        auto inputTensorsNum = computeNodeInfo->GetIrInputsNum();
        if (opInputType != ForeachInputType::TYPE_TENSORS) {
            inputTensorsNum -= 1;
        }

        // check tensors num
        OP_CHECK_IF(
            CheckInputTensorlistSize(static_cast<size_t>(inputTensorsNum)),
            OP_LOGE(tilingContext->GetNodeName(), "CheckInputTensorlistSize failed."), return ge::GRAPH_FAILED);

        // check shape
        OP_CHECK_IF(
            CheckInputTensorlistShape(static_cast<size_t>(inputTensorsNum)) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "CheckInputTensorlistShape failed."), return ge::GRAPH_FAILED);
        if (opInputType == ForeachInputType::TYPE_SCALAR) {
            OP_CHECK_IF(
                CheckScalarTenorShapeInfo(static_cast<size_t>(inputTensorsNum)) != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext->GetNodeName(), "CheckScalarTenorShapeInfo failed."), return ge::GRAPH_FAILED);
        }
        if (opInputType == ForeachInputType::TYPE_SCALARS_TENSOR) {
            OP_CHECK_IF(
                CheckScalarsTenorShapeInfo(static_cast<size_t>(inputTensorsNum)) != ge::GRAPH_SUCCESS,
                OP_LOGE(tilingContext->GetNodeName(), "CheckScalarsTenorShapeInfo failed."), return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(
            CheckOutputShapeAndDtype() != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "CheckOutputShapeAndDtype failed."), return ge::GRAPH_FAILED);

        // check tensor list dtype
        OP_CHECK_IF(
            CheckTensorListDtype(static_cast<size_t>(inputTensorsNum)) != ge::GRAPH_SUCCESS,
            OP_LOGE(tilingContext->GetNodeName(), "CheckTensorListDtype() failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: Check tensorList dtype consistent
     */
    ge::graphStatus CheckTensorListDtype(size_t inputTensorsNum)
    {
        uint64_t startIndex = 0;
        if (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
            startIndex = static_cast<uint64_t>(1);
            inputTensorsNum++;
        }

        auto computeNodeInfo = tilingContext->GetComputeNodeInfo();
        OP_CHECK_IF(
            computeNodeInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetComputeNodeInfo failed."),
            return ge::GRAPH_FAILED);
        for (uint64_t i = startIndex; i < inputTensorsNum; i++) {
            auto anchorInstanceInfo = computeNodeInfo->GetInputInstanceInfo(i);
            OP_CHECK_IF(
                anchorInstanceInfo == nullptr, OP_LOGE(tilingContext->GetNodeName(), "GetInputInstanceInfo failed."),
                return ge::GRAPH_FAILED);
            for (uint32_t j = 0; j < totalTensorCount; j++) {
                auto tempDesc = tilingContext->GetDynamicInputDesc(i, j);
                OP_CHECK_IF(
                    tempDesc == nullptr, OP_LOGE(tilingContext->GetNodeName(), "The input %u desc is null.", j),
                    return ge::GRAPH_FAILED);
                auto checkDtype = tempDesc->GetDataType();
                // Determine whether all data types are consistent.
                if (dataType == ge::DT_UNDEFINED) {
                    return ge::GRAPH_FAILED;
                } else if (dataType != checkDtype) {
                    std::string paramName = "x" + std::to_string(i + 1);
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        tilingContext->GetNodeName(), paramName.c_str(),
                        ge::TypeUtils::DataTypeToSerialString(checkDtype).c_str(),
                        ("The dtypes of all tensors in the tensor list must be the same. "
                         "Currently, the dtype of the " + std::to_string(j) + "th tensor is inconsistent with that (" +
                         ge::TypeUtils::DataTypeToSerialString(dataType) + ") of other tensors").c_str());
                    return ge::GRAPH_FAILED;
                }
            }
        }
        return ge::GRAPH_SUCCESS;
    }

    /**
     ** function: CeilA2B
     */
    template <typename T1, typename T2>
    inline auto CeilA2B(T1 a, T2 b) const -> T1
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    /**
     ** function: GetTilingN
     */
    uint64_t GetTilingN()
    {
        switch (dataType) {
            case ge::DT_FLOAT:
                return TILING_FLOAT_N_SCALAR;
            case ge::DT_FLOAT16:
                return TILING_HALF_N_SCALAR;
            case ge::DT_INT32:
                return TILING_INT_N_SCALAR;
            case ge::DT_BF16:
                return TILING_BF16_N_SCALAR;
            default:
                return TILING_HALF_N_SCALAR;
        }
    }

    /**
     ** function: GetNeedCoreNum
     */
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform)
    {
        uint32_t tempCoreNum = static_cast<uint32_t>(CeilA2B(totalDataCount, elementsPerBlock));
        if (tempCoreNum == 0U) {
            tempCoreNum = static_cast<uint32_t>(1);
        }
        if (tempCoreNum < coreNumPlatform) {
            return tempCoreNum;
        } else {
            return coreNumPlatform;
        }
    }

    /**
     ** function: AssignDataToEachCore
     */
    void AssignDataToEachCore(int64_t needCoreNum)
    {
        // Kernel the input data according to 32 byte alignment.
        int64_t blockCount = CeilA2B(totalDataCount, elementsPerBlock);
        // Divisible, representing the amount of data each core needs to process.
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        int64_t tempPerCoreCount = blockCount / needCoreNum * static_cast<int64_t>(elementsPerBlock);
        int64_t remainderCount = blockCount % needCoreNum; // remainder.
        uint16_t coreIndex = 0;
        int64_t dataCount = 0;
        int64_t curCmpCount = 0;
        int64_t cursorPosition = 0;
        tensorStartList[coreIndex] = static_cast<uint16_t>(0);
        tensorStartOffsetList[coreIndex] = 0;
        for (uint16_t i = 0; i < totalTensorCount; i++) {
            // When the remainder is not 0, each kernel index with less than the remainder processes one more block of
            // data.
            if (remainderCount && static_cast<int64_t>(coreIndex) < remainderCount) {
                curCmpCount = tempPerCoreCount + static_cast<int64_t>(elementsPerBlock);
            } else {
                curCmpCount = tempPerCoreCount;
            }
            int64_t tempCount = tensorDataCountList[i] - cursorPosition;

            if (dataCount + tempCount < curCmpCount) {
                dataCount += tempCount;
                cursorPosition = 0;
                continue;
            }
            // dataCount >= curCmpCount, Calculate the offset
            tensorEndList[coreIndex] = i;
            cursorPosition = cursorPosition + curCmpCount - dataCount;
            tensorEndOffsetList[coreIndex] = cursorPosition - 1;
            dataCount = 0;
            coreIndex++;
            if (cursorPosition < tensorDataCountList[i]) {
                tensorStartList[coreIndex] = i;
                tensorStartOffsetList[coreIndex] = cursorPosition;
                --i; // The next loop continues to allocate the current tensor
            } else if (static_cast<int64_t>(coreIndex) != needCoreNum) {
                tensorStartList[coreIndex] = static_cast<uint16_t>(i + 1);
                tensorStartOffsetList[coreIndex] = 0;
                cursorPosition = 0;
            }
        }
        /* The temporary count variable is not 0, which means that the last tensor is truncated,
            and you need to manually set the offset of the last core. */
        if (dataCount) {
            tensorEndList[coreIndex] = static_cast<uint16_t>(totalTensorCount - 1);
            tensorEndOffsetList[coreIndex] = tensorDataCountList[totalTensorCount - 1] - 1;
        }
    }

    /**
     ** funtion: DivideUbMemory
     */
    void DivideUbMemory(uint64_t ubSizePlatForm)
    {
        if (opCode <= FOREACH_POINTWISE_OP_CODE) {
            DivideUbMemory1(ubSizePlatForm);
        } else if (opCode <= FOREACH_POW_TENSOR_OP_CODE) {
            DivideUbMemory2(ubSizePlatForm);
        } else if (opCode <= FOREACH_ERF_OP_CODE) {
            DivideUbMemory3(ubSizePlatForm);
        } else if (opCode <= FOREACH_TAN_OP_CODE) {
            DivideUbMemory4(ubSizePlatForm);
        } else if (opCode <= FOREACH_ATAN_OP_CODE) {
            DivideUbMemory5(ubSizePlatForm);
        } else if (opCode <= FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE) {
            DivideUbMemory6(ubSizePlatForm);
        } else if (opCode <= FOREACH_MUL_SCALAR_OP_CODE) {
            DivideUbMemory7(ubSizePlatForm);
        } else if (opCode <= FOREACH_ADD_LIST_OP_CODE) {
            DivideUbMemory8(ubSizePlatForm);
        } else if (opCode <= FOREACH_DIV_SCALAR_OP_CODE) {
            DivideUbMemory9(ubSizePlatForm);
        } else if (opCode <= FOREACH_COPY_OP_CODE) {
            DivideUbMemory9(ubSizePlatForm);
        } else if (opCode <= FOREACH_SIGN_OP_CODE) {
            DivideUbMemory10(ubSizePlatForm);
        }
    }

    /**
     ** funtion: DivideUbMemory1
     */
    void DivideUbMemory1(uint64_t ubSizePlatForm)
    {
        if (opCode == ZERO_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_add_scalar/add_scalar_list/expm1/sqrt/zero_inplace
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / 2U);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == SOLO_LOG_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_log/log1p/log10
            uint32_t totalSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / 2U);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == BINARY_LIST_OP_CODE) {
            // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
            // foreach_div_list/minimum_list/mul_list/sub_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_POINTWISE_OP_CODE) {
            // foreach_addcdiv_scalar/addcdiv_scalar_list/addcmul_scalar/addcmul_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_POINTWISE_DIVIDER; // double buffer
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ?
                                     canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                     canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory2
     */
    void DivideUbMemory2(uint64_t ubSizePlatForm)
    {
        if (opCode == FOREACH_COS_OP_CODE) { // foreach_cos
            uint32_t tilingConstant = 6;
            if (dataTypeSize == BYTE_LEN_4) {
                tilingConstant = TILING_FLOAT_N_SCALAR;
            }
            uint32_t reserveUbSize = BYTE_BASIC_BLOCK * tilingConstant * dataTypeSize;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - reserveUbSize);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == SOLO_LOG2_OP_CODE) { // foreach_log2
            uint32_t extraBuf = 0;                // need extra space
            GetLog2TmpBufferFactorSize(
                dataTypeSize, extraBuf, LOG2_HALF_FOR_LOG2, LOG2_FLOAT_FOR_LOG2,
                LOG2_BASIC_FOR_LOG2); // reuse source is true
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / 2U - extraBuf);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == SOLO_NEG_OP_CODE) { // need extra buffer of one block: 32 bytes  foreach_neg/reciprocal
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / 2U - static_cast<uint32_t>(BYTE_PER_BLOCK);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_POW_TENSOR_OP_CODE) { // foreach_pow_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            uint32_t canUseUbSize;
            if (dataType == ge::DT_BF16) {
                canUseUbSize = totalSize / (BINARY_LIST_UB_DIVIDER * UB_DIVIDER_FOR_TEMP_CASTING +
                                            POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyByDtypeOnly(dataType) - 1]);
            } else {
                canUseUbSize = totalSize / (BINARY_LIST_UB_DIVIDER +
                                            POW_TENSOR_TENSOR_CALC_PROC[GetTilingKeyByDtypeOnly(dataType) - 1]);
            }
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory3
     */
    void DivideUbMemory3(uint64_t ubSizePlatForm)
    {
        if (opCode == FOREACH_BINARY_SCALAR_OP_CODE) {
            // foreach_maximum_scalar/maximum_scalar_list/minimum_scalar/minimum_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_SCALAR_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_POINTWISE_LIST_OP_CODE) {
            // foreach_addcdiv_list, foreach_addcmul_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_POINTWISE_LIST_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_SIGMOID_OP_CODE) {
            // foreach_sigmoid
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 1024);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_SCALAR_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_ERF_OP_CODE) {
            // foreach_erf
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            // erf ascend C need 3 times for float or 8 times for half inputData size reserved for every buffer
            uint32_t canUseUbSize = totalSize / FOREACH_ERF_FLOAT_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
            if (dataTypeSize == BYTE_LEN_2) {
                canUseUbSize = totalSize / FOREACH_ERF_HALF_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
            }
            // 32 bytes align
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory4
     */
    void DivideUbMemory4(uint64_t ubSizePlatForm)
    {
        if ((opCode == FOREACH_ASIN_OP_CODE)) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_cosh/asin/acos
            uint32_t calcPro = COSH_HALF_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = COSH_FLOAT_CALC_PROC;
            }
            uint32_t extraBuffer = calcPro * static_cast<uint32_t>(dataTypeSize) * COSH_BASIC_BLOCK_SIZE * 8U;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_SINH_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_sinh
            uint32_t calcPro = SINH_HALF_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = SINH_FLOAT_CALC_PROC;
            }
            uint32_t extraBuffer = calcPro * dataTypeSize * SINH_BASIC_BLOCK_SIZE;
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_TAN_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_tan
            uint32_t calcPro = TAN_HALF_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = TAN_FLOAT_CALC_PROC;
            }
            uint32_t extraBuffer = calcPro * static_cast<uint32_t>(dataTypeSize) * TAN_BASIC_BLOCK_SIZE * 8U;
            while (static_cast<uint32_t>(ubSizePlatForm) <= static_cast<uint32_t>(tilingData.GetDataSize() + extraBuffer)) {
                extraBuffer = extraBuffer / BUFFER_ATTENUATION;
            }
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory5
     */
    void DivideUbMemory5(uint64_t ubSizePlatForm)
    {
        if (opCode == FOREACH_ERFC_OP_CODE) {
            // foreach_erfc
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            // erfc ascend C need 7 times for float or 16 times for half inputData size reserved for every buffer
            uint32_t canUseUbSize = totalSize / FOREACH_ERFC_FLOAT_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
            if (dataTypeSize == BYTE_LEN_2) {
                canUseUbSize = totalSize / FOREACH_ERFC_HALF_DIVIDER / FOREACH_ERF_BUFFER_DIVIDER;
            }
            // 32 bytes align
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_TANH_OP_CODE) {
            // foreach_tanh
            uint32_t calcPro = TANH_FLOAT_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_2) {
                calcPro = TANH_HALF_CALC_PROC;
            }
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 1024);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / (calcPro + FOREACH_TANH_DIVIDER);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_ATAN_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_atan
            uint32_t calcPro = ATAN_HALF_CALC_PROC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = ATAN_FLOAT_CALC_PROC;
            }
            uint32_t extraBuffer = calcPro * static_cast<uint32_t>(dataTypeSize) * ATAN_BASIC_BLOCK_SIZE * 8U;
            while (static_cast<uint32_t>(ubSizePlatForm) <= static_cast<uint32_t>(tilingData.GetDataSize() + extraBuffer)) {
                extraBuffer = extraBuffer / BUFFER_ATTENUATION;
            }
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - extraBuffer);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_COS_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory6
     */
    void DivideUbMemory6(uint64_t ubSizePlatForm)
    {
        if (opCode == FOREACH_LERP_SCALAR_OP_CODE) {
            // foreach_lerp_scalar
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 128);
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_LERP_SCALAR_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ?
                                     canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                     canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_LERP_LIST_OP_CODE) {
            // foreach_lerp_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_LERP_LIST_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ?
                                     canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                     canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
            inputsTensorUbSize = inputsTensorUbSize / BYTE_PER_REPEAT * BYTE_PER_REPEAT;
        } else if ((opCode == FOREACH_POW_SCALAR_OP_CODE) || (opCode == FOREACH_POW_SCALAR_AND_TENSOR_OP_CODE)) {
            // foreach_pow_scalar/pow_scalar_list/pow_scalar_and_tensor
            uint32_t reserveUbSize = static_cast<uint32_t>(BYTE_BASIC_BLOCK * GetTilingN() * dataTypeSize);
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - reserveUbSize);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / FOREACH_POW_SCALAR_DIVIDER; // double buffer
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory7
     */
    void DivideUbMemory7(uint64_t ubSizePlatForm)
    {
        if (opCode == FOREACH_SIN_OP_CODE) {
            // foreach_sin
            uint32_t calcPro = SIN_HALF_CALC_FAC;
            if (dataTypeSize == BYTE_LEN_4) {
                calcPro = SIN_FLOAT_CALC_FAC;
            }
            uint32_t reservedUbSize = static_cast<uint32_t>(4U * SIN_BASIC_BLOCK * calcPro * dataTypeSize);
            uint32_t totalSize = static_cast<uint32_t>(
                ubSizePlatForm - static_cast<uint32_t>(tilingData.GetDataSize()) - reservedUbSize);
            if (dataType == ge::DT_BF16) {
                totalSize = static_cast<uint32_t>(totalSize / UB_DIVIDER_FOR_TEMP_CASTING);
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / FOREACH_SIN_DIVIDER); // 4
            inputsTensorUbSize = static_cast<uint32_t>(canUseUbSize / BYTE_BLOCK * BYTE_BLOCK);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_ABS_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_abs
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - 2048);
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BYTE_LEN_4;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_MUL_SCALAR_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_mul_scalar/mul_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / 2U);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory8
     */
    void DivideUbMemory8(uint64_t ubSizePlatForm)
    {
        if (opCode == FOREACH_EXP_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_exp
            uint32_t totalSize = uint32_t(ubSizePlatForm - 1024 - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / 2U);
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_MAXIMUM_LIST_OP_CODE) {
            // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
            // foreach_maximum_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_ADD_LIST_OP_CODE) {
            // The remaining UB size is split in six, double buffer enabled, and rounded down 32 bytes.
            // foreach_add_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_BF16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = totalSize / BINARY_LIST_UB_DIVIDER;
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** funtion: DivideUbMemory9
     */
    void DivideUbMemory9(uint64_t ubSizePlatForm)
    {
        if (opCode == FOREACH_ROUND_OFF_NUM_OP_CODE) {
            // foreach_round_off_number
            uint32_t canUseUbSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize()) / 2;
            uint32_t predictSGUbSize = uint32_t(BYTE_REPEAT / (BYTE_REPEAT + 2.0 * dataTypeSize) * canUseUbSize);
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                predictSGUbSize = predictSGUbSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ?
                                     predictSGUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                     predictSGUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_SUB_SCALAR_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_sub_scalar/sub_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - BYTE_BLOCK);
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / 4U); // one block bytes is 32
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ?
                                     canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                     canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_DIV_SCALAR_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_div_scalar/div_scalar_list
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize() - BYTE_BLOCK);
            if (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / 4U); // one block bytes is 32
            inputsTensorUbSize = (dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT16) ?
                                     canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                     canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        } else if (opCode == FOREACH_COPY_OP_CODE) {
            // The remaining UB size is one buffer enabled, and rounded down 32 bytes.
            // foreach_copy
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());

            uint32_t canUseUbSize = totalSize;
            inputsTensorUbSize = static_cast<uint64_t>(canUseUbSize / BYTE_BLOCK * BYTE_BLOCK);
            if (dataType != dstDataType) {
                inputsTensorUbSize /= BUFFER_ATTENUATION;
            }
        }
    }

    /**
     ** funtion: DivideUbMemory10
     */
    void DivideUbMemory10(uint64_t ubSizePlatForm)
    {
        if (opCode == FOREACH_SIGN_OP_CODE) {
            // The remaining UB size is split in two, double buffer enabled, and rounded down 32 bytes.
            // foreach_sign
            uint32_t totalSize = uint32_t(ubSizePlatForm - tilingData.GetDataSize());
            if (dataType == ge::DT_FLOAT || dataType == ge::DT_FLOAT16) {
                uint32_t extraBuffer = SIGN_CALC_PROC * dataTypeSize * SIGN_BASIC_BLOCK_SIZE * 8;
                totalSize = totalSize - extraBuffer;
            }
            if (dataType == ge::DT_BF16 || dataType == ge::DT_INT64 || dataType == ge::DT_INT8) {
                totalSize = totalSize / UB_DIVIDER_FOR_TEMP_CASTING;
            }
            uint32_t canUseUbSize = static_cast<uint32_t>(totalSize / 4U); // one block bytes is 32
            inputsTensorUbSize = (dataType == ge::DT_BF16) ? canUseUbSize / BYTE_BLOCK_FOR_BF16 * BYTE_BLOCK_FOR_BF16 :
                                                             canUseUbSize / BYTE_BLOCK * BYTE_BLOCK;
        }
    }

    /**
     ** function: FillTilingData
     */
    void FillTilingData()
    {
        tilingData.set_inputsTensorUbSize(inputsTensorUbSize);
        tilingData.set_tensorDataCountList(tensorDataCountList);
        tilingData.set_tensorStartList(tensorStartList);
        tilingData.set_tensorEndList(tensorEndList);
        tilingData.set_tensorStartOffsetList(tensorStartOffsetList);
        tilingData.set_tensorEndOffsetList(tensorEndOffsetList);

        tilingData.SaveToBuffer(
            tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
        tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

    /**
     ** function: GetLog2TmpBufferFactorSize
     */
    void GetLog2TmpBufferFactorSize(
        const uint32_t typeSize, uint32_t& extraBuf, uint32_t log2Half = LOG2_HALF_FOR_LOG2,
        uint32_t log2Float = LOG2_FLOAT_FOR_LOG2, uint32_t log2Basic = LOG2_BASIC_FOR_LOG2)
    {
        auto caclFactor = (typeSize == sizeof(float)) ? log2Float : log2Half;
        extraBuf = log2Basic * caclFactor * typeSize;
    }

private:
    ForeachCommonTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;

    ge::DataType dataType = ge::DT_UNDEFINED;
    ge::DataType dstDataType = ge::DT_UNDEFINED;

    uint64_t inputsTensorUbSize = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    int64_t totalDataCount = 0;
    uint8_t dataTypeSize = 4;
    uint8_t elementsPerBlock = 0;
    uint16_t totalTensorCount = 0;
    uint8_t opCode = 0;
    ForeachInputType opInputType = ForeachInputType::TYPE_TENSORS;
    ForeachReturnType opReturnType = ForeachReturnType::TYPE_BASE;
};

} // namespace optiling

#endif // FOREACH_TILING_CLASS_H_
