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
 * \file add_rms_norm_quant_tiling_arch35.cpp
 * \brief
 */

#include "add_rms_norm_quant_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

constexpr uint32_t LOG_2 = 2;
constexpr uint32_t MAX_DIM_CNT = 8;
constexpr uint32_t B32_BLOCK_NUM = 8;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ALING_FACTOR_512 = 512;
constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr uint32_t DEFAULT_SYS_WORKSPACE = 16 * 1024 * 1024;

constexpr uint32_t LEVEL_BUFFER_CNT = 3;
constexpr uint32_t MULTI_FACTOR_2 = 2;
constexpr uint32_t ZERO_POINTS1_BIN_OFFSET = 2;
constexpr uint32_t SCALES2_BIN_OFFSET = 1;

ge::graphStatus AddRmsNormQuantRegbaseTiling::CheckDtypeVaild(
    ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList, string srcName)
{
    for (const auto& supportedDtype : supportDtypeList) {
        if (supportedDtype == srcDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(
        nodeName.c_str(), "Dtype check invalid, %s dtype is %s, not in supportDtypeList.", srcName.c_str(),
        Ops::Base::ToString(srcDtype).c_str());
    return ge::GRAPH_FAILED;
}

bool AddRmsNormQuantRegbaseTiling::CheckShapeNull()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling CheckShapeNull.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* scales1Shape = context_->GetInputShape(SCALES1_INDEX);
    const gert::StorageShape* y1Shape = context_->GetOutputShape(Y1_INDEX);
    const gert::StorageShape* y2Shape = context_->GetOutputShape(Y2_INDEX);
    const gert::StorageShape* xShape = context_->GetOutputShape(X_INDEX);

    OP_CHECK_IF(
        (nullptr == x1Shape) || (nullptr == x2Shape) || (nullptr == gammaShape) || (nullptr == scales1Shape) ||
            (nullptr == y1Shape) || (nullptr == y2Shape) || (nullptr == xShape),
        , return false);
    return true;
}

void AddRmsNormQuantRegbaseTiling::CheckOptionalInput()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling CheckOptionalInput.");
    const gert::StorageShape* scales2Shape = context_->GetOptionalInputShape(SCALES2_INDEX);
    const gert::StorageShape* zeroPoints1Shape = context_->GetOptionalInputShape(ZERO_POINTS1_INDEX);
    const gert::StorageShape* zeroPoints2Shape = context_->GetOptionalInputShape(ZERO_POINTS2_INDEX);

    tilingParams.quantBufCnt = 0;
    if (scales2Shape != nullptr) {
        tilingParams.hasScales2 = true;
        tilingParams.quantBufCnt++;
    }
    if (zeroPoints1Shape != nullptr) {
        tilingParams.hasZeroPoints1 = true;
        tilingParams.quantBufCnt++;
    }
    if (zeroPoints2Shape != nullptr) {
        tilingParams.hasZeroPoints2 = true;
        tilingParams.quantBufCnt++;
    }
    tilingParams.hasY2 = tilingParams.hasScales2 || tilingParams.hasZeroPoints2;
}

bool AddRmsNormQuantRegbaseTiling::CheckInputShapeDim()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling CheckInputShapeDim.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* scales1Shape = context_->GetInputShape(SCALES1_INDEX);
    const gert::StorageShape* scales2Shape = context_->GetOptionalInputShape(SCALES2_INDEX);
    const gert::StorageShape* zp1Shape = context_->GetOptionalInputShape(ZERO_POINTS1_INDEX);
    const gert::StorageShape* zp2Shape = context_->GetOptionalInputShape(ZERO_POINTS2_INDEX);

    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t scales1DimNum = scales1Shape->GetStorageShape().GetDimNum();
    size_t scales2DimNum = 0;
    if (tilingParams.hasScales2) {
        scales2DimNum = scales2Shape->GetStorageShape().GetDimNum();
    }
    size_t zp1DimNum = 0;
    if (tilingParams.hasZeroPoints1) {
        zp1DimNum = zp1Shape->GetStorageShape().GetDimNum();
    }
    size_t zp2DimNum = 0;
    if (tilingParams.hasZeroPoints2) {
        zp2DimNum = zp2Shape->GetStorageShape().GetDimNum();
    }
    OP_CHECK_IF(
        (x1DimNum > MAX_DIM_CNT) || (x2DimNum > MAX_DIM_CNT) || (gammaDimNum > MAX_DIM_CNT) ||
            (scales1DimNum > MAX_DIM_CNT) || (scales2DimNum > MAX_DIM_CNT && tilingParams.hasScales2) ||
            (zp1DimNum > MAX_DIM_CNT && tilingParams.hasZeroPoints1) ||
            (zp2DimNum > MAX_DIM_CNT && tilingParams.hasZeroPoints2),
        OP_LOGE(nodeName.c_str(), "All input dim should not bigger than %u.", MAX_DIM_CNT), return false);
    return true;
}

bool AddRmsNormQuantRegbaseTiling::CheckInputShapeValue()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling CheckInputShapeValue.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* scales1Shape = context_->GetInputShape(SCALES1_INDEX);
    const gert::StorageShape* scales2Shape = context_->GetOptionalInputShape(SCALES2_INDEX);
    const gert::StorageShape* zeroPoints1Shape = context_->GetOptionalInputShape(ZERO_POINTS1_INDEX);
    const gert::StorageShape* zeroPoints2Shape = context_->GetOptionalInputShape(ZERO_POINTS2_INDEX);
    const gert::StorageShape* y1Shape = context_->GetOutputShape(Y1_INDEX);
    const gert::StorageShape* y2Shape = context_->GetOutputShape(Y2_INDEX);
    const gert::StorageShape* xShape = context_->GetOutputShape(X_INDEX);

    // Check x1&x2&y1&y2&x's shape should be equal
    if (!NormCheck::CheckShapeSame(x1Shape, x2Shape, nodeName, "x1", "x2")) {
        return false;
    };
    if (!NormCheck::CheckShapeSame(x1Shape, y1Shape, nodeName, "x1", "y1")) {
        return false;
    };
    if (tilingParams.hasScales2 && !NormCheck::CheckShapeSame(x1Shape, y2Shape, nodeName, "x1", "y2")) {
        return false;
    };
    if (!NormCheck::CheckShapeSame(x1Shape, xShape, nodeName, "x1", "x")) {
        return false;
    };

    // Check gamma&scales&zeropoints's shape should be equal
    if (!NormCheck::CheckShapeLastDim(gammaShape, scales1Shape, nodeName, "gamma", "scales1")) {
        return false;
    };

    // Check gamma&scales&zeropoints's shape Numel should be equal
    if (!NormCheck::CheckShapeNumel(gammaShape, scales1Shape, nodeName, "gamma", "scales1")) {
        return false;
    };

    if (tilingParams.hasScales2 &&
        !NormCheck::CheckShapeSame(scales1Shape, scales2Shape, nodeName, "scales1", "scales2")) {
        return false;
    };
    if (tilingParams.hasZeroPoints1 &&
        !NormCheck::CheckShapeSame(scales1Shape, zeroPoints1Shape, nodeName, "scales1", "zeroPoints1")) {
        return false;
    };
    if (tilingParams.hasZeroPoints2 &&
        !NormCheck::CheckShapeSame(scales1Shape, zeroPoints2Shape, nodeName, "scales1", "zeroPoints2")) {
        return false;
    };

    // Check gamma should be last few dim of x
    if (!NormCheck::CheckShapeBC(x1Shape, gammaShape, nodeName, "x1", "gamma")) {
        return false;
    };

    return true;
}

bool AddRmsNormQuantRegbaseTiling::CheckInputDtype()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling CheckInputDtype.");
    std::vector<ge::DataType> supportedXGammaDtypes = {
        ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    std::vector<ge::DataType> supportedScalesDtypes = {
        ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    std::vector<ge::DataType> supportedZeroPointsDtypes = {
        ge::DataType::DT_INT32, ge::DataType::DT_FLOAT, ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16};
    std::vector<ge::DataType> supportedYDtypes = {ge::DataType::DT_INT8};

    const uint32_t totalCheckCnt = 7;
    string checkNameList[totalCheckCnt] = {"x1", "x2", "gamma", "scales1", "scales2", "zeroPoints1", "zeroPoints2"};
    uint32_t idxList[totalCheckCnt] = {X1_INDEX,      X2_INDEX,           GAMMA_INDEX,       SCALES1_INDEX,
                                       SCALES2_INDEX, ZERO_POINTS1_INDEX, ZERO_POINTS2_INDEX};
    bool isOptionalList[totalCheckCnt] = {false, false, false, false, true, true, true};
    bool needCheckList[totalCheckCnt] = {
        true, true, true, true, tilingParams.hasScales2, tilingParams.hasZeroPoints1, tilingParams.hasZeroPoints2};
    std::vector<ge::DataType>* supportedList[totalCheckCnt] = {
        &supportedXGammaDtypes, &supportedXGammaDtypes,     &supportedXGammaDtypes,    &supportedScalesDtypes,
        &supportedScalesDtypes, &supportedZeroPointsDtypes, &supportedZeroPointsDtypes};

    for (uint32_t checkIdx = 0; checkIdx < totalCheckCnt; checkIdx++) {
        if (!needCheckList[checkIdx]) {
            continue;
        }

        ge::DataType srcDtype;
        if (isOptionalList[checkIdx]) {
            srcDtype = context_->GetOptionalInputTensor(idxList[checkIdx])->GetDataType();
        } else {
            srcDtype = context_->GetInputTensor(idxList[checkIdx])->GetDataType();
        }
        if (ge::GRAPH_SUCCESS != CheckDtypeVaild(srcDtype, *(supportedList[checkIdx]), checkNameList[checkIdx])) {
            return false;
        };
    }

    ge::DataType x1Dtype = context_->GetInputTensor(X1_INDEX)->GetDataType();
    ge::DataType x2Dtype = context_->GetInputTensor(X2_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();
    ge::DataType scales1Dtype = context_->GetInputTensor(SCALES1_INDEX)->GetDataType();
    ge::DataType scales2Dtype = ge::DT_BOOL;     // Init one not support dtype
    ge::DataType zeroPoints1Dtype = ge::DT_BOOL; // Init one not support dtype
    ge::DataType zeroPoints2Dtype = ge::DT_BOOL; // Init one not support dtype
    if (tilingParams.hasScales2) {
        scales2Dtype = context_->GetOptionalInputTensor(SCALES2_INDEX)->GetDataType();
    }
    if (tilingParams.hasZeroPoints1) {
        zeroPoints1Dtype = context_->GetOptionalInputTensor(ZERO_POINTS1_INDEX)->GetDataType();
    }
    if (tilingParams.hasZeroPoints2) {
        zeroPoints2Dtype = context_->GetOptionalInputTensor(ZERO_POINTS2_INDEX)->GetDataType();
    }
    if ((x1Dtype != x2Dtype) || (x1Dtype != gammaDtype)) {
        OP_LOGE(nodeName.c_str(), "Input x1/gamma/xout dtype should be equal.");
        return false;
    }
    if ((x1Dtype != scales1Dtype) && (scales1Dtype != ge::DataType::DT_FLOAT)) {
        OP_LOGE(nodeName.c_str(), "Input x1/scalse1 dtype should be equal or scalse1 be fp32.");
        return false;
    }
    if (tilingParams.hasScales2 && (scales1Dtype != scales2Dtype)) {
        OP_LOGE(nodeName.c_str(), "Input scalse1/scalse2 dtype should be equal.");
        return false;
    }
    if (tilingParams.hasZeroPoints1) {
        if (zeroPoints1Dtype == ge::DataType::DT_INT32) {
            if (scales1Dtype != ge::DataType::DT_FLOAT) {
                OP_LOGE(nodeName.c_str(), "Input zeroPoints1 dtype is int32, scales1 dtype should be fp32.");
                return false;
            }
            if ((x1Dtype != ge::DataType::DT_FLOAT16) && (x1Dtype != ge::DataType::DT_BF16)) {
                OP_LOGE(nodeName.c_str(), "Input zeroPoints1 dtype is int32, x1 dtype should be fp16 or bf16.");
                return false;
            }
        } else {
            if (zeroPoints1Dtype != scales1Dtype) {
                OP_LOGE(nodeName.c_str(), "Input zeroPoints1 dtype not int32, should be equal to scales1 dtype.");
                return false;
            }
        }
    }
    if (tilingParams.hasZeroPoints2) {
        if (zeroPoints2Dtype == ge::DataType::DT_INT32) {
            if (tilingParams.hasScales2 && (scales2Dtype != ge::DataType::DT_FLOAT)) {
                OP_LOGE(nodeName.c_str(), "Input zeroPoints2 dtype is int32, scales2 dtype should be fp32.");
                return false;
            }
            if ((x1Dtype != ge::DataType::DT_FLOAT16) && (x1Dtype != ge::DataType::DT_BF16)) {
                OP_LOGE(nodeName.c_str(), "Input zeroPoints2 dtype is int32, x1 dtype should be fp16 or bf16.");
                return false;
            }
        } else {
            if (tilingParams.hasScales2 && (zeroPoints2Dtype != scales2Dtype)) {
                OP_LOGE(nodeName.c_str(), "Input zeroPoints2 dtype not int32, should be equal to scales2 dtype.");
                return false;
            }
        }
    }

    return true;
}

ge::graphStatus AddRmsNormQuantRegbaseTiling::SetInputParams()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling SetInputParams.");
    // Set input dim
    const gert::Shape x1Shape = context_->GetInputShape(X1_INDEX)->GetStorageShape();
    const gert::Shape gammaShape = context_->GetInputShape(GAMMA_INDEX)->GetStorageShape();
    size_t x1DimNum = x1Shape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();
    uint64_t numM = 1;
    uint64_t numN = 1;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        numM *= x1Shape.GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numN *= gammaShape.GetDim(i);
    }
    tilingParams.numM = numM;
    tilingParams.numN = numN;

    // Set input dtype
    auto xDataType = context_->GetInputTensor(X_INDEX)->GetDataType();
    auto quantDataType = context_->GetInputTensor(SCALES1_INDEX)->GetDataType();
    tilingParams.xDtypeSize = GetSizeByDataType(xDataType);
    tilingParams.quantDtypeSize = GetSizeByDataType(quantDataType);
    tilingParams.xDtypeAlignNum = BLOCK_SIZE / tilingParams.xDtypeSize;
    tilingParams.xReduceAlignNum = ALING_FACTOR_512 / tilingParams.xDtypeSize;
    tilingParams.quantDtypeAlignNum = BLOCK_SIZE / tilingParams.quantDtypeSize;
    tilingParams.vecLength = Ops::Base::GetVRegSize(context_) / sizeof(float);

    // Set input attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilon = attrs->GetFloat(EPS_ATTR_INDEX);
    tilingParams.epsilon = *epsilon;
    if (0 == numN) {
        OP_LOGE(nodeName.c_str(), "Can not div zero.");
        return ge::GRAPH_FAILED;
    }
    tilingParams.avgFactor = 1.0f / static_cast<float>(numN);

    const bool* divModePtr = attrs->GetBool(DIV_MODE_ATTR_INDEX);
    tilingParams.divMode = (divModePtr == nullptr) ? true : *divModePtr;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormQuantRegbaseTiling::GetShapeAttrsInfo()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling GetShapeAttrsInfo.");
    OP_CHECK_IF(
        !CheckShapeNull(), OP_LOGE(nodeName.c_str(), "The not optional input is null."), return ge::GRAPH_FAILED);
    CheckOptionalInput();
    OP_CHECK_IF(
        !CheckInputShapeDim(), OP_LOGE(nodeName.c_str(), "The input shape dim is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckInputShapeValue(), OP_LOGE(nodeName.c_str(), "The input shape relationship is invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputDtype(), OP_LOGE(nodeName.c_str(), "The input dtype is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(nodeName.c_str(), "Set input shape failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormQuantRegbaseTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling GetPlatformInfo.");
    if (context_->GetCompileInfo() == nullptr) {
        OP_LOGD(nodeName.c_str(), "GetPlatformInfo return nullptr, need re get later.");
        tilingParams.needGetCompileInfo = true;
    } else {
        auto compileInfo = reinterpret_cast<const AddRmsNormQuantCompileInfo*>(context_->GetCompileInfo());
        tilingParams.totalCoreNum = compileInfo->totalCoreNum;
        tilingParams.maxUbSize = compileInfo->maxUbSize;
        tilingParams.needGetCompileInfo = false;
    }

    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormQuantRegbaseTiling::IsCapable()
{
    return true;
}

/**
 * @brief: Cal base UB total size
 *         totalSize = xCount      * 1 * AlignB512(N*xDtype) +
 *                     tmpBufCount * 1 * AlignB32(AlignB512(N)/(2*VL)) +
 *                     rstdCount   * 1 * AlignB32(M*sizeof(float)) +
 *                     gammaYCount * 1 * AlignB32(N*xDtype) +
 *                     QuantCount  * 1 * AlignB32(N*quantDtype)
 *                     yCount      * 1 * AlignB32(N*yDtype)
 * @param M dim Num
 * @param N dim Num
 * @return Bytes of UBSize
 */
uint64_t AddRmsNormQuantRegbaseTiling::CalUBTotalSize(
    uint64_t baseM, uint64_t baseN, const uint32_t tilingType = TILING_TYPE_NORMAL)
{
    uint64_t baseMB32Align = Ops::Base::CeilAlign(baseM, static_cast<uint64_t>(B32_BLOCK_NUM));
    uint64_t baseNB8Align = Ops::Base::CeilAlign(baseN, static_cast<uint64_t>(BLOCK_SIZE));
    uint64_t baseNReduceAlign = Ops::Base::CeilAlign(baseN, tilingParams.xReduceAlignNum);
    uint64_t baseNDtypeAlign = Ops::Base::CeilAlign(baseN, tilingParams.xDtypeAlignNum);
    uint64_t baseNQuantAlign = Ops::Base::CeilAlign(baseN, tilingParams.quantDtypeAlignNum);
    uint64_t reduceBufLen = baseNReduceAlign / (2 * tilingParams.vecLength);
    uint64_t reduceBufLenAlign = Ops::Base::CeilAlign(reduceBufLen, static_cast<uint64_t>(B32_BLOCK_NUM));

    uint64_t totalSize =
        3 * baseNReduceAlign * tilingParams.xDtypeSize +                                 // x1/x2/xout
        1 * reduceBufLenAlign * sizeof(float) +                                          // reduceBuf
        1 * baseNDtypeAlign * tilingParams.xDtypeSize +                                  // gamma
        (1 + tilingParams.quantBufCnt) * baseNQuantAlign * tilingParams.quantDtypeSize + // scales/zeropoints
        1 * baseNB8Align * sizeof(int8_t);                                               // y1

    if (tilingParams.hasY2) {
        totalSize += 1 * baseNB8Align * sizeof(int8_t); // y2
    }
    if (TILING_TYPE_NORMAL == tilingType) {
        totalSize += 1 * baseMB32Align * sizeof(float); // rstd
    } else {
        totalSize += 1 * B32_BLOCK_NUM * sizeof(float);                   // rstd
        totalSize += LEVEL_BUFFER_CNT * ONCE_VECTOR_SIZE * sizeof(float); // levelbuf
        totalSize += 1 * tilingParams.vecLength * sizeof(float);          // tempBuf
    }

    return totalSize;
}

ge::graphStatus AddRmsNormQuantRegbaseTiling::SetTilingParams()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling SetTilingParams.");
    uint64_t tmpUBSize;
    tilingParams.powerLoop = 1;

    // 1. No cut
    tmpUBSize = CalUBTotalSize(tilingParams.numM, tilingParams.numN);
    if (tmpUBSize < tilingParams.maxUbSize) {
        tilingParams.baseM = tilingParams.numM;
        tilingParams.baseN = tilingParams.numN;
        tilingParams.tilingType = TILING_TYPE_NORMAL;
        return ge::GRAPH_SUCCESS;
    }

    // 2. Cut m
    tmpUBSize = CalUBTotalSize(1, tilingParams.numN);
    if (tmpUBSize <= tilingParams.maxUbSize) {
        tilingParams.baseN = tilingParams.numN;
        uint64_t justNUBSize = CalUBTotalSize(0, tilingParams.baseN);
        uint64_t rstdRemainUBSize = BLOCK_SIZE;
        uint64_t rstdCount = 1; // rstd
        // Note: CalUBTotalSize(M, N) can be see as:
        //       CalUBTotalSize(M, N) = rstdCount*AlignB32(M) + b*Align1(N) + c*M*Align2(N)
        //       CalUBTotalSize(1, N) = rstdCount*BLOCK_SIZE + b*Align1(N) + c*Align2(N)
        //       CalUBTotalSize(0, N) = b*Align1(N)
        // Note:
        //       rstdCount*M*sizeof(float) + b*Align1(N) + c*M*Align2(N) ~= UBSize - rstdCount*BLOCK_SIZE
        //       baseM ~= (UBSize - rstdCount*BLOCK_SIZE - b*Align1(N)) / (c*Align2(N) + rstdCount*sizeof(float))
        //       baseM ~= (UBSize - rstdCount*BLOCK_SIZE - CalUBTotalSize(1, N)) /
        //                (CalUBTotalSize(1, N) - rstdCount*BLOCK_SIZE - CalUBTotalSize(0, N) + rstdCount*sizeof(float))
        tilingParams.baseM = 1;
        if (rstdRemainUBSize + justNUBSize <= tilingParams.maxUbSize) {
            tilingParams.baseM = (tilingParams.maxUbSize - rstdRemainUBSize - justNUBSize) /
                                 (tmpUBSize - rstdRemainUBSize - justNUBSize + rstdCount * sizeof(float));
        }

        tilingParams.tilingType = TILING_TYPE_NORMAL;
        return ge::GRAPH_SUCCESS;
    }

    // 3. Cut n
    tmpUBSize = CalUBTotalSize(1, tilingParams.xReduceAlignNum, TILING_TYPE_SPILT);
    if (tmpUBSize <= tilingParams.maxUbSize) {
        uint64_t tmpPower = tilingParams.xReduceAlignNum;
        while (CalUBTotalSize(1, tmpPower * MULTI_FACTOR_2, TILING_TYPE_SPILT) <= tilingParams.maxUbSize) {
            tmpPower *= MULTI_FACTOR_2;
        }
        tilingParams.powerSplit = tmpPower;
        uint64_t tmpLoop = 1;
        while (tmpLoop * MULTI_FACTOR_2 * tilingParams.powerSplit <= tilingParams.numN) {
            tmpLoop *= MULTI_FACTOR_2;
        }
        tilingParams.powerLoop = tmpLoop;
        tilingParams.baseM = 1;
        tilingParams.baseN = tilingParams.powerSplit;
        tilingParams.tilingType = TILING_TYPE_SPILT;
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(nodeName.c_str(), "Can not find one tiling.");
    return ge::GRAPH_FAILED;
}

ge::graphStatus AddRmsNormQuantRegbaseTiling::DoOpTiling()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormQuantRegbaseTiling DoOpTiling.");
    if (tilingParams.needGetCompileInfo) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tilingParams.maxUbSize);
        tilingParams.totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    }

    tilingParams.mPerCore = Ops::Base::CeilDiv(tilingParams.numM, tilingParams.totalCoreNum);
    tilingParams.usedCoreNum = Ops::Base::CeilDiv(tilingParams.numM, tilingParams.mPerCore);
    tilingParams.mLastCore = tilingParams.numM - (tilingParams.usedCoreNum - 1) * tilingParams.mPerCore;

    ge::graphStatus res = SetTilingParams();
    OP_CHECK_IF(ge::GRAPH_SUCCESS != res, , return res);

    // Set align params
    tilingParams.baseNDtypeAlign = Ops::Base::CeilAlign(tilingParams.baseN, tilingParams.xDtypeAlignNum);
    tilingParams.baseNQuantAlign = Ops::Base::CeilAlign(tilingParams.baseN, tilingParams.quantDtypeAlignNum);
    tilingParams.baseNB8Align = Ops::Base::CeilAlign(tilingParams.baseN, static_cast<uint64_t>(BLOCK_SIZE));
    tilingParams.baseNReduceAlign = Ops::Base::CeilAlign(tilingParams.baseN, tilingParams.xReduceAlignNum);
    uint64_t reduceBufLen = tilingParams.baseNReduceAlign / (2 * tilingParams.vecLength);
    tilingParams.reduceBufLenAlign = Ops::Base::CeilAlign(reduceBufLen, static_cast<uint64_t>(B32_BLOCK_NUM));

    if (TILING_TYPE_NORMAL == tilingParams.tilingType) {
        uint64_t tmpPower = std::floor(std::log(tilingParams.baseNReduceAlign) / std::log(LOG_2));
        tilingParams.powerSplit = std::pow(LOG_2, tmpPower);
    }

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormQuantRegbaseTiling::SetTilingData()
{
    tilingData.set_numM(tilingParams.numM);
    tilingData.set_numN(tilingParams.numN);
    tilingData.set_baseM(tilingParams.baseM);
    tilingData.set_baseN(tilingParams.baseN);
    tilingData.set_baseNDtypeAlign(tilingParams.baseNDtypeAlign);
    tilingData.set_baseNReduceAlign(tilingParams.baseNReduceAlign);
    tilingData.set_powerSplit(tilingParams.powerSplit);
    tilingData.set_powerLoop(tilingParams.powerLoop);
    tilingData.set_mPerCore(tilingParams.mPerCore);
    tilingData.set_mLastCore(tilingParams.mLastCore);
    tilingData.set_epsilon(tilingParams.epsilon);
    tilingData.set_avgFactor(tilingParams.avgFactor);
}

void AddRmsNormQuantRegbaseTiling::PrintTilingData()
{
    OP_LOGI(
        nodeName.c_str(),
        "TilingData numM: %lu, numN: %lu, baseM: %lu, baseN: %lu, "
        "baseNDtypeAlign: %lu, baseNReduceAlign: %lu, powerSplit: %lu, powerLoop: %lu, "
        "mPerCore: %lu, mLastCore: %lu, "
        "epsilon: %f, avgFactor: %f.",
        tilingData.get_numM(), tilingData.get_numN(), tilingData.get_baseM(), tilingData.get_baseN(),
        tilingData.get_baseNDtypeAlign(), tilingData.get_baseNReduceAlign(), tilingData.get_powerSplit(),
        tilingData.get_powerLoop(), tilingData.get_mPerCore(), tilingData.get_mLastCore(), tilingData.get_epsilon(),
        tilingData.get_avgFactor());
}

ge::graphStatus AddRmsNormQuantRegbaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormQuantRegbaseTiling::GetWorkspaceSize()
{
    tilingParams.workspaceSize = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormQuantRegbaseTiling::PostTiling()
{
    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t usrWorkspaceSize = tilingParams.workspaceSize;
    size_t sysWorkSpaceSize = DEFAULT_SYS_WORKSPACE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormQuantRegbaseTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILING_OFFSET_REGBASE;
    tilingKey += tilingParams.tilingType;
    tilingKey +=
        TILING_OFFSET_HAS_QUANT * ((tilingParams.hasZeroPoints1 << ZERO_POINTS1_BIN_OFFSET) |
                                   (tilingParams.hasScales2 << SCALES2_BIN_OFFSET) | tilingParams.hasZeroPoints2);
    if (tilingParams.divMode) {
        tilingKey += TILING_OFFSET_DIV_MODE;
    }
    OP_LOGI(nodeName.c_str(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}

} // namespace optiling
