/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file add_rms_norm_cast_regbase_tiling.cpp
 * \brief
 */

#include "add_rms_norm_cast_tiling.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

constexpr uint64_t X1_INDEX = 0;
constexpr uint64_t X2_INDEX = 1;
constexpr uint64_t GAMMA_INDEX = 2;
constexpr uint64_t Y1_INDEX = 0;
constexpr uint64_t Y2_INDEX = 1;
constexpr uint64_t RSTD_INDEX = 2;
constexpr uint64_t X_INDEX = 3;
constexpr uint64_t EPS_ATTR_INDEX = 0;

constexpr uint32_t LOG_2 = 2;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t MAX_DIM_CNT = 8;
constexpr uint32_t WORKSPACE_COUNT = 1;
constexpr uint32_t B32_BLOCK_NUM = 8;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t ALING_FACTOR_512 = 512;
constexpr uint32_t ONCE_VECTOR_SIZE = 256;
constexpr uint32_t DEFAULT_SYS_WORKSPACE = 16 * 1024 * 1024;

constexpr uint32_t LEVEL_BUFFER_CNT = 3;
constexpr uint32_t MULTI_FACTOR_2 = 2;
constexpr uint32_t SMOOTH_SCALE1_BIN_OFFSET = 1;
constexpr uint32_t NDDMA_BETTER_STAGE = 512;

ge::graphStatus AddRmsNormCastRegbaseTiling::CheckDtypeVaild(
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

bool AddRmsNormCastRegbaseTiling::CheckShapeNull()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling CheckShapeNull.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
    const gert::StorageShape* y1Shape = context_->GetOutputShape(Y1_INDEX);
    const gert::StorageShape* y2Shape = context_->GetOutputShape(Y2_INDEX);
    const gert::StorageShape* rstdShape = context_->GetOutputShape(RSTD_INDEX);
    const gert::StorageShape* xShape = context_->GetOutputShape(X_INDEX);

    OP_CHECK_IF(
        (nullptr == x1Shape) || (nullptr == x2Shape) || (nullptr == gammaShape) || (nullptr == y1Shape) ||
            (nullptr == y2Shape) || (nullptr == rstdShape) || (nullptr == xShape),
        , return false);
    return true;
}

bool AddRmsNormCastRegbaseTiling::CheckInputShapeDim()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling CheckInputShapeDim.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);

    // Not support zero shape.
    size_t x1DimNum = x1Shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2Shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (x1DimNum > MAX_DIM_CNT) || (x2DimNum > MAX_DIM_CNT),
        OP_LOGE(nodeName.c_str(), "Input x1/x2 dim should not bigger than %u.", MAX_DIM_CNT), return false);
    OP_CHECK_IF(!CheckDimBiggerZero(x1Shape, x1DimNum, nodeName, "x1"), , return false);
    OP_CHECK_IF(!CheckDimBiggerZero(x2Shape, x2DimNum, nodeName, "x2"), , return false);
    return true;
}

bool AddRmsNormCastRegbaseTiling::CheckInputShapeValue()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling CheckInputShapeValue.");
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gammaShape = context_->GetInputShape(GAMMA_INDEX);
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
    if (!NormCheck::CheckShapeSame(x1Shape, y2Shape, nodeName, "x1", "y2")) {
        return false;
    };
    if (!NormCheck::CheckShapeSame(x1Shape, xShape, nodeName, "x1", "x")) {
        return false;
    };

    // Check gamma should be last few dim of x
    if (!NormCheck::CheckShapeBC(x1Shape, gammaShape, nodeName, "x1", "gamma")) {
        return false;
    };

    return true;
}

bool AddRmsNormCastRegbaseTiling::CheckInputDtype()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling CheckInputDtype.");
    std::vector<ge::DataType> supportedXGammaDtypes = {
        ge::DataType::DT_FLOAT16, ge::DataType::DT_BF16, ge::DataType::DT_FLOAT};

    ge::DataType x1Dtype = context_->GetInputTensor(X1_INDEX)->GetDataType();
    ge::DataType x2Dtype = context_->GetInputTensor(X2_INDEX)->GetDataType();
    ge::DataType gammaDtype = context_->GetInputTensor(GAMMA_INDEX)->GetDataType();
    if ((x1Dtype != x2Dtype) || (x1Dtype != gammaDtype)) {
        OP_LOGE(nodeName.c_str(), "Input x1/x2/gamma dtype should be equal.");
        return false;
    }

    return true;
}

ge::graphStatus AddRmsNormCastRegbaseTiling::SetInputParams()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling SetInputParams.");
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
    auto xDataType = context_->GetInputTensor(X1_INDEX)->GetDataType();
    tilingParams.xDtypeSize = GetSizeByDataType(xDataType);
    tilingParams.xDtypeAlignNum = BLOCK_SIZE / tilingParams.xDtypeSize;
    tilingParams.xReduceAlignNum = ALING_FACTOR_512 / tilingParams.xDtypeSize;
    tilingParams.vecLength = Ops::Base::GetVRegSize(context_) / sizeof(float);

    // Set input attr
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const float* epsilon = attrs->GetFloat(EPS_ATTR_INDEX);
    tilingParams.epsilon = *epsilon;
    tilingParams.needRun = true;
    if ((0 == numN) || (0 == numM)) {
        tilingParams.needRun = false;
    }
    tilingParams.avgFactor = (0 == numN) ? 0.0f : 1.0f / static_cast<float>(numN);
    tilingParams.isNddma = numN >= NDDMA_BETTER_STAGE ? 0 : 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormCastRegbaseTiling::GetShapeAttrsInfo()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling GetShapeAttrsInfo.");
    OP_CHECK_IF(
        !CheckShapeNull(), OP_LOGE(nodeName.c_str(), "The not optional input is null."), return ge::GRAPH_FAILED);
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

ge::graphStatus AddRmsNormCastRegbaseTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling GetPlatformInfo.");
    if (context_->GetCompileInfo() == nullptr) {
        OP_LOGD(nodeName.c_str(), "GetPlatformInfo return nullptr, need re get later.");
        tilingParams.needGetCompileInfo = true;
    } else {
        auto compileInfo = reinterpret_cast<const AddRmsNormCastCompileInfo*>(context_->GetCompileInfo());
        tilingParams.totalCoreNum = compileInfo->totalCoreNum;
        tilingParams.maxUbSize = compileInfo->totalUbSize;
        tilingParams.needGetCompileInfo = false;
    }
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormCastRegbaseTiling::IsCapable()
{
    return true;
}

/* *
 * @brief: Cal base UB total size
 * @param M dim Num
 * @param N dim Num
 * @return Bytes of UBSize
 */
uint64_t AddRmsNormCastRegbaseTiling::CalUBTotalSize(
    uint64_t baseM, uint64_t baseN, const uint32_t tilingType = TILING_TYPE_NORMAL)
{
    uint64_t baseMB32Align = Ops::Base::CeilAlign(baseM, static_cast<uint64_t>(B32_BLOCK_NUM));
    uint64_t baseNReduceAlign = Ops::Base::CeilAlign(baseN, tilingParams.xReduceAlignNum);
    uint64_t baseNDtypeAlign = Ops::Base::CeilAlign(baseN, tilingParams.xDtypeAlignNum);
    uint64_t reduceSumBufLen = baseNReduceAlign / (2 * tilingParams.vecLength);
    uint64_t reduceSumBufLenAlign = Ops::Base::CeilAlign(reduceSumBufLen, static_cast<uint64_t>(B32_BLOCK_NUM));

    uint64_t totalSize;

    if (TILING_TYPE_NORMAL == tilingType) {
        totalSize = LEVEL_BUFFER_CNT * baseM * baseNReduceAlign * tilingParams.xDtypeSize + // x1/x2/xout
                    1 * baseM * baseNReduceAlign * sizeof(float) +           // xoutTmp(alloc bigger than use)
                    1 * reduceSumBufLenAlign * sizeof(float) +               // reduceBuf
                    1 * baseNReduceAlign * tilingParams.xDtypeSize +         // gamma(alloc bigger than use)
                    1 * baseM * baseNReduceAlign * sizeof(float) +           // y1(alloc bigger than use)
                    1 * baseM * baseNReduceAlign * tilingParams.xDtypeSize + // y2(alloc bigger than use)
                    1 * baseMB32Align * sizeof(float);                       // rstd
    } else {
        totalSize = LEVEL_BUFFER_CNT * baseNReduceAlign * tilingParams.xDtypeSize + // x1/x2/xout
                    1 * baseNReduceAlign * sizeof(float) +                          // xoutTmp(alloc bigger than use)
                    1 * reduceSumBufLenAlign * sizeof(float) +                      // reduceBuf
                    1 * baseNDtypeAlign * tilingParams.xDtypeSize +                 // gamma
                    1 * baseNDtypeAlign * sizeof(float) +                           // y1(alloc bigger than use)
                    1 * baseNDtypeAlign * tilingParams.xDtypeSize +                 // y2
                    1 * B32_BLOCK_NUM * sizeof(float) +                             // rstd
                    LEVEL_BUFFER_CNT * ONCE_VECTOR_SIZE * sizeof(float) +           // levelbuf
                    1 * tilingParams.vecLength * sizeof(float);                     // tempBuf
    }

    return totalSize;
}

ge::graphStatus AddRmsNormCastRegbaseTiling::SetTilingParams()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling SetTilingParams.");
    uint64_t tmpUBSize;
    tilingParams.powerLoop = 1;

    // 1. No cut
    tmpUBSize = CalUBTotalSize(tilingParams.mPerCore, tilingParams.numN);
    if (tmpUBSize < tilingParams.maxUbSize) {
        tilingParams.baseM = tilingParams.mPerCore;
        tilingParams.baseN = tilingParams.numN;
        tilingParams.tilingType = (tilingParams.mPerCore > 1) ? TILING_TYPE_NORMAL : TILING_TYPE_MAGIC_AND_WONDERFUL;
        return ge::GRAPH_SUCCESS;
    }

    // 2. Cut m
    tmpUBSize = CalUBTotalSize(1, tilingParams.numN);
    if (tmpUBSize <= tilingParams.maxUbSize) {
        tilingParams.baseN = tilingParams.numN;
        uint64_t justNUBSize = CalUBTotalSize(0, tilingParams.baseN);
        uint64_t rstdCount = 1; // rstd
        uint64_t rstdRemainUBSize = rstdCount * BLOCK_SIZE;
        // Note: CalUBTotalSize(M, N) can be see as:
        //       CalUBTotalSize(M, N) = rstdCount*AlignB32(M) + b*Align1(N) + c*M*Align2(N)
        //       CalUBTotalSize(1, N) = rstdCount*BLOCK_SIZE + b*Align1(N) + c*Align2(N)
        //       CalUBTotalSize(0, N) = b*Align1(N)
        // Note:
        //       rstdCount*M*sizeof(float) + b*Align1(N) + c*M*Align2(N) ~= UBSize - rstdCount*BLOCK_SIZE
        //       baseM ~= (UBSize - rstdCount*BLOCK_SIZE - b*Align1(N)) / (c*Align2(N) + rstdCount*sizeof(float))
        //       baseM ~= (UBSize - rstdCount*BLOCK_SIZE - CalUBTotalSize(0, N)) /
        //                (CalUBTotalSize(1, N) - rstdCount*BLOCK_SIZE - CalUBTotalSize(0, N) + rstdCount*sizeof(float))
        tilingParams.baseM = 1;
        if (rstdRemainUBSize + justNUBSize <= tilingParams.maxUbSize) {
            tilingParams.baseM = (tilingParams.maxUbSize - rstdRemainUBSize - justNUBSize) /
                                 (tmpUBSize - rstdRemainUBSize - justNUBSize + rstdCount * sizeof(float));
        }
        tilingParams.tilingType = (tilingParams.mPerCore > 1) ? TILING_TYPE_NORMAL : TILING_TYPE_MAGIC_AND_WONDERFUL;
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

ge::graphStatus AddRmsNormCastRegbaseTiling::DoOpTiling()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormCastRegbaseTiling DoOpTiling.");
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
    tilingParams.baseNReduceAlign = Ops::Base::CeilAlign(tilingParams.baseN, tilingParams.xReduceAlignNum);

    if (TILING_TYPE_NORMAL == tilingParams.tilingType || TILING_TYPE_MAGIC_AND_WONDERFUL == tilingParams.tilingType) {
        uint64_t tmpPower = std::floor(std::log(tilingParams.baseNReduceAlign) / std::log(LOG_2));
        tilingParams.powerSplit = std::pow(LOG_2, tmpPower);
    }

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormCastRegbaseTiling::SetTilingData()
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
    tilingData.set_isNddma(tilingParams.isNddma);
}

void AddRmsNormCastRegbaseTiling::PrintTilingData()
{
    OP_LOGI(
        nodeName.c_str(),
        "TilingData numM: %lu, numN: %lu, baseM: %lu, baseN: %lu, "
        "baseNDtypeAlign: %lu, baseNReduceAlign: %lu, powerSplit: %lu, powerLoop: %lu, "
        "mPerCore: %lu, mLastCore: %lu, "
        "epsilon: %f, avgFactor: %f, isNddma: %u.",
        tilingData.get_numM(), tilingData.get_numN(), tilingData.get_baseM(), tilingData.get_baseN(),
        tilingData.get_baseNDtypeAlign(), tilingData.get_baseNReduceAlign(), tilingData.get_powerSplit(),
        tilingData.get_powerLoop(), tilingData.get_mPerCore(), tilingData.get_mLastCore(), tilingData.get_epsilon(),
        tilingData.get_avgFactor(), tilingData.get_isNddma());
}

ge::graphStatus AddRmsNormCastRegbaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormCastRegbaseTiling::GetWorkspaceSize()
{
    tilingParams.workspaceSize = 0;
    if (TILING_TYPE_SPILT == tilingParams.tilingType) {
        tilingParams.workspaceSize = WORKSPACE_COUNT * tilingParams.usedCoreNum * tilingParams.numN * sizeof(float);
    } else if (TILING_TYPE_MAGIC_AND_WONDERFUL == tilingParams.tilingType) {
        tilingParams.workspaceSize = tilingParams.usedCoreNum * MIGIC_AND_WONDERFUL_BYTES;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormCastRegbaseTiling::PostTiling()
{
    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    OP_LOGW(nodeName.c_str(), "Tiling usrWorkspaceSize is %lu.", tilingParams.workspaceSize);

    size_t usrWorkspaceSize = tilingParams.workspaceSize;
    size_t sysWorkSpaceSize = DEFAULT_SYS_WORKSPACE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormCastRegbaseTiling::GetTilingKey() const
{
    uint64_t tilingKey = TILING_OFFSET_REGBASE;
    tilingKey += tilingParams.tilingType;

    if (!tilingParams.needRun) {
        tilingKey = TILING_KEY_UNRUN;
    }
    OP_LOGI(nodeName.c_str(), "TilingKey is %lu.", tilingKey);
    return tilingKey;
}
} // namespace optiling
