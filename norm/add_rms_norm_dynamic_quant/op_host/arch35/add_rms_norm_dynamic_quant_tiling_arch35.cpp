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
 * \file add_rms_norm_dynamic_quant_tiling_arch35.cpp
 * \brief
 */

#include "add_rms_norm_dynamic_quant_tiling_arch35.h"
#include "norm/norm_common/op_host/norm_tiling_check_common.h"

namespace optiling {
using namespace NormCheck;

constexpr uint64_t EPS_ATTR_INDEX = 0;
constexpr uint32_t LOG_2 = 2;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t CONST_FOUR = 4;
constexpr uint32_t CONST_EIGHT = 8;
constexpr uint32_t CONST_SIXTEEN = 16;
constexpr uint32_t CONST_THIRTY_TWO = 32;
constexpr uint32_t WORKSPACE_COUNT = 3;
constexpr uint32_t ALIGN_FACTOR_512 = 512;
constexpr uint32_t DEFAULT_SYS_WORKSPACE = 16 * 1024 * 1024;
// NormCommon::LevelMergeRstd folds 4 sub-vectors per pass, one fp32 vreg each, so one level
// buffer holds 4 * VL_FP32 elements. Must stay equal to the kernel side
// RmsNorm::ONCE_VECTOR_SIZE used to allocate level1/2/3Buf_.
constexpr uint32_t LEVEL_MERGE_PARALLEL = 4;

constexpr uint32_t LEVEL_BUFFER_CNT = 3;
constexpr uint32_t MULTI_FACTOR_2 = 2;
constexpr uint32_t ALIGN_SPACE = 1 * 1024;
constexpr uint32_t DOUBLE_BUFFER = 2;
// 按 baseN 计量的 x 侧 buffer 个数：x1、x2、xout
constexpr uint32_t X_BUF_CNT = 3;
// 按行计量的标量 buffer 个数：rstd、xReduceTmp
constexpr uint32_t RSTD_TMP_BUF_CNT = 2;
// NormCommon 的 reduce 以 2 个 fp32 vreg 为一个 repeat（见 norm_common/op_kernel/
// reduce_common_regbase.h 中 remainRepeats / masterRepeats 的算法），reduceBuf 定长须同口径
constexpr uint32_t REDUCE_VREG_PER_REPEAT = 2;
// SingleRow 模板每个元素占用的 UB 字节数，等于 add_rms_norm_dynamic_quant_regbase_single_row.h
// 中 Init 分配的各 buffer 之和：inRowsQue 2*sizeof(T_X) + yQue sizeof(T_X) + xBufFp32 4 +
// yBufFp32 4 + smoothBuf sizeof(T_X)，T_X 为 b16 时合计 16
constexpr uint32_t SINGLE_ROW_UB_BYTES_PER_ELEM = 16;

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::SetInputParams()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling SetInputParams.");
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
    tilingParams.xDtypeSize = GetSizeByDataType(xDataType);
    // Set platform derived params
    tilingParams.vecLength = Ops::Base::GetVRegSize(context_) / sizeof(float);
    tilingParams.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    tilingParams.b32BlockNum = tilingParams.ubBlockSize / sizeof(float);
    tilingParams.b8BlockNum = tilingParams.ubBlockSize / sizeof(int8_t);
    tilingParams.xDtypeAlignNum = tilingParams.ubBlockSize / tilingParams.xDtypeSize;
    tilingParams.xReduceAlignNum = ALIGN_FACTOR_512 / tilingParams.xDtypeSize;

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
    // Sync output flags from base class member variables (set by CheckInputAttr)
    tilingParams.hasSmoothScale1 = hasSmoothScale1_;
    tilingParams.hasSmoothScale2 = hasSmoothScale2_;
    tilingParams.hasBeta = hasBeta_;
    tilingParams.quantBufCnt = (hasSmoothScale1_ ? 1 : 0) + (hasSmoothScale2_ ? 1 : 0);
    tilingParams.outQuant1Flag = outQuant1Flag_;
    tilingParams.outQuant2Flag = outQuant2Flag_;
    tilingParams.hasY3 = hasY3_;
    tilingParams.hasY4 = hasY4_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling GetPlatformInfo.");
    if (context_->GetCompileInfo() == nullptr) {
        OP_LOGD(nodeName.c_str(), "GetPlatformInfo return nullptr, need re get later.");
        tilingParams.needGetCompileInfo = true;
    } else {
        auto compileInfo = static_cast<const AddRmsNormDynamicQuantCompileInfo*>(context_->GetCompileInfo());
        tilingParams.totalCoreNum = compileInfo->totalCoreNum;
        tilingParams.maxUbSize = compileInfo->maxUbSize;
        tilingParams.needGetCompileInfo = false;
    }
    return ge::GRAPH_SUCCESS;
}

bool AddRmsNormDynamicQuantRegbaseTiling::IsCapable() { return true; }

/**
 * @brief: Cal base UB total size
 * @param M dim Num
 * @param N dim Num
 * @return Bytes of UBSize
 */
uint64_t AddRmsNormDynamicQuantRegbaseTiling::CalUBTotalSize(uint64_t baseM, uint64_t baseN, const uint32_t tilingType)
{
    uint64_t baseMB32Align = Ops::Base::CeilAlign(baseM, tilingParams.b32BlockNum);
    uint64_t baseNB8Align = Ops::Base::CeilAlign(baseN, tilingParams.b8BlockNum);
    uint64_t baseNReduceAlign = Ops::Base::CeilAlign(baseN, tilingParams.xReduceAlignNum);
    uint64_t baseNDtypeAlign = Ops::Base::CeilAlign(baseN, tilingParams.xDtypeAlignNum);
    uint64_t baseNB32Align = Ops::Base::CeilAlign(baseN, tilingParams.b32BlockNum);
    uint64_t reduceSumBufLen = baseNReduceAlign / (REDUCE_VREG_PER_REPEAT * tilingParams.vecLength);
    uint64_t reduceSumBufLenAlign = Ops::Base::CeilAlign(reduceSumBufLen, tilingParams.b32BlockNum);

    uint64_t totalSize = X_BUF_CNT * baseNReduceAlign * tilingParams.xDtypeSize + // x1/x2/xout
                         1 * baseNReduceAlign * sizeof(float) +                   // xoutTmp(alloc bigger than use)
                         1 * reduceSumBufLenAlign * sizeof(float) +               // reduceBuf
                         1 * baseNDtypeAlign * tilingParams.xDtypeSize +          // gamma
                         tilingParams.quantBufCnt * baseNDtypeAlign *
                             tilingParams.xDtypeSize; // smoothScale1/smoothScale2

    if (tilingParams.outQuant1Flag) {
        totalSize += 1 * baseNB8Align * sizeof(int8_t) + // y1
                     1 * baseNB32Align * sizeof(float);  // y1Tmp
    }

    if (tilingParams.hasBeta) {
        totalSize += 1 * baseNDtypeAlign * tilingParams.xDtypeSize; // beta
    }

    if (tilingParams.outQuant2Flag) {
        totalSize += 1 * baseNB8Align * sizeof(int8_t) + // y2
                     1 * baseNB32Align * sizeof(float);  // y2Tmp
    }

    if (tilingParams.hasY3) {
        totalSize += baseNB32Align * sizeof(float); // y3
    }

    if (tilingParams.hasY4) {
        totalSize += baseNDtypeAlign * tilingParams.xDtypeSize; // y4
    }

    if (TILING_TYPE_NORMAL == tilingType) {
        totalSize += 1 * baseMB32Align * sizeof(float); // rstd
        if (tilingParams.outQuant1Flag) {
            totalSize += 1 * baseMB32Align * sizeof(float); // scale1
        }
        if (tilingParams.outQuant2Flag) {
            totalSize += 1 * baseMB32Align * sizeof(float); // scale2
        }
    } else {
        totalSize += 1 * tilingParams.b32BlockNum * sizeof(float); // rstd
        if (tilingParams.outQuant1Flag) {
            totalSize += 1 * tilingParams.b32BlockNum * sizeof(float); // scale1
        }
        if (tilingParams.outQuant2Flag) {
            totalSize += 1 * tilingParams.b32BlockNum * sizeof(float); // scale2
        }
        uint64_t onceVectorSize = LEVEL_MERGE_PARALLEL * tilingParams.vecLength;
        totalSize += LEVEL_BUFFER_CNT * onceVectorSize * sizeof(float); // levelbuf
        totalSize += 1 * tilingParams.vecLength * sizeof(float);        // tempBuf
    }

    return totalSize;
}

int64_t AddRmsNormDynamicQuantRegbaseTiling::CalFullLoadBaseM(uint64_t baseN, int64_t& tmpPower)
{
    uint64_t baseNB8Align = Ops::Base::CeilAlign(baseN, tilingParams.b8BlockNum);
    uint64_t baseNB32Align = Ops::Base::CeilAlign(baseN, tilingParams.b32BlockNum);
    uint64_t baseNDtypeAlign = Ops::Base::CeilAlign(baseN, tilingParams.xDtypeAlignNum);
    tmpPower = std::floor(std::log(baseNDtypeAlign - 1) / std::log(LOG_2));
    tmpPower = std::pow(LOG_2, tmpPower);
    int64_t blockSize = Ops::Base::GetUbBlockSize(context_);
    int64_t vectorLength = Ops::Base::GetVRegSize(context_) / sizeof(float);
    int64_t firstVcaddLength = Ops::Base::CeilDiv(Ops::Base::CeilDiv(tmpPower, vectorLength), blockSize) * blockSize;
    int64_t LastUbSize = tilingParams.maxUbSize - baseNDtypeAlign * tilingParams.xDtypeSize - // gamma
                         tilingParams.quantBufCnt * baseNDtypeAlign *
                             tilingParams.xDtypeSize - // smoothScale1/smoothScale2
                         ALIGN_SPACE;                  // Scale1/rstd/xReduceTmp align space
    if (tilingParams.hasBeta) {
        LastUbSize -= baseNDtypeAlign * tilingParams.xDtypeSize; // beta
    }
    int64_t mutilBaseM = X_BUF_CNT * baseNDtypeAlign * tilingParams.xDtypeSize + // x1/x2/xout
                         baseNDtypeAlign * sizeof(float) +                       // xoutTmp
                         RSTD_TMP_BUF_CNT * sizeof(float) +                      // rstd/xReduceTmp
                         firstVcaddLength * sizeof(float) +                      // xTmp
                         baseNB32Align * sizeof(float);                          // yTmp

    if (tilingParams.outQuant1Flag) {
        mutilBaseM += DOUBLE_BUFFER * baseNB8Align * sizeof(int8_t) + // y1 * double
                      DOUBLE_BUFFER * sizeof(float);                  // Scale1 * double
    }

    if (tilingParams.outQuant2Flag) {
        mutilBaseM += DOUBLE_BUFFER * baseNB8Align * sizeof(int8_t) + // y2 * double
                      DOUBLE_BUFFER * sizeof(float);                  // Scale2 * double
    }

    if (tilingParams.hasY3) {
        mutilBaseM += DOUBLE_BUFFER * baseNB32Align * sizeof(float); // y3 * double
    }

    if (tilingParams.hasY4) {
        mutilBaseM += DOUBLE_BUFFER * baseNDtypeAlign * tilingParams.xDtypeSize; // y4 * double
    }
    int64_t fullLoadBaseM = LastUbSize / mutilBaseM;
    uint64_t usedUbSize = CalUsedSize(fullLoadBaseM, baseNB8Align, baseNB32Align, baseNDtypeAlign, firstVcaddLength);
    while (usedUbSize > tilingParams.maxUbSize && fullLoadBaseM > 0) {
        fullLoadBaseM--;
        usedUbSize = CalUsedSize(fullLoadBaseM, baseNB8Align, baseNB32Align, baseNDtypeAlign, firstVcaddLength);
    }
    return fullLoadBaseM;
}

uint64_t AddRmsNormDynamicQuantRegbaseTiling::CalUsedSize(uint64_t baseM, uint64_t baseNB8Align, uint64_t baseNB32Align,
                                                          uint64_t baseNDtypeAlign, int64_t firstVcaddLength)
{
    uint64_t ubFactorRstd = Ops::Base::CeilAlign(baseM, tilingParams.b32BlockNum);
    uint64_t totalSize = 0;
    totalSize += baseNDtypeAlign * tilingParams.xDtypeSize +
                 tilingParams.quantBufCnt * baseNDtypeAlign * tilingParams.xDtypeSize + ALIGN_SPACE;
    totalSize += baseM * (X_BUF_CNT * baseNDtypeAlign * tilingParams.xDtypeSize +
                          baseNDtypeAlign * sizeof(float)); // x1/x2/xout + xTmp
    totalSize += baseM * firstVcaddLength * sizeof(float);  // xTmp
    if (tilingParams.outQuant1Flag) {
        totalSize += baseM * (baseNB8Align * sizeof(int8_t) * DOUBLE_BUFFER + // y1 * double
                              baseNB32Align * sizeof(float));                 // y1Tmp
    }
    totalSize += RSTD_TMP_BUF_CNT * ubFactorRstd * sizeof(float); // rstd/xReduceTmp
    if (tilingParams.outQuant1Flag) {
        totalSize += DOUBLE_BUFFER * ubFactorRstd * sizeof(float); // Scale1 * double
    }
    if (tilingParams.outQuant2Flag) {
        totalSize += baseM * (baseNB8Align * sizeof(int8_t) * DOUBLE_BUFFER); // y2 * double
        totalSize += DOUBLE_BUFFER * ubFactorRstd * sizeof(float);            // Scale2 * double
    }
    if (tilingParams.hasBeta) {
        totalSize += baseNDtypeAlign * tilingParams.xDtypeSize; // beta
    }
    if (tilingParams.hasY3) {
        totalSize += baseM * DOUBLE_BUFFER * baseNB32Align * sizeof(float); // y3 * double
    }
    if (tilingParams.hasY4) {
        totalSize += baseM * DOUBLE_BUFFER * baseNDtypeAlign * tilingParams.xDtypeSize; // y4 * double
    }
    return totalSize;
}

static uint64_t GetSingleRowPowerSplit(uint64_t n)
{
    n |= n >> 1;
    n |= n >> CONST_TWO;
    n |= n >> CONST_FOUR;
    n |= n >> CONST_EIGHT;
    n |= n >> CONST_SIXTEEN;
    n |= n >> CONST_THIRTY_TWO;
    return (n + 1) >> 1;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::SetTilingParams()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling SetTilingParams.");
    tilingParams.powerLoop = 1;

    if (TryPerfTiling() || TryNormTiling() || TrySingleRowTiling() || TrySplitTiling()) {
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(nodeName.c_str(), "Can not find one tiling.");
    return ge::GRAPH_FAILED;
}

bool AddRmsNormDynamicQuantRegbaseTiling::TryPerfTiling()
{
    int64_t tmpPower = 0;
    int64_t fullLoadBaseM = CalFullLoadBaseM(tilingParams.numN, tmpPower);
    int64_t vlFp32 = Ops::Base::GetVRegSize(context_) / sizeof(float);
    uint64_t fullLoadRMax = MULTI_FACTOR_2 * vlFp32 * vlFp32 * DOUBLE_BUFFER;
    if (fullLoadBaseM >= 1 && tilingParams.numN <= fullLoadRMax) {
        tilingParams.baseN = tilingParams.numN;
        tilingParams.baseM = std::min(fullLoadBaseM, static_cast<int64_t>(tilingParams.mPerCore));
        tilingParams.powerSplit = tmpPower;
        tilingParams.tilingType = TILING_TYPE_PERF;
        return true;
    }
    return false;
}

bool AddRmsNormDynamicQuantRegbaseTiling::TryNormTiling()
{
    uint64_t tmpUBSize = CalUBTotalSize(1, tilingParams.numN, TILING_TYPE_NORMAL);
    if (tmpUBSize <= tilingParams.maxUbSize) {
        tilingParams.baseN = tilingParams.numN;
        uint64_t justNUBSize = CalUBTotalSize(0, tilingParams.baseN, TILING_TYPE_NORMAL);
        uint64_t rstdCount = 1 + (tilingParams.outQuant1Flag ? 1 : 0) + (tilingParams.outQuant2Flag ? 1 : 0);
        uint64_t rstdRemainUBSize = rstdCount * tilingParams.ubBlockSize;
        tilingParams.baseM = 1;
        if (rstdRemainUBSize + justNUBSize <= tilingParams.maxUbSize) {
            tilingParams.baseM = (tilingParams.maxUbSize - rstdRemainUBSize - justNUBSize) /
                                 (tmpUBSize - rstdRemainUBSize - justNUBSize + rstdCount * sizeof(float));
        }
        tilingParams.tilingType = TILING_TYPE_NORMAL;
        return true;
    }
    return false;
}

bool AddRmsNormDynamicQuantRegbaseTiling::TrySingleRowTiling()
{
    uint64_t yDtypeSize = sizeof(int8_t);
    uint64_t outputAlign = (yDtypeSize > 0) ? (tilingParams.ubBlockSize / yDtypeSize) : tilingParams.ubBlockSize;
    uint64_t D_aligned = Ops::Base::CeilAlign(tilingParams.numN, outputAlign);
    uint64_t singleRowUbSize = SINGLE_ROW_UB_BYTES_PER_ELEM * D_aligned + ALIGN_SPACE;
    if (singleRowUbSize <= tilingParams.maxUbSize) {
        tilingParams.baseN = tilingParams.numN;
        tilingParams.baseM = 1;
        tilingParams.tilingType = TILING_TYPE_SINGLE_ROW;
        tilingParams.powerSplit = GetSingleRowPowerSplit(tilingParams.numN);
        return true;
    }
    return false;
}

bool AddRmsNormDynamicQuantRegbaseTiling::TrySplitTiling()
{
    uint64_t tmpUBSize = CalUBTotalSize(1, tilingParams.xReduceAlignNum, TILING_TYPE_SPILT);
    if (tmpUBSize <= tilingParams.maxUbSize) {
        uint64_t tmpPowerCutN = tilingParams.xReduceAlignNum;
        while (CalUBTotalSize(1, tmpPowerCutN * MULTI_FACTOR_2, TILING_TYPE_SPILT) <= tilingParams.maxUbSize) {
            tmpPowerCutN *= MULTI_FACTOR_2;
        }
        tilingParams.powerSplit = tmpPowerCutN;
        uint64_t curLoop = 1;
        while (curLoop * MULTI_FACTOR_2 * tilingParams.powerSplit <= tilingParams.numN) {
            curLoop *= MULTI_FACTOR_2;
        }
        tilingParams.powerLoop = curLoop;
        tilingParams.baseM = 1;
        tilingParams.baseN = tilingParams.powerSplit;
        tilingParams.tilingType = TILING_TYPE_SPILT;
        return true;
    }
    return false;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::DoOpTiling()
{
    OP_LOGD(nodeName.c_str(), "Enter AddRmsNormDynamicQuantRegbaseTiling DoOpTiling.");
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
    tilingParams.baseNB8Align = Ops::Base::CeilAlign(tilingParams.baseN, tilingParams.b8BlockNum);
    tilingParams.baseNReduceAlign = Ops::Base::CeilAlign(tilingParams.baseN, tilingParams.xReduceAlignNum);
    uint64_t reduceBufLen = tilingParams.baseNReduceAlign / (REDUCE_VREG_PER_REPEAT * tilingParams.vecLength);
    tilingParams.reduceBufLenAlign = Ops::Base::CeilAlign(reduceBufLen, tilingParams.b32BlockNum);

    if (TILING_TYPE_NORMAL == tilingParams.tilingType) {
        uint64_t tmpPower = std::floor(std::log(tilingParams.baseNReduceAlign) / std::log(LOG_2));
        tilingParams.powerSplit = std::pow(LOG_2, tmpPower);
    }

    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AddRmsNormDynamicQuantRegbaseTiling::SetTilingData()
{
    tilingData.numM = tilingParams.numM;
    tilingData.numN = tilingParams.numN;
    tilingData.baseM = tilingParams.baseM;
    tilingData.baseN = tilingParams.baseN;
    tilingData.baseNDtypeAlign = tilingParams.baseNDtypeAlign;
    tilingData.baseNReduceAlign = tilingParams.baseNReduceAlign;
    tilingData.powerSplit = tilingParams.powerSplit;
    tilingData.powerLoop = tilingParams.powerLoop;
    tilingData.mPerCore = tilingParams.mPerCore;
    tilingData.mLastCore = tilingParams.mLastCore;
    tilingData.avgFactor = tilingParams.avgFactor;
    tilingData.epsilon = tilingParams.epsilon;
    tilingData.hasSmoothScale1 = static_cast<uint32_t>(tilingParams.hasSmoothScale1);
    tilingData.hasSmoothScale2 = static_cast<uint32_t>(tilingParams.hasSmoothScale2);
    tilingData.hasBeta = static_cast<uint32_t>(tilingParams.hasBeta);
    tilingData.outQuant1Flag = tilingParams.outQuant1Flag;
    tilingData.outQuant2Flag = tilingParams.outQuant2Flag;
}

void AddRmsNormDynamicQuantRegbaseTiling::PrintTilingData()
{
    OP_LOGI(nodeName.c_str(),
            "TilingData numM: %lu, numN: %lu, baseM: %lu, baseN: %lu, "
            "baseNDtypeAlign: %lu, baseNReduceAlign: %lu, powerSplit: %lu, powerLoop: %lu, "
            "mPerCore: %lu, mLastCore: %lu, "
            "hasS1: %u, hasS2: %u, hasBeta: %u, outQ1: %u, outQ2: %u, epsilon: %f, avgFactor: %f.",
            tilingData.numM, tilingData.numN, tilingData.baseM, tilingData.baseN, tilingData.baseNDtypeAlign,
            tilingData.baseNReduceAlign, tilingData.powerSplit, tilingData.powerLoop, tilingData.mPerCore,
            tilingData.mLastCore, tilingData.hasSmoothScale1, tilingData.hasSmoothScale2, tilingData.hasBeta,
            tilingData.outQuant1Flag, tilingData.outQuant2Flag, tilingData.epsilon, tilingData.avgFactor);
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::GetWorkspaceSize()
{
    tilingParams.workspaceSize = 0;
    if (TILING_TYPE_SPILT == tilingParams.tilingType) {
        tilingParams.workspaceSize = WORKSPACE_COUNT * tilingParams.usedCoreNum * tilingParams.numN * sizeof(float);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddRmsNormDynamicQuantRegbaseTiling::PostTiling()
{
    OP_LOGD(nodeName.c_str(), "Tiling usedCoreNum is %lu.", tilingParams.usedCoreNum);
    context_->SetBlockDim(tilingParams.usedCoreNum);
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(sizeof(tilingData) > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
                        sizeof(tilingData), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&tilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(memcpy_s(ptrData, capSize, ptrStruct, sizeof(tilingData)) != 0,
                OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(tilingData));

    size_t usrWorkspaceSize = tilingParams.workspaceSize;
    size_t sysWorkSpaceSize = DEFAULT_SYS_WORKSPACE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrWorkspaceSize + sysWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

uint64_t AddRmsNormDynamicQuantRegbaseTiling::GetTilingKey() const
{
    Y3Mode y3Mode = tilingParams.hasY3 ? Y3Mode::HAS_Y3 : Y3Mode::NO_Y3;
    Y4Mode y4Mode = tilingParams.hasY4 ? Y4Mode::HAS_Y4 : Y4Mode::NO_Y4;
    if (!tilingParams.needRun) {
        // When numM==0 or numN==0, kernel checks numM==0 and returns early.
        // Return NORMAL mode — any non-EMPTY compute mode is valid here
        // since the kernel will skip processing when numM==0.
        AddRmsNormDynamicQuantTilingKey tilingKey;
        tilingKey.SetComputeMode(ComputeMode::NORMAL, y3Mode, y4Mode);
        return tilingKey.GetTilingKey();
    }
    AddRmsNormDynamicQuantTilingKey tilingKey;
    switch (tilingParams.tilingType) {
        case TILING_TYPE_PERF:
            tilingKey.SetComputeMode(ComputeMode::PERF, y3Mode, y4Mode);
            break;
        case TILING_TYPE_NORMAL:
            tilingKey.SetComputeMode(ComputeMode::NORMAL, y3Mode, y4Mode);
            break;
        case TILING_TYPE_SINGLE_ROW:
            tilingKey.SetComputeMode(ComputeMode::SINGLE_ROW, y3Mode, y4Mode);
            break;
        case TILING_TYPE_SPILT:
            tilingKey.SetComputeMode(ComputeMode::SPLIT, y3Mode, y4Mode);
            break;
        default:
            tilingKey.SetComputeMode(ComputeMode::REDUCE_EMPTY, y3Mode, y4Mode);
            break;
    }
    return tilingKey.GetTilingKey();
}

ge::graphStatus Tiling4AddRmsNormDynamicQuant(gert::TilingContext* context)
{
    OP_TILING_CHECK(nullptr == context, OP_LOGE("AddRmsNormDynamicQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter Tiling4AddRmsNormDynamicQuant (A5)");
    auto colShape = context->GetInputShape(GAMMA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, colShape);
    auto colStorageShape = colShape->GetStorageShape();
    uint32_t col_val = colStorageShape.GetDim(0);
    bool isEmptyTensor = (col_val == 0);
    if (isEmptyTensor) {
        AddRmsNormDynamicQuantEmptyTiling emptyTiling(context);
        return emptyTiling.DoTiling();
    }
    AddRmsNormDynamicQuantRegbaseTiling regbaseTiling(context);
    return regbaseTiling.DoTiling();
}

ge::graphStatus TilingPrepare4AddRmsNormDynamicQuant(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(nullptr == context, OP_LOGE("AddRmsNormDynamicQuant", "Context is null"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare4AddRmsNormDynamicQuant.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "PlatformInfoPtr is null");

    auto compileInfoPtr = context->GetCompiledInfo<AddRmsNormDynamicQuantCompileInfo>();
    OP_LOGE_IF(compileInfoPtr == nullptr, ge::GRAPH_FAILED, context->GetNodeName(), "CompileInfoPtr is null");

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddRmsNormDynamicQuant)
    .Tiling(Tiling4AddRmsNormDynamicQuant)
    .TilingParse<AddRmsNormDynamicQuantCompileInfo>(TilingPrepare4AddRmsNormDynamicQuant);

} // namespace optiling
