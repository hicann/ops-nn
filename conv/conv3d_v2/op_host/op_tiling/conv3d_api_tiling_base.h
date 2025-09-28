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
 * \file conv3d_tiling_base.h
 * \brief
 */

#ifndef ASCENDC_TIKCFW_TILING_CONV3D_TILING_BASE_H
#define ASCENDC_TIKCFW_TILING_CONV3D_TILING_BASE_H

#include "conv3d_api_tiling_utils.h"
#include "conv3d_v2_tiling.h"

namespace Conv3dApiTiling {

constexpr uint32_t NUB_LOAD = 0; // LoadChannelType::NORMAL
constexpr uint32_t SINGLECO_LOAD = 1; // LoadChannelType::LOAD_TOTAL_CORE
constexpr uint32_t NL0_LOAD = 2; // LoadChannelType::LOAD_TOTAL_LC0
constexpr uint32_t KL0_LIMIT = 4096;  // MMad limits
constexpr uint8_t PER_CHANNEL_NO_OFFSET = 6; // QuantType::PER_CHANNEL_NO_OFFSET

constexpr int64_t INITIAL_SHAPE_MINUS_ONE = -1;
constexpr uint64_t INITIAL_SHAPE_ZERO = 0;

struct Conv3DL1Tiling {
    uint64_t kAL1 = INITIAL_TILING;
    uint64_t kBL1 = INITIAL_TILING;
    uint64_t mAL1Value = INITIAL_TILING;
    uint64_t nBL1Value = INITIAL_TILING;
    uint64_t mAL1DivmL0 = INITIAL_TILING;
    uint64_t nBL1DivnL0 = INITIAL_TILING;
    uint64_t cin1InAL1 = INITIAL_TILING;
    uint64_t kAL1Tail = INITIAL_TILING;
    uint64_t cin1InAL1Tail = INITIAL_TILING;
    uint64_t KBL1Divk0 = INITIAL_TILING;
    uint64_t kBL1Tail = INITIAL_TILING;
    uint64_t KBL1TailDivk0 = INITIAL_TILING;

    IterateMNOrder iterateMNOrder;
    bool isWeightBypass;
    bool biasFullLoadFlag;
    bool fixpParamsFullLoadFlag;
    bool al1FullLoad;
    bool bl1FullLoad;
};

struct Conv3DL0Tiling {
    uint64_t mL0 = INITIAL_TILING;
    uint64_t kL0 = INITIAL_TILING;
    uint64_t nL0 = INITIAL_TILING;
    uint64_t nL0xk0 = INITIAL_TILING;
    uint64_t kL0xorgCoAlignN0 = INITIAL_TILING;
};

struct Conv3DUBTiling {
    uint64_t mUB = INITIAL_TILING;
    uint64_t nUB = INITIAL_TILING;
    uint32_t scaleAndBiasLoadType = NUB_LOAD;
};

struct Conv3DInputshape {
    int64_t orgkH = INITIAL_SHAPE_MINUS_ONE;
    int64_t orgkW = INITIAL_SHAPE_MINUS_ONE;
    int64_t orgkD = INITIAL_SHAPE_MINUS_ONE;
    int64_t orgCo = INITIAL_SHAPE_MINUS_ONE;
    int64_t coutOpt = INITIAL_SHAPE_MINUS_ONE;
    int64_t orgCi = INITIAL_SHAPE_MINUS_ONE;
    int64_t cinOpt = INITIAL_SHAPE_MINUS_ONE;
    int64_t orgDi = INITIAL_SHAPE_MINUS_ONE;
    int64_t orgHi = INITIAL_SHAPE_MINUS_ONE;
    int64_t orgWi = INITIAL_SHAPE_MINUS_ONE;

    int64_t singlekH = INITIAL_SHAPE_MINUS_ONE;
    int64_t singlekW = INITIAL_SHAPE_MINUS_ONE;
    int64_t singlekD = INITIAL_SHAPE_MINUS_ONE;
    int64_t singleCi = INITIAL_SHAPE_MINUS_ONE;
    int64_t singleCo = INITIAL_SHAPE_MINUS_ONE;
    int64_t singleDo = INITIAL_SHAPE_MINUS_ONE;
    int64_t singleM = INITIAL_SHAPE_MINUS_ONE;
    int64_t singleHo = INITIAL_SHAPE_MINUS_ONE;
    int64_t singleWo = INITIAL_SHAPE_MINUS_ONE;
    int64_t singleCoreGroupOpt = INITIAL_SHAPE_MINUS_ONE;
};

struct Conv3DInputAttr {
    int64_t groups = 1;
    int64_t groupOpt = 1;

    int64_t padHead = 0;
    int64_t padTail = 0;
    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;

    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t strideD = 1;

    int64_t dilationH = 1;
    int64_t dilationW = 1;
    int64_t dilationD = 1;

    int8_t offsetx = INITIAL_OFFSET;
    uint8_t hf32Enable = 0;
    uint8_t hf32TransMode = 0;
};

struct Conv3DCalcShape {
    uint64_t singleCi1 = INITIAL_SHAPE_ZERO;
    uint64_t singleCo1 = INITIAL_SHAPE_ZERO;
    uint64_t singleM1 = INITIAL_SHAPE_ZERO;
    uint64_t orgHo = INITIAL_SHAPE_ZERO;
    uint64_t orgWo = INITIAL_SHAPE_ZERO;
    uint64_t orgDo = INITIAL_SHAPE_ZERO;
};

struct Conv3DDesc {
    ConvType fMapType  = {ConvFormat::NDC1HWC0, ConvDtype::FLOAT16, TPosition::GM};
    ConvType weightType  = {ConvFormat::FRACTAL_Z_3D, ConvDtype::FLOAT16, TPosition::GM};
    ConvType biasType = {ConvFormat::ND, ConvDtype::FLOAT16, TPosition::GM};
    ConvType outputType = {ConvFormat::NDC1HWC0, ConvDtype::FLOAT16, TPosition::CO1};
    ConvType quantScaleType = {ConvFormat::ND, ConvDtype::INT64, TPosition::GM};
};

struct DoubleBufferValue {
    uint8_t pbAL1 = INITIAL_PING_PONG_AL1;
    uint8_t pbBL1 = INITIAL_PING_PONG_BL1;
    uint8_t pbAL0 = INITIAL_PING_PONG_AL0;
    uint8_t pbBL0 = INITIAL_PING_PONG_BL0;
    uint8_t pbCL0 = INITIAL_PING_PONG_CL0;
    uint8_t pbUB = INITIAL_PING_PONG_UB;
    uint64_t pBufferFlag;
};

struct CubeInfo {
    uint32_t m0;
    uint32_t k0;
    uint32_t n0;
    ConvDtype madType;
    ConvDtype biasType;
    uint32_t minBurstNum;
};

class Conv3dTilingBase {
public:
    Conv3dTilingBase() = default;
    explicit Conv3dTilingBase(const platform_ascendc::PlatformAscendC& ascendcPlatform);
    explicit Conv3dTilingBase(const PlatformInfo& platform);
    virtual ~Conv3dTilingBase() = default;
    virtual int64_t GetTiling(optiling::TConv3DTiling& tiling) = 0;
    void SetOrgWeightShape(int64_t orgCo, int64_t orgKd, int64_t orgKh, int64_t orgKw);
    void SetOrgFmapShape(int64_t orgCi, int64_t orgDi, int64_t orgHi, int64_t orgWi);
    void SetSingleWeightShape(int64_t singleCi, int64_t singleKd, int64_t singleKh, int64_t singleKw);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleM);
    void SetSingleOutputShape(int64_t singleCo, int64_t singleDo, int64_t singleHo, int64_t singleWo);
    void SetOutputOrder(int8_t outputOrder);
    void SetWeightType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetFmapType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetBiasType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetScaleType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetQuantType();
    void SetOutputType(TPosition pos, ConvFormat format, ConvDtype dtype);
    void SetPadding(std::vector<int64_t>& padList);
    void SetDilation(int64_t dilationH, int64_t dilationW, int64_t dilationD);
    void SetStride(int64_t strideH, int64_t strideW, int64_t strideD);
    void SetHF32(bool hf32Enable, bool hf32TransMode);
    bool CalOptGroupParams(const Conv3DOriGroupInfo &oriGroupInfo, Conv3DGroupOptInfo &groupOptInfo) const;
    void SetGroups(int64_t groups);
    void SetOptGroupInfo(int64_t groupOpt, int64_t singleCoreGroupOpt, int64_t cinOpt, int64_t coutOpt);

    Conv3DDesc descInfo;
    Conv3DInputshape shapeInfo;
    Conv3DCalcShape shapeCalc;
    Conv3DInputAttr attrInfo;
    CubeInfo cubeInfo;
    Conv3DL1Tiling l1TilingInfo;
    Conv3DL0Tiling l0TilingInfo;
    Conv3DUBTiling ubTilingInfo;
    DoubleBufferValue dbValue;
    PlatformInfo platformInfo;

    bool hasBias = false;
    bool hasQuantScale = false;
    uint8_t quantType = DISABLE_QUANT;

    bool hf32Enable_ = false;
    bool hf32TransMode_ = false;
    int8_t outputOrder_ = M_Mode;

protected:
    virtual int64_t Compute() = 0;
    void SetFinalTilingBasicInfo(optiling::TConv3DTiling& tiling);
    void SetFinalTilingDecisionInfo(optiling::TConv3DTiling& tiling);
    void SetFinalTiling(optiling::TConv3DTiling& tiling);
    void PrintTilingDataBasicInfo() const;
    void PrintTilingDataDecision() const;
    void PrintTilingData() const;
    bool CheckInputParam() const;
    bool CheckInputParamPointWise() const;
    void GetCubeInfo();
    bool ShapeInitCalc();
    bool CheckParamsOverflow() const;

private:
    bool CheckInputAttr() const;
    bool CheckInputAttrPointWise() const;
    bool CheckOrgInputInfo() const;
    bool CheckOrgInputShapeWithPad() const;
    bool CheckOrgInputInfoPointWise() const;
    bool CheckSingleInputInfo() const;
    bool CheckSingleInputInfoPointWise() const;
    bool CheckInputConstraint() const;
    bool CheckInputShape() const;
    bool CheckInputShapePointWise() const;
    bool CheckInputFormat() const;
    bool CheckInputFormatPointWise() const;
    bool CheckParamsDtypeHasQuantScale() const;
    bool CheckParamsDtypeHasBias() const;
    bool CheckParamsDtypeEssential() const;
    bool CheckParamsDtype() const;
    bool CheckParamsDtypePointWise() const;
    bool CheckLoad3DLimits() const;
    bool CheckInstructionLimits() const;
    bool CheckHF32() const;
};

} // namespace Conv3dApiTiling

#endif // ASCENDC_TIKCFW_TILING_CONV3D_TILING_BASE_H