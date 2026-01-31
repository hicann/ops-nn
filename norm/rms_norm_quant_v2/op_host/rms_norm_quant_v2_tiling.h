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
 * \file rms_norm_quant_v2_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_QUANT_V2_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_QUANT_V2_H_

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;
namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNormQuantV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, reverse);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(RmsNormQuantV2RegbaseFullLoadTilingData)
TILING_DATA_FIELD_DEF(int64_t, a);           // a
TILING_DATA_FIELD_DEF(int64_t, r);           // r
TILING_DATA_FIELD_DEF(int64_t, q);           // q 传入的scales值，1或者R
TILING_DATA_FIELD_DEF(int64_t, blockFactor); //单核处理a行数
TILING_DATA_FIELD_DEF(int64_t, blockTail);   //尾核处理a行数
TILING_DATA_FIELD_DEF(int64_t, ubFactor);    // ub处理a行数
TILING_DATA_FIELD_DEF(int64_t, binaryAdd);   // r轴二分累加折叠点
TILING_DATA_FIELD_DEF(uint64_t, optionMask); // scales2  zero_points1  zero_points2 beta是否存在
TILING_DATA_FIELD_DEF(int64_t, divMode);     //量化参数
TILING_DATA_FIELD_DEF(int64_t, dstDtype);    //输出类型
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor); // avg_value  1/R
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(RmsNormQuantV2RegbaseRecomputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, numM);
TILING_DATA_FIELD_DEF(int64_t, numN);
TILING_DATA_FIELD_DEF(int64_t, baseM);
TILING_DATA_FIELD_DEF(int64_t, baseN);
TILING_DATA_FIELD_DEF(int64_t, mPerCore);       // 单核处理 A 行数
TILING_DATA_FIELD_DEF(int64_t, mLastCore);      // 尾核处理 A 行数
TILING_DATA_FIELD_DEF(int64_t, nUbLoops);       // ub 处理 r 轴循环次数
TILING_DATA_FIELD_DEF(int64_t, binAddQuotient); // ub整块二分累加折叠点
TILING_DATA_FIELD_DEF(int64_t, powerSplit);     // R 轴 二分点
TILING_DATA_FIELD_DEF(int64_t, mainFoldCount);  // 折叠部分的主块长度
TILING_DATA_FIELD_DEF(int64_t, foldTail);       // 折叠部分的尾块
TILING_DATA_FIELD_DEF(uint64_t, optionMask); // 5个可选参数. needBrc | scales2 | zero_points1 | zero_points2 | beta
TILING_DATA_FIELD_DEF(uint64_t, divMode);  // 量化模式
TILING_DATA_FIELD_DEF(uint64_t, dstDtype); // 输出类型
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);   // 1 / R
END_TILING_DATA_DEF;

constexpr uint32_t RMSNORMQUANTV2_REGBASE_NORMAL = 5000;
constexpr uint32_t RMSNORMQUANTV2_REGBASE_RECOMPUTE = 6000;

REGISTER_TILING_DATA_CLASS(RmsNormQuantV2, RmsNormQuantV2TilingData)
REGISTER_TILING_DATA_CLASS(RmsNormQuantV2_5000, RmsNormQuantV2RegbaseFullLoadTilingData)
REGISTER_TILING_DATA_CLASS(RmsNormQuantV2_6000, RmsNormQuantV2RegbaseRecomputeTilingData)

constexpr int64_t X_INDEX = 0;
constexpr int64_t GAMMA_INDEX = 1;
constexpr int64_t SCALES1_INDEX = 2;
constexpr int64_t SCALES2_INDEX = 3;
constexpr int64_t ZERO_POINTS1_INDEX = 4;
constexpr int64_t ZERO_POINTS2_INDEX = 5;
constexpr int64_t BETA_INDEX = 6;
constexpr int64_t Y1_INDEX = 0;
constexpr int64_t Y2_INDEX = 1;

constexpr int64_t EPS_ATTR_INDEX = 0;
constexpr int64_t DIV_MODE_ATTR_INDEX = 1;

struct RmsNormQuantV2CompileInfo {
    NpuArch curSocVersion = NpuArch::DAV_3510;
    int64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

struct RmsNormQuantV2RegbaseTilingParams {
    // Platform
    uint64_t maxUbSize{0};
    int64_t totalCoreNum{0};
    int64_t vecLength{0};
    int64_t ubBlockSize;
    // Input Info
    int64_t a{1};
    int64_t r{1};
    int64_t q{1};
    int64_t xDtypeSize{0};
    int64_t scaleDtypeSize{0};
    int64_t zeroPointDtypeSize{0};
    int64_t dstDtype{0};
    uint64_t optionMask{0};
    int64_t xDtypeAlignNum{0};
    int64_t scaleDtypeAlignNum{0};
    int64_t zeroPointDtypeAlignNum{0};
    // Cal params
    int64_t baseA{0};
    int64_t baseR{0};
    int64_t rAlign{0};
    int64_t rXDtypeAlign{0};
    int64_t rScaleAlign{0};
    int64_t rZeroPointAlign{0};
    int64_t powerLoop{0};    
    int64_t blockFactor{0};
    int64_t blockTail{0};
    int64_t ubFactor{0};
    int64_t binaryAdd{0};
    int64_t usedCoreNum{0};
    // Workspace
    int64_t workspaceSize{0};
    // Tiling key parmas
    int64_t tilingType{0};

    float epsilon{0};
    float avgFactor{0};
    uint32_t quantBufCnt{0};
    bool divMode{false};
    bool hasScales2{false};
    bool hasZeroPoints1{false};
    bool hasZeroPoints2{false};
    bool hasBeta{false};
    bool hasY2{false};
    bool needGetCompileInfo{false};
};

class RmsNormQuantV2RegbaseTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit RmsNormQuantV2RegbaseTilingBase(gert::TilingContext* tilingContext) : Ops::NN::Optiling::TilingBaseClass(tilingContext)
    {}
    ~RmsNormQuantV2RegbaseTilingBase() override
    {}

    bool CheckShapeBC(
        const gert::StorageShape* srcBcShape, const gert::StorageShape* srcShape, string srcBcName, string srcName);
    ge::graphStatus CheckDtypeVaild(
        ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList, string srcName);
    bool CheckShapeNull();
    bool CheckOptionalInput();
    bool CheckInputShapeDim();
    bool CheckInputShapeValue();
    bool CheckInputDtype();
    bool CheckOutputDtype();
    bool CheckShapeSame(const gert::StorageShape* src1Shape, const gert::StorageShape* src2Shape,
    string nodeName, string src1Name, string src2Name);
    bool CheckShapeBC(const gert::StorageShape* srcBcShape, const gert::StorageShape* srcShape,
    string nodeName, string srcBcName, string srcName);
    bool CheckAllDimsAreOne(const gert::StorageShape* storegeShape);
    ge::graphStatus SetInputParams();
    int64_t CalUBTotalSize(int64_t baseM, int64_t baseN, const uint32_t tilingType);
    ge::graphStatus SetTilingParams();
    void SetTilingData();
    void PrintTilingData();

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override; 
    RmsNormQuantV2RegbaseTilingParams tilingParams;
    const std::string nodeName = "RmsNormQuantV2RegbaseTiling";
};

class RmsNormQuantV2RegbaseTilingFullLoad : public RmsNormQuantV2RegbaseTilingBase {
public:
    explicit RmsNormQuantV2RegbaseTilingFullLoad(gert::TilingContext* tilingContext) : RmsNormQuantV2RegbaseTilingBase(tilingContext)
    {}
    ~RmsNormQuantV2RegbaseTilingFullLoad() override = default;
    void Reset(gert::TilingContext* context) override
    {
        RmsNormQuantV2RegbaseTilingBase::Reset(context);
    }
    void SetTilingData();
    void PrintTilingData();
protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    RmsNormQuantV2RegbaseFullLoadTilingData tilingData;
};

class RmsNormQuantV2RegbaseTilingRecompute : public RmsNormQuantV2RegbaseTilingBase {
public:
    explicit RmsNormQuantV2RegbaseTilingRecompute(gert::TilingContext* tilingContext) : RmsNormQuantV2RegbaseTilingBase(tilingContext)
    {}
    ~RmsNormQuantV2RegbaseTilingRecompute() override = default;
    void Reset(gert::TilingContext* context) override
    {
        RmsNormQuantV2RegbaseTilingBase::Reset(context);
    }
    void PrintTilingData();
protected:
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    int64_t GetPowerSplit(int64_t numN);
    int64_t GetCacheID(const int64_t idx);
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;

private:
    int64_t baseN{64};  // 确保 baseN_  是 2次幂
    int64_t baseM{128};  // 确保 baseM 32B对齐，rstd一个vf
    RmsNormQuantV2RegbaseRecomputeTilingData tilingData;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_RMS_NORM_QUANT_V2_H_
