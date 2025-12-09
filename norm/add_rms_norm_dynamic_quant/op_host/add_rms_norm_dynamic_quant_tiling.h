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
 * \file add_rms_norm_dynamic_quant_tiling.h
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_TILING_H
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "error_util.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(AddRmsNormDynamicQuantTilingData)
TILING_DATA_FIELD_DEF(uint64_t, useCore);
TILING_DATA_FIELD_DEF(uint64_t, numFirstDim);
TILING_DATA_FIELD_DEF(uint64_t, numLastDim);
TILING_DATA_FIELD_DEF(uint64_t, numLastDimAligned);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerCore);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerCoreTail);
TILING_DATA_FIELD_DEF(uint64_t, firstDimPerLoop);
TILING_DATA_FIELD_DEF(uint64_t, lastDimLoopNum);
TILING_DATA_FIELD_DEF(uint64_t, lastDimSliceLen);
TILING_DATA_FIELD_DEF(uint64_t, lastDimSliceLenTail);
TILING_DATA_FIELD_DEF(uint32_t, smoothNum1);
TILING_DATA_FIELD_DEF(uint32_t, smoothNum2);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int32_t, outQuant1Flag);
TILING_DATA_FIELD_DEF(int32_t, outQuant2Flag);
TILING_DATA_FIELD_DEF(float, avgFactor);
TILING_DATA_FIELD_DEF(uint32_t, betaFlag);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(AddRmsNormDynamicQuantRegbaseTilingData)
TILING_DATA_FIELD_DEF(uint64_t, numM);
TILING_DATA_FIELD_DEF(uint64_t, numN);
TILING_DATA_FIELD_DEF(uint64_t, baseM);
TILING_DATA_FIELD_DEF(uint64_t, baseN);
TILING_DATA_FIELD_DEF(uint64_t, baseNDtypeAlign);
TILING_DATA_FIELD_DEF(uint64_t, baseNReduceAlign);
TILING_DATA_FIELD_DEF(uint64_t, powerSplit);
TILING_DATA_FIELD_DEF(uint64_t, powerLoop);
TILING_DATA_FIELD_DEF(uint64_t, mPerCore);
TILING_DATA_FIELD_DEF(uint64_t, mLastCore);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avgFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuant, AddRmsNormDynamicQuantTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuant_100, AddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuant_120, AddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuant_130, AddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuant_101, AddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuant_121, AddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuant_131, AddRmsNormDynamicQuantRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormDynamicQuant_199, AddRmsNormDynamicQuantRegbaseTilingData)

constexpr uint32_t TILING_TYPE_NORMAL = 0;
constexpr uint32_t TILING_TYPE_SPILT = 1;
constexpr uint32_t TILING_OFFSET_HAS_QUANT = 10;
constexpr uint32_t TILING_OFFSET_REGBASE = 100;
constexpr uint64_t TILING_KEY_UNRUN = 199;

struct AddRmsNormDynamicQuantCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910_95;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
};

enum class UB_TILING_POLICY : std::int32_t
{
    NORMAL,
    SINGLE_ROW,
    SLICE_D
};

class AddRmsNormDynamicQuantTilingHelper {
public:
    explicit AddRmsNormDynamicQuantTilingHelper(gert::TilingContext* context) : context_(context)
    {}

    ~AddRmsNormDynamicQuantTilingHelper() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(AddRmsNormDynamicQuantTilingData* tiling);

private:
    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool DoUbTiling();
    bool CheckInputOutputShape();

    bool CheckUbNormalTiling();
    bool CheckUbSingleRowTiling();
    bool CheckUbSliceDTiling();

    gert::TilingContext* context_;

    ge::DataType xDtype_{ge::DataType::DT_FLOAT16};
    uint64_t dtSize_{2};
    uint64_t socCoreNums_{1};
    uint64_t ubSize_{1};
    uint64_t sysWorkspaceSize_{1};

    uint64_t useCore_{1};
    uint64_t numFirstDim_{1};
    uint64_t numLastDim_{1};
    uint64_t numLastDimAligned_{1};
    uint64_t firstDimPerCore_{1};
    uint64_t firstDimPerCoreTail_{1};
    uint64_t firstDimPerLoop_{1};
    uint64_t lastDimSliceLen_{1};
    uint64_t lastDimLoopNum_{1};
    uint64_t lastDimSliceLenTail_{1};
    float eps_{1e-6};
    int32_t outQuant1Flag{0};
    int32_t outQuant2Flag{0};
    float avgFactor_{0.0};
    uint32_t smoothNum1_{0};
    uint32_t smoothNum2_{0};
    uint32_t betaFlag_{0};
    UB_TILING_POLICY ubTilingPolicy_{UB_TILING_POLICY::SINGLE_ROW};
};

struct AddRmsNormDynamicQuantRegbaseTilingParams {
    // Platform
    uint64_t maxUbSize{0};
    uint64_t totalCoreNum{0};
    uint64_t vecLength{0};
    // Input Info
    uint64_t numM{0};
    uint64_t numN{0};
    uint64_t xDtypeSize{0};
    uint64_t xDtypeAlignNum{0};
    uint64_t xReduceAlignNum{0};
    // Cal params
    uint64_t baseM{0};
    uint64_t baseN{0};
    uint64_t baseNB8Align{0};
    uint64_t baseNDtypeAlign{0};
    uint64_t baseNReduceAlign{0};
    uint64_t reduceBufLenAlign{0};
    uint64_t powerSplit{0};
    uint64_t powerLoop{0};
    uint64_t mPerCore{0};
    uint64_t mLastCore{0};
    uint64_t usedCoreNum{0};
    // Workspace
    uint64_t workspaceSize{0};
    // Tiling key parmas
    uint64_t tilingType{0};

    float epsilon{0};
    float avgFactor{0};
    uint32_t quantBufCnt{0};
    bool hasSmoothScale1{false};
    bool hasSmoothScale2{false};
    bool hasY2Scale2{false};
    bool needGetCompileInfo{false};
    bool needRun{true};
};

class AddRmsNormDynamicQuantRegbaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit AddRmsNormDynamicQuantRegbaseTiling(gert::TilingContext* tilingContext) : Ops::NN::Optiling::TilingBaseClass(tilingContext)
    {}
    ~AddRmsNormDynamicQuantRegbaseTiling() override
    {}

    const string nodeName = "AddRmsNormDynamicQuantRegbase";
    AddRmsNormDynamicQuantRegbaseTilingData tilingData;
    AddRmsNormDynamicQuantRegbaseTilingParams tilingParams;

    ge::graphStatus CheckDtypeVaild(
        ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList, string srcName);
    bool CheckShapeNull();
    bool CheckOptionalInput();
    bool CheckInputShapeDim();
    bool CheckInputShapeValue();
    bool CheckInputDtype();
    ge::graphStatus SetInputParams();
    uint64_t CalUBTotalSize(uint64_t baseM, uint64_t baseN, const uint32_t tilingType);
    ge::graphStatus SetTilingParams();
    void SetTilingData();
    void PrintTilingData();

protected:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->PostTiling->GetTilingKey
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_TILING_H
