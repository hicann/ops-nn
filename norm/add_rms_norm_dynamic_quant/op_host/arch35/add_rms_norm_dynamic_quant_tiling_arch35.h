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
 * \file add_rms_norm_dynamic_quant_tiling_arch35.h
 * \brief A5 (arch35/ascend950) tiling header
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_TILING_ARCH35_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_TILING_ARCH35_H
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "../../op_kernel/arch35/add_rms_norm_dynamic_quant_tiling_data.h"
#include "../../op_kernel/arch35/add_rms_norm_dynamic_quant_tiling_key.h"
#include "tiling/tiling_api.h"

namespace optiling {

// Input/Output indices
constexpr uint64_t X1_INDEX = 0;
constexpr uint64_t X2_INDEX = 1;
constexpr uint64_t GAMMA_INDEX = 2;
constexpr uint64_t SMOOTH_SCALE1_INDEX = 3;
constexpr uint64_t SMOOTH_SCALE2_INDEX = 4;
constexpr uint64_t BETA_INDEX = 5;
constexpr uint64_t Y1_INDEX = 0;
constexpr uint64_t Y2_INDEX = 1;
constexpr uint64_t X_INDEX = 2;
constexpr uint64_t SCALE1_INDEX = 3;
constexpr uint64_t SCALE2_INDEX = 4;

constexpr uint64_t Y3_INDEX = 2;
constexpr uint64_t Y4_INDEX = 3;
constexpr uint64_t X_INDEX_V2 = 4;
constexpr uint64_t SCALE1_INDEX_V2 = 5;
constexpr uint64_t SCALE2_INDEX_V2 = 6;

constexpr uint32_t MAX_DIM_CNT = 8;
constexpr int OUTPUT_MASK_ATTR_IDX = 1;
constexpr int DST_TYPE_ATTR_INDEX = 2;
constexpr int INT4_PACK_RATIO = 2; // two int4 elements packed into one byte
constexpr int OUTPUT_MASK_LEN_V1 = 2;
constexpr int OUTPUT_MASK_LEN_V2 = 4;
constexpr int OUTPUT_MASK_NULLPTR_LEN = 0;

// Internal tiling type constants used by DoOpTiling strategy selection
constexpr uint32_t TILING_TYPE_PERF = 0;
constexpr uint32_t TILING_TYPE_NORMAL = 1;
constexpr uint32_t TILING_TYPE_SINGLE_ROW = 2;
constexpr uint32_t TILING_TYPE_SPILT = 3;

// Compute mode enum matching ASCENDC_TPL_ARGS_DECL in tiling_key.h
enum class ComputeMode : uint64_t {
    PERF = 0,
    NORMAL = 1,
    SINGLE_ROW = 2,
    SPLIT = 3,
    REDUCE_EMPTY = 4,
};

enum class Y3Mode : uint64_t {
    NO_Y3 = 0,
    HAS_Y3 = 1,
};

enum class Y4Mode : uint64_t {
    NO_Y4 = 0,
    HAS_Y4 = 1,
};

class AddRmsNormDynamicQuantTilingKey {
public:
    AddRmsNormDynamicQuantTilingKey& SetComputeMode(ComputeMode mode, Y3Mode y3Mode, Y4Mode y4Mode)
    {
        computeMode_ = mode;
        y3Mode_ = y3Mode;
        y4Mode_ = y4Mode;
        return *this;
    }

    uint64_t GetTilingKey() const
    {
        return GET_TPL_TILING_KEY(static_cast<uint64_t>(computeMode_), static_cast<uint64_t>(y3Mode_),
                                  static_cast<uint64_t>(y4Mode_));
    }

private:
    ComputeMode computeMode_ = ComputeMode::NORMAL;
    Y3Mode y3Mode_ = Y3Mode::HAS_Y3;
    Y4Mode y4Mode_ = Y4Mode::HAS_Y4;
};

struct AddRmsNormDynamicQuantCompileInfo {
    platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND950;
    uint64_t totalCoreNum = 0;
    uint64_t maxUbSize = 0;
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
    bool hasBeta{false};
    uint32_t outQuant1Flag{1};
    uint32_t outQuant2Flag{0};
    bool needGetCompileInfo{false};
    bool needRun{true};
    bool hasY3{false};
    bool hasY4{false};
};

class AddRmsNormDynamicQuantRegbaseTilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit AddRmsNormDynamicQuantRegbaseTilingBase(gert::TilingContext* tilingContext)
        : Ops::NN::Optiling::TilingBaseClass(tilingContext)
    {}
    ~AddRmsNormDynamicQuantRegbaseTilingBase() override {}

    const string nodeName_ = "AddRmsNormDynamicQuantRegbase";

    // Common validation methods
    ge::graphStatus CheckDtypeVaild(ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList,
                                    string srcName);
    bool CheckShapeNull();
    bool ParseOutputFlags();
    bool CheckInputAttr();
    bool CheckInputShapeDim();
    bool CheckInputShapeValue();
    bool CheckOutputShapeValue();
    bool CheckInputDtype();
    bool CheckOutputDtype();
    virtual ge::graphStatus SetInputParams() = 0;
    ge::graphStatus GetShapeAttrsInfo() override;

protected:
    uint64_t aivCoreNum_{0};
    uint64_t numN_{0};
    uint64_t numM_{0};
    bool hasSmoothScale1_{false};
    bool hasSmoothScale2_{false};
    bool hasBeta_{false};
    bool hasY3_{false};
    bool hasY4_{false};
    bool isV2_{false};
    uint32_t outQuant1Flag_{1};
    uint32_t outQuant2Flag_{0};
};

class AddRmsNormDynamicQuantRegbaseTiling : public AddRmsNormDynamicQuantRegbaseTilingBase {
public:
    explicit AddRmsNormDynamicQuantRegbaseTiling(gert::TilingContext* tilingContext)
        : AddRmsNormDynamicQuantRegbaseTilingBase(tilingContext)
    {}
    ~AddRmsNormDynamicQuantRegbaseTiling() override {}

    const string nodeName = "AddRmsNormDynamicQuantRegbase";
    AddRmsNormDynamicQuantRegbaseTilingData tilingData;
    AddRmsNormDynamicQuantRegbaseTilingParams tilingParams;

    ge::graphStatus SetInputParams();
    uint64_t CalUBTotalSize(uint64_t baseM, uint64_t baseN, const uint32_t tilingType);
    int64_t CalFullLoadBaseM(uint64_t baseN, int64_t& tmpPower);
    uint64_t CalUsedSize(uint64_t baseM, uint64_t baseNB8Align, uint64_t baseNB32Align, uint64_t baseNDtypeAlign,
                         int64_t firstVcaddLength);
    ge::graphStatus SetTilingParams();
    void SetTilingData();
    void PrintTilingData();

private:
    bool TryPerfTiling();
    bool TryNormTiling();
    bool TrySingleRowTiling();
    bool TrySplitTiling();

protected:
    // Order: GetShapeAttrsInfo->GetPlatformInfo->
    //        IsCapable->DoOpTiling->DoLibApiTiling->
    //        GetWorkspaceSize->PostTiling->GetTilingKey
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
};

class AddRmsNormDynamicQuantEmptyTiling : public AddRmsNormDynamicQuantRegbaseTilingBase {
public:
    explicit AddRmsNormDynamicQuantEmptyTiling(gert::TilingContext* tilingContext)
        : AddRmsNormDynamicQuantRegbaseTilingBase(tilingContext)
    {}
    ~AddRmsNormDynamicQuantEmptyTiling() override {}

    ge::graphStatus SetInputParams();
    uint64_t CalUBTotalSize(uint64_t baseM, uint64_t baseN, const uint32_t tilingType);
    ge::graphStatus SetTilingParams();
    void SetTilingData();

protected:
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;

private:
    uint64_t usedCoreNum_{0};
    uint64_t mPerCore_{0};
    uint64_t mPerUB_{0};
    uint64_t mTailUb_{0};
    uint64_t lastCoreBlockCount_{0};
    uint64_t mlastCoreTailUb_{0};
    uint64_t coreUbBlockCount_{0};
    uint64_t mLastCore_{0};
    uint64_t ubSize_{0};

    uint32_t tilingKey_;
    AddRmsNormDynamicQuantEmptyTilingData tilingData_;

    void CalcTilingData();
    uint64_t NearestLowerPowerOfTwo(uint64_t tmp);
    void CalcUsedCoreNum();
    void LogTilingResult();
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_DYN_QUANT_TILING_ARCH35_H
