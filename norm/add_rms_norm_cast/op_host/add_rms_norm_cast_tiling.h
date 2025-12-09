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
 * \file add_rms_norm_cast_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_CAST_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_CAST_H_

#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddRMSNormCastTilingData)
TILING_DATA_FIELD_DEF(uint32_t, num_row);
TILING_DATA_FIELD_DEF(uint32_t, num_col);
TILING_DATA_FIELD_DEF(uint32_t, block_factor);
TILING_DATA_FIELD_DEF(uint32_t, row_factor);
TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, avg_factor);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(AddRmsNormCastRegbaseTilingData)
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
TILING_DATA_FIELD_DEF(uint32_t, isNddma);
END_TILING_DATA_DEF;

constexpr uint32_t TILING_TYPE_NORMAL = 0;
constexpr uint32_t TILING_TYPE_SPILT = 1;
constexpr uint32_t TILING_TYPE_MAGIC_AND_WONDERFUL = 2;
constexpr uint32_t TILING_OFFSET_REGBASE = 100;
constexpr uint64_t TILING_KEY_UNRUN = 199;

constexpr uint64_t MIGIC_AND_WONDERFUL_BYTES = 128;

struct AddRmsNormCastCompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

REGISTER_TILING_DATA_CLASS(AddRmsNormCast, AddRMSNormCastTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormCast_100, AddRmsNormCastRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormCast_101, AddRmsNormCastRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormCast_102, AddRmsNormCastRegbaseTilingData)
REGISTER_TILING_DATA_CLASS(AddRmsNormCast_199, AddRmsNormCastRegbaseTilingData)

struct AddRmsNormCastRegbaseTilingParams {
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
    uint64_t baseNDtypeAlign{0};
    uint64_t baseNReduceAlign{0};
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
    bool needGetCompileInfo{false};
    bool needRun{true};
    uint32_t isNddma{1};
};

class AddRmsNormCastRegbaseTiling : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit AddRmsNormCastRegbaseTiling(gert::TilingContext* tilingContext) :
        Ops::NN::Optiling::TilingBaseClass(tilingContext) {}
    ~AddRmsNormCastRegbaseTiling() override = default;  

    const string nodeName = "AddRmsNormCastRegbase";
    AddRmsNormCastRegbaseTilingData tilingData;
    AddRmsNormCastRegbaseTilingParams tilingParams;

    ge::graphStatus CheckDtypeVaild(
        ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList, string srcName);
    bool CheckShapeNull();
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

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_ADD_RMS_NORM_CAST_H_
