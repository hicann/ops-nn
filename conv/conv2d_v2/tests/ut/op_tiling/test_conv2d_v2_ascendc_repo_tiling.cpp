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
 * \file test_conv2d_v2_ascendc_repo_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

#include "log/log.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"

#include "runtime_kb_api.h"
#include "test_cube_util.h"
#include "../../../op_host/op_tiling/arch35/conv2d_v2_tuning_tiling.h"
#include "../../../op_host/op_tiling/arch35/conv2d_v2_base_tiling.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base.h"
#include "test_conv2d_v2_ascendc_utils_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace conv_tiling_utils;

namespace {
string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }
    return result;
}

struct Conv2DTilingTestParamRepo {
    string case_name;
    string op_type;
    string tiling_data;
    string info_dict;
};

string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy_s(&tmp, sizeof(tmp), data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}
} // namespace

class Conv2DTilingRepo : public testing::TestWithParam<Conv2DTilingTestParamRepo> {
protected:
    void SetUp() {}
    void TearDown() {}
    static void TearDownTestCase() {}
    void PrepareTest(Conv2DTilingTestParamRepo& param)
    {
        PrepareKnowledge(param);
        PrepareInfoDict(param);
    }
    void PrepareKnowledge(Conv2DTilingTestParamRepo& param)
    {
        auto j = nlohmann::json::parse(param.tiling_data);
        conv2d_knowledge.groups = j["groups"];
        conv2d_knowledge.orgCi = j["orgCi"];
        conv2d_knowledge.orgHi = j["orgHi"];
        conv2d_knowledge.orgWi = j["orgWi"];
        conv2d_knowledge.orgCo = j["orgCo"];
        conv2d_knowledge.orgHo = j["orgHo"];
        conv2d_knowledge.orgWo = j["orgWo"];
        conv2d_knowledge.kernelH = j["kernelH"];
        conv2d_knowledge.kernelW = j["kernelW"];
        conv2d_knowledge.singleCoreCi = j["singleCoreCi"];
        conv2d_knowledge.singleCoreCo = j["singleCoreCo"];
        conv2d_knowledge.singleCoreHo = j["singleCoreHo"];
        conv2d_knowledge.singleCoreWo = j["singleCoreWo"];
        conv2d_knowledge.hoL1 = j["hoL1"];
        conv2d_knowledge.woL1 = j["woL1"];
        conv2d_knowledge.kAL1 = j["kAL1"];
        conv2d_knowledge.kBL1 = j["kBL1"];
        conv2d_knowledge.hoL0 = j["hoL0"];
        conv2d_knowledge.woL0 = j["woL0"];
        conv2d_knowledge.kL0 = j["kL0"];
        conv2d_knowledge.nL0 = j["nL0"];
        conv2d_knowledge.pBufferFlag = j["pBufferFlag"];
        conv2d_knowledge.nBL1 = j["nBL1"];
        conv2d_knowledge.strideH = j["strideH"];
        conv2d_knowledge.strideW = j["strideW"];
        conv2d_knowledge.dilationH = j["dilationH"];
        conv2d_knowledge.dilationW = j["dilationW"];
        conv2d_knowledge.padTop = j["padTop"];
        conv2d_knowledge.padBottom = j["padBottom"];
        conv2d_knowledge.padLeft = j["padLeft"];
        conv2d_knowledge.padRight = j["padRight"];
        conv2d_knowledge.iterateMNOrder = j["iterateMNOrder"];
        conv2d_knowledge.biasFullLoadFlag = j["biasFullLoadFlag"];
        conv2d_knowledge.fixpParamsFullLoadFlag = j["fixpParamsFullLoadFlag"];
        conv2d_knowledge.offsetx = j["offsetx"];
        conv2d_knowledge.hf32Enable = j["hf32Enable"];
        conv2d_knowledge.hf32TransMode = j["hf32TransMode"];
        conv2d_knowledge.isC04Flag = j["isC04Flag"];
        conv2d_knowledge.roundMode = j["roundMode"];
        conv2d_knowledge.batchDim = j["batchDim"];
        conv2d_knowledge.hoDim = j["hoDim"];
        conv2d_knowledge.nDim = j["nDim"];
        conv2d_knowledge.nDim = j["nDim"];
        conv2d_knowledge.groupDim = j["groupDim"];
        conv2d_knowledge.mMode = j["mMode"];
    }

    void PrepareInfoDict(Conv2DTilingTestParamRepo& param)
    {
        auto j = nlohmann::json::parse(param.info_dict);
        conv2d_info_dict.aDtype = j["aDtype"];
        conv2d_info_dict.bDtype = j["bDtype"];
        conv2d_info_dict.cDtype = j["cDtype"];
        conv2d_info_dict.biasDtype = j["biasDtype"];
        conv2d_info_dict.aShapeN = j["aShapeN"];
        conv2d_info_dict.aShapeH = j["aShapeH"];
        conv2d_info_dict.aShapeW = j["aShapeW"];
        conv2d_info_dict.bShapeN = j["bShapeN"];
        conv2d_info_dict.bShapeC = j["bShapeC"];
        conv2d_info_dict.bShapeH = j["bShapeH"];
        conv2d_info_dict.bShapeW = j["bShapeW"];
        conv2d_info_dict.cShapeH = j["cShapeH"];
        conv2d_info_dict.cShapeW = j["cShapeW"];
        conv2d_info_dict.aFormat = j["aFormat"];
        conv2d_info_dict.bFormat = j["bFormat"];
        conv2d_info_dict.cFormat = j["cFormat"];
        conv2d_info_dict.groups = j["groups"];
        conv2d_info_dict.strideH = j["strideH"];
        conv2d_info_dict.strideW = j["strideW"];
        conv2d_info_dict.dilationH = j["dilationH"];
        conv2d_info_dict.dilationW = j["dilationW"];
        conv2d_info_dict.padTop = j["padTop"];
        conv2d_info_dict.padBottom = j["padBottom"];
        conv2d_info_dict.padLeft = j["padLeft"];
        conv2d_info_dict.padRight = j["padRight"];
        conv2d_info_dict.biasFlag = j["biasFlag"];
        conv2d_info_dict.hf32Flag = j["hf32Flag"];
    }
    std::string bank_path;
    const char* optype = "Conv2DV2";
    gert::TilingContext* context = nullptr;
    tuningtiling::Conv2DV2TunnerTiling conv2d_knowledge;
    tuningtiling::Conv2DV2InputArgs conv2d_info_dict;
};

TEST_P(Conv2DTilingRepo, general_cases_001)
{
    Conv2DTilingTestParamRepo param = GetParam();
    PrepareTest(param);
    shared_ptr<tuningtiling::Conv2DV2InputArgs> conv2DInput = std::make_shared<tuningtiling::Conv2DV2InputArgs>();
    conv2DInput->aDtype = ge::DT_FLOAT;
    conv2DInput->bDtype = ge::DT_FLOAT;
    conv2DInput->cDtype = ge::DT_FLOAT;
    conv2DInput->biasDtype = ge::DT_FLOAT;
    conv2DInput->aShapeN = 2048;
    conv2DInput->aShapeH = 1;
    conv2DInput->aShapeW = 16;
    conv2DInput->bShapeN = 64;
    conv2DInput->bShapeC = 960;
    conv2DInput->bShapeH = 1;
    conv2DInput->bShapeW = 1;
    conv2DInput->cShapeH = 1;
    conv2DInput->cShapeW = 16;
    conv2DInput->aFormat = ge::FORMAT_NCHW;
    conv2DInput->bFormat = ge::FORMAT_NCHW;
    conv2DInput->cFormat = ge::FORMAT_NCHW;
    conv2DInput->groups = 1;
    conv2DInput->strideH = 1;
    conv2DInput->strideW = 1;
    conv2DInput->dilationH = 1;
    conv2DInput->dilationW = 1;
    conv2DInput->padTop = 0;
    conv2DInput->padBottom = 0;
    conv2DInput->padLeft = 0;
    conv2DInput->padRight = 0;
    conv2DInput->biasFlag = true;
    conv2DInput->hf32Flag = true;
    conv2DInput->reserverdParam4 = 0;
    conv2DInput->reserverdParam5 = 0;
    conv2DInput->reserverdParam6 = 0;
    size_t inputArgsSize = sizeof(tuningtiling::Conv2DV2InputArgs);
    shared_ptr<tuningtiling::TuningTilingDef> tuningTiling = nullptr;
    std::string kbFile = "Ascend950PR_9589_32_AiCore_Conv2DV2_runtime_kb.json";
    std::string socVersion = "Ascend950PR_9589";

    tuningtiling::TuningTilingDefPtr tiling = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance(optype);
    nlohmann::json a;

    conv2d_knowledge.ToJson(a);
    tiling->FromJson(a);

    gert::StorageShape featuremap = {
        {conv2d_info_dict.aShapeN, conv2d_info_dict.bShapeC, conv2d_info_dict.aShapeH, conv2d_info_dict.aShapeW},
        {conv2d_info_dict.aShapeN, conv2d_info_dict.bShapeC, conv2d_info_dict.aShapeH, conv2d_info_dict.aShapeW}};
    gert::StorageShape weight = {
        {conv2d_info_dict.bShapeN, conv2d_info_dict.bShapeC, conv2d_info_dict.bShapeH, conv2d_info_dict.bShapeW},
        {conv2d_info_dict.bShapeN, conv2d_info_dict.bShapeC, conv2d_info_dict.bShapeH, conv2d_info_dict.bShapeW}};
    gert::StorageShape bias = {{conv2d_info_dict.bShapeN}, {conv2d_info_dict.bShapeN}};
    gert::StorageShape quantScale = {{conv2d_info_dict.bShapeN}, {conv2d_info_dict.bShapeN}};
    gert::StorageShape offset_w;

    gert::StorageShape output = {
        {conv2d_info_dict.aShapeN, conv2d_info_dict.bShapeN, conv2d_info_dict.cShapeH, conv2d_info_dict.cShapeW},
        {conv2d_info_dict.aShapeN, conv2d_info_dict.bShapeN, conv2d_info_dict.cShapeH, conv2d_info_dict.cShapeW}};
    std::vector<void*> input_shape_ref;

    bool hasScale = false;
    bool hasBias = conv2d_info_dict.biasFlag;

    if (hasBias) {
        input_shape_ref = {&featuremap, &weight, &bias, nullptr};
    } else {
        input_shape_ref = {&featuremap, &weight, nullptr, nullptr};
    }

    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides = {1, 1, conv2d_info_dict.strideH, conv2d_info_dict.strideW};
    std::vector<int64_t> pads = {conv2d_info_dict.padTop, conv2d_info_dict.padBottom, conv2d_info_dict.padLeft,
                                 conv2d_info_dict.padRight};
    std::vector<int64_t> dilations = {1, 1, conv2d_info_dict.dilationH, conv2d_info_dict.dilationW};

    std::string op_type = "Conv2DV2";

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    string compile_info_string = R"({"hardware_info":
      {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
       "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
       "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
       "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
       "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
       "core_type_list": "CubeCore,VectorCore"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    map<string, string> soc_version_infos = {{"NpuArch", "3510"}};
    aicore_spec.insert({"fb0_size", "4096"});
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;
    compile_info.aicoreNum = AIC_NUM;
    compile_info.socVersion = "Ascend950PR_9589";
    compile_info.shortSocVersion = "Ascend950";

    auto tilingDataPtr = gert::TilingData::CreateCap(MEM_SIZE_1K);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(MEM_SIZE_1K);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shapes_ref)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, conv2d_info_dict.aDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, conv2d_info_dict.aDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(2, conv2d_info_dict.aDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, conv2d_info_dict.aDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, conv2d_info_dict.aDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(tilingDataPtr.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
}

static Conv2DTilingTestParamRepo general_cases_params_repo[] = {
    {"Conv2D_repo_test_0", "Conv2DV2",
     R"({"groups":1,"orgCi":1280,"orgHi":32,"orgWi":32,"orgCo":1280,"orgHo":32,"orgWo":32,"kernelH":3,"kernelW":3,
    "singleCoreCi":1280,"singleCoreCo":80,"singleCoreHo":16,"singleCoreWo":32,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":1,"padBottom":1,"padLeft":1,"padRight":1,"biasFullLoadFlag":1,"fixpParamsFullLoadFlag":1,
    "offsetx":0,"hf32Enable":0,"hf32TransMode":0,"isC04Flag":0,"roundMode":1,"batchDim":1,"hoDim":2,"woDim":1,"nDim":16,
    "groupDim":1,"mMode":0,"woL1":32,"woL0":32,"kAL1":576,"kBL1":576,"hoL1":16,"nBL1":80,"kL0":32,"nL0":80,"hoL0":16,
    "iterateMNOrder":1,"pBufferFlag":27})",
     R"({"aDtype":1,"bDtype":1, "cDtype":1,"biasDtype":1,"aShapeN":1,
    "aShapeH":32,"aShapeW":32,"bShapeN":1280,"bShapeC":1280,"bShapeH":3,"bShapeW":3,
    "cShapeH":32,"cShapeW":32,"aFormat":0,"bFormat":0,"cFormat":0,"groups":1,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":1,"padBottom":1,"padLeft":1,"padRight":1,"biasFlag":true,"hf32Flag":false,
    "reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0,
    "reserverdParam6": 0})"},
    {"Conv2D_repo_test_1_no_bias", "Conv2DV2",
     R"({"groups":1,"orgCi":64,"orgHi":56,"orgWi":56,"orgCo":128,"orgHo":56,"orgWo":56,"kernelH":1,"kernelW":1,
    "singleCoreCi":64,"singleCoreCo":128,"singleCoreHo":56,"singleCoreWo":56,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":0,"padBottom":0,"padLeft":0,"padRight":0,"biasFullLoadFlag":1,"fixpParamsFullLoadFlag":1,
    "offsetx":0,"hf32Enable":0,"hf32TransMode":0,"isC04Flag":0,"roundMode":0,"batchDim":1,"hoDim":1,"woDim":1,"nDim":1,
    "groupDim":1,"mMode":0,"woL1":56,"woL0":56,"kAL1":64,"kBL1":64,"hoL1":56,"nBL1":128,"kL0":64,"nL0":128,"hoL0":56,
    "iterateMNOrder":1,"pBufferFlag":25})",
     R"({"aDtype":0,"bDtype":0, "cDtype":0,"biasDtype":0,"aShapeN":1,
    "aShapeH":56,"aShapeW":56,"bShapeN":128,"bShapeC":64,"bShapeH":1,"bShapeW":1,
    "cShapeH":56,"cShapeW":56,"aFormat":0,"bFormat":0,"cFormat":0,"groups":1,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":0,"padBottom":0,"padLeft":0,"padRight":0,"biasFlag":false,"hf32Flag":false,
    "reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0,
    "reserverdParam6": 0})"},
    {"Conv2D_repo_test_2_fp32_hf32", "Conv2DV2",
     R"({"groups":1,"orgCi":960,"orgHi":1,"orgWi":16,"orgCo":64,"orgHo":1,"orgWo":16,"kernelH":1,"kernelW":1,
    "singleCoreCi":960,"singleCoreCo":64,"singleCoreHo":1,"singleCoreWo":16,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":0,"padBottom":0,"padLeft":0,"padRight":0,"biasFullLoadFlag":0,"fixpParamsFullLoadFlag":0,
    "offsetx":0,"hf32Enable":1,"hf32TransMode":0,"isC04Flag":0,"roundMode":0,"batchDim":1,"hoDim":1,"woDim":1,"nDim":1,
    "groupDim":1,"mMode":0,"woL1":16,"woL0":16,"kAL1":960,"kBL1":960,"hoL1":1,"nBL1":64,"kL0":32,"nL0":64,"hoL0":1,
    "iterateMNOrder":1,"pBufferFlag":11})",
     R"({"aDtype":0,"bDtype":0, "cDtype":0,"biasDtype":0,"aShapeN":2048,
    "aShapeH":1,"aShapeW":16,"bShapeN":64,"bShapeC":960,"bShapeH":1,"bShapeW":1,
    "cShapeH":1,"cShapeW":16,"aFormat":0,"bFormat":0,"cFormat":0,"groups":1,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":0,"padBottom":0,"padLeft":0,"padRight":0,"biasFlag":true,"hf32Flag":true,
    "reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0,
    "reserverdParam6": 0})"}};

INSTANTIATE_TEST_CASE_P(Conv2DV2, Conv2DTilingRepo, testing::ValuesIn(general_cases_params_repo));

namespace {
// ----------------------------------------------------------------------------
// Runtime-KB translation coverage: the ascend960dt build ships no
// *_runtime_kb.json for Conv2DV2, so QueryBank never succeeds and
// TranslateRepoTiling is never entered through the normal tiling entry.
// The tests below drive the translation helpers directly with a tunner
// instance created by the tuning class factory.
// ----------------------------------------------------------------------------
using namespace optiling::conv_ops_tiling;

void InitRepoTilingState(Conv2dBaseTiling& sut, ConvTilingParseInfo& opInfo, tuningtiling::Conv2DV2TunnerTiling& tunner)
{
    opInfo.aicoreNum = AIC_NUM;
    opInfo.l1Size = MEM_SIZE_512K;
    opInfo.l0aSize = MEM_SIZE_64K;
    opInfo.l0bSize = MEM_SIZE_64K;
    opInfo.l0cSize = MEM_SIZE_256K;
    opInfo.ubSize = MEM_SIZE_200K;
    opInfo.btSize = MEM_SIZE_4K;
    opInfo.npuArch = NpuArch::DAV_9201;
    opInfo.socVersion = "Ascend960DT";
    sut.opInfo_ = &opInfo;

    sut.shapeInfo_.batch = 1;
    sut.shapeInfo_.ci = 16;
    sut.shapeInfo_.hi = 16;
    sut.shapeInfo_.wi = 16;
    sut.shapeInfo_.co = 16;
    sut.shapeInfo_.kh = 3;
    sut.shapeInfo_.kw = 3;
    sut.shapeInfo_.ho = 14;
    sut.shapeInfo_.wo = 14;
    sut.attrInfo_.groups = 1;
    sut.attrInfo_.strideH = 1;
    sut.attrInfo_.strideW = 1;
    sut.attrInfo_.dilationH = 1;
    sut.attrInfo_.dilationW = 1;
    sut.attrInfo_.roundMode = 1;
    sut.descInfo_.fMapDtype = ge::DT_FLOAT16;
    sut.descInfo_.weightDtype = ge::DT_FLOAT16;
    sut.descInfo_.biasDtype = ge::DT_FLOAT16;
    sut.descInfo_.outDtype = ge::DT_FLOAT16;
    sut.descInfo_.fMapFormat = ge::FORMAT_NCHW;
    sut.descInfo_.weightFormat = ge::FORMAT_NCHW;
    sut.descInfo_.outFormat = ge::FORMAT_NCHW;
    sut.flagInfo_.hasBias = true;
    sut.flagInfo_.convGroupType = ConvGroupType::NORMAL_CONV;
    sut.featureFlagInfo_ = ConvAscendcFeatureFlag::IS_LOAD3D_FLAG;
    sut.paramInfo_.nodeType = "Conv2DV2";
    sut.convOpsConstParams_.m0 = CUBE_M0;
    sut.convOpsConstParams_.k0 = CUBE_K0_16;
    sut.convOpsConstParams_.n0 = CUBE_N0;

    conv_tiling::PlatformInfo platform;
    platform.l1Size = MEM_SIZE_512K;
    platform.l0CSize = MEM_SIZE_256K;
    platform.ubSize = MEM_SIZE_200K;
    platform.l0ASize = MEM_SIZE_64K;
    platform.l0BSize = MEM_SIZE_64K;
    platform.btSize = MEM_SIZE_4K;
    platform.aivPerAic = NUM_2;
    platform.npuArch = NpuArch::DAV_9201;
    sut.conv2dApiTiling_ = conv_tiling::Conv2dTiling(platform);

    tunner.groups = 1;
    tunner.orgCi = 16;
    tunner.orgHi = 16;
    tunner.orgWi = 16;
    tunner.orgCo = 16;
    tunner.orgHo = 14;
    tunner.orgWo = 14;
    tunner.kernelH = 3;
    tunner.kernelW = 3;
    tunner.khL1 = 3;
    tunner.kwL1 = 3;
    tunner.singleCoreCi = 16;
    tunner.singleCoreCo = 16;
    tunner.singleCoreHo = 14;
    tunner.singleCoreWo = 14;
    tunner.hoL1 = 14;
    tunner.woL1 = 14;
    tunner.kAL1 = 144;
    tunner.kBL1 = 144;
    tunner.nBL1 = 16;
    tunner.multiNBL1 = 1;
    tunner.hoL0 = 16;
    tunner.woL0 = 16;
    tunner.kL0 = 16;
    tunner.nL0 = 16;
    tunner.nL1DivBlockSize = 1;
    tunner.pBufferFlag = 27;
    tunner.strideH = 1;
    tunner.strideW = 1;
    tunner.dilationH = 1;
    tunner.dilationW = 1;
    tunner.iterateMNOrder = 1;
    tunner.biasFullLoadFlag = 1;
    tunner.fixpParamsFullLoadFlag = 1;
    tunner.roundMode = 1;
    tunner.batchDim = 1;
    tunner.hoDim = 1;
    tunner.woDim = 1;
    tunner.nDim = 1;
    tunner.groupDim = 1;
    tunner.mMode = 0;
    tunner.innerBatch = 1;
    tunner.enlarge = 0;
    tunner.isC04Flag = 0;
    tunner.isWeightUbTransFlag = 0;
}
} // namespace

class Conv2DTilingRepoTranslate : public testing::Test {
protected:
    void SetUp() override
    {
        sut_ = std::make_shared<Conv2dBaseTiling>(nullptr);
        tiling_ = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance("Conv2DV2");
        ASSERT_NE(tiling_, nullptr);
        tunner_ = std::static_pointer_cast<tuningtiling::Conv2DV2TunnerTiling>(tiling_);
        ASSERT_NE(tunner_, nullptr);
        InitRepoTilingState(*sut_, opInfo_, *tunner_);
    }
    void TearDown() override {}

    std::shared_ptr<Conv2dBaseTiling> sut_;
    tuningtiling::TuningTilingDefPtr tiling_ = nullptr;
    std::shared_ptr<tuningtiling::Conv2DV2TunnerTiling> tunner_ = nullptr;
    ConvTilingParseInfo opInfo_;
};

TEST_F(Conv2DTilingRepoTranslate, translate_load3d_tiling)
{
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_EQ(sut_->tilingData_.get_groups(), 1u);
    EXPECT_EQ(sut_->tilingData_.get_orgCi(), 16u);
    EXPECT_EQ(sut_->tilingData_.get_singleCoreBatch(), 1u);
    EXPECT_EQ(sut_->tilingData_.get_kernelHxkernelW(), 9u);
    EXPECT_EQ(sut_->tilingData_.get_khL1(), 0u);
    EXPECT_EQ(sut_->tilingData_.get_bUbNStep(), 0u);
    EXPECT_EQ(sut_->tilingData_.get_khUb(), 0u);
    EXPECT_EQ(sut_->tilingData_.get_multiNBL1(), 1u);
    EXPECT_EQ(sut_->tilingData_.get_aL1SpaceSize() % MEM_SIZE_64B, 0u);
}

TEST_F(Conv2DTilingRepoTranslate, translate_dma_tiling)
{
    sut_->featureFlagInfo_ = ConvAscendcFeatureFlag::IS_DMA_FLAG;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_EQ(sut_->tilingData_.get_khL1(), 3u);
    EXPECT_EQ(sut_->tilingData_.get_kwL1(), 3u);
    EXPECT_GT(sut_->tilingData_.get_khUb(), 0u);
    EXPECT_GT(sut_->tilingData_.get_kwUb(), 0u);
}

TEST_F(Conv2DTilingRepoTranslate, translate_c04_mmode_inner_batch)
{
    tunner_->isC04Flag = 1;
    tunner_->mMode = 1;
    tunner_->innerBatch = 2;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_TRUE(sut_->flagInfo_.enableC04Flag);
    EXPECT_TRUE(sut_->flagInfo_.mSplitModeFlag);
    EXPECT_GT(sut_->tilingData_.get_bUbNStep(), 0u);
    EXPECT_EQ(sut_->tilingData_.get_bUbKStep(), 0u);
    EXPECT_GT(sut_->tilingData_.get_aL1SpaceSize(), 0u);
}

TEST_F(Conv2DTilingRepoTranslate, translate_c04_hw_wo_full_load)
{
    tunner_->isC04Flag = 1;
    tunner_->mMode = 0;
    tunner_->orgWo = 14;
    tunner_->woL1 = 14;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_TRUE(sut_->flagInfo_.enableC04Flag);
    EXPECT_FALSE(sut_->flagInfo_.mSplitModeFlag);
    EXPECT_GT(sut_->tilingData_.get_aL1SpaceSize(), 0u);
}

TEST_F(Conv2DTilingRepoTranslate, translate_weight_ub_trans)
{
    tunner_->isWeightUbTransFlag = 1;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_GT(sut_->tilingData_.get_bUbNStep(), 0u);
    EXPECT_GT(sut_->tilingData_.get_bUbKStep(), 0u);
}

TEST_F(Conv2DTilingRepoTranslate, translate_mmode_al1_space)
{
    tunner_->mMode = 1;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_TRUE(sut_->flagInfo_.mSplitModeFlag);
    EXPECT_GT(sut_->tilingData_.get_aL1SpaceSize(), 0u);
}

TEST_F(Conv2DTilingRepoTranslate, translate_opt_group_enlarge)
{
    sut_->flagInfo_.convGroupType = ConvGroupType::OPT_GROUP_CONV;
    sut_->oriGroupInfo_.ciPerGroup = 8;
    sut_->oriGroupInfo_.coPerGroup = 8;
    sut_->optGroupInfo_.enlarge = 2;
    sut_->optGroupInfo_.groupOpt = 1;
    tunner_->groups = 2;
    tunner_->enlarge = 2;
    tunner_->groupDim = 1;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_EQ(sut_->tilingData_.get_enlarge(), 2u);
    EXPECT_EQ(sut_->tilingData_.get_cinOpt(), 16u);
    EXPECT_EQ(sut_->tilingData_.get_coutOpt(), 16u);
    EXPECT_EQ(sut_->tilingData_.get_groupOpt(), 1u);
    EXPECT_EQ(sut_->tilingData_.get_singleCoreGroups(), 2u);
}

TEST_F(Conv2DTilingRepoTranslate, translate_null_tiling)
{
    tuningtiling::TuningTilingDefPtr nullTiling = nullptr;
    EXPECT_FALSE(sut_->TranslateRepoTiling(nullTiling));
}

TEST_F(Conv2DTilingRepoTranslate, get_tiling_input_args)
{
    sut_->attrInfo_.hf32Mode = 1;
    sut_->flagInfo_.extendConvFlag = true;
    sut_->fixpipeInfo_.dualOutput = 1;
    shared_ptr<void> inputArgs = nullptr;
    size_t inputArgsSize = 0;
    sut_->GetTilingInputArgs(inputArgs, inputArgsSize);
    ASSERT_NE(inputArgs, nullptr);
    EXPECT_EQ(inputArgsSize, sizeof(tuningtiling::Conv2DV2InputArgs));
    auto conv2dArgs = reinterpret_cast<tuningtiling::Conv2DV2InputArgs*>(inputArgs.get());
    EXPECT_TRUE(conv2dArgs->hf32Flag);
    EXPECT_EQ(conv2dArgs->groups, 1);
    EXPECT_EQ(conv2dArgs->bShapeC, 16);
}

TEST_F(Conv2DTilingRepoTranslate, cache_tiling_support_and_translate)
{
    EXPECT_TRUE(sut_->CheckSupportCacheTiling());
    sut_->paramInfo_.nodeType = "QuantConv2D";
    EXPECT_FALSE(sut_->CheckSupportCacheTiling());
    sut_->paramInfo_.nodeType = "Conv2DV2";

    sut_->GetCacheTilingInputArgs();
    EXPECT_EQ(sut_->cacheInputArgs_.inputShapeN, 1);
    EXPECT_EQ(sut_->cacheInputArgs_.weightShapeN, 16);

    sut_->GetCachedTilingData();
    sut_->TranslateCachedTilingData();
    EXPECT_EQ(sut_->tilingData_.get_orgCi(), 16u);
    EXPECT_EQ(sut_->tilingData_.get_orgHi(), 16u);

    Conv2dTilingCache::GetInstance().cachedTiling.clear();
    ASSERT_TRUE(sut_->AddTilingToCache());
    Conv2dCacheTilingData cachedData;
    EXPECT_TRUE(Conv2dTilingCache::GetInstance().GetCachedTiling(sut_->cacheInputArgs_, cachedData));
    Conv2dTilingCache::GetInstance().cachedTiling.clear();
}

TEST_F(Conv2DTilingRepoTranslate, cache_tiling_full)
{
    Conv2dTilingCache::GetInstance().cachedTiling.clear();
    ConvInputArgs args;
    for (uint64_t idx = 0; idx < MAX_CACHE_SIZE; ++idx) {
        args.inputShapeN = static_cast<int64_t>(idx);
        ASSERT_TRUE(Conv2dTilingCache::GetInstance().AddCachedTiling(args, Conv2dCacheTilingData()));
    }
    EXPECT_FALSE(sut_->AddTilingToCache());
    Conv2dTilingCache::GetInstance().cachedTiling.clear();
}
