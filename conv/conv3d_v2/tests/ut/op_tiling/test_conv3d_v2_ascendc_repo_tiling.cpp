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
 * \file test_conv3d_v2_ascendc_repo_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "log/log.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "tests/ut/common/kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "../../../op_host/op_tiling/arch35/conv3d_v2_tuning_tiling.h"
#include "../../../op_host/op_tiling/arch35/conv3d_v2_base_tiling.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base.h"
#include "../../../op_host/op_tiling/conv3d_base_tiling.h"
#include "../../../op_host/op_tiling/conv3d_tuning_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t*>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
}

struct Conv3DTilingTestParamRepo {
    string case_name;
    string op_type;
    string info_dict;
    string tiling_data;
};
} // namespace

class Conv3DTilingRepo : public testing::TestWithParam<Conv3DTilingTestParamRepo> {
protected:
    void SetUp() {}
    void TearDown() {}
    static void TearDownTestCase() {}
    void PrepareTest(Conv3DTilingTestParamRepo& param)
    {
        PrepareKnowledgeShape(param);
        PrepareKnowledgeTiling(param);
        PrepareInfoDict(param);
    }

    void PrepareKnowledgeShape(Conv3DTilingTestParamRepo& param)
    {
        auto j = nlohmann::json::parse(param.tiling_data);
        conv3d_knowledge.groups = j["groups"];
        conv3d_knowledge.singleCoreCi = j["singleCoreCi"]; // wyh
        conv3d_knowledge.singleCoreDo = j["singleCoreDo"];
        conv3d_knowledge.singleCoreCo = j["singleCoreCo"];
        conv3d_knowledge.singleCoreHo = j["singleCoreHo"];
        conv3d_knowledge.singleCoreWo = j["singleCoreWo"];
        conv3d_knowledge.orgDo = j["orgDo"];
        conv3d_knowledge.orgCo = j["orgCo"];
        conv3d_knowledge.orgHo = j["orgHo"];
        conv3d_knowledge.orgWo = j["orgWo"];
        conv3d_knowledge.orgCi = j["orgCi"];
        conv3d_knowledge.orgDi = j["orgDi"];
        conv3d_knowledge.orgHi = j["orgHi"];
        conv3d_knowledge.orgWi = j["orgWi"];
        conv3d_knowledge.kernelD = j["kernelD"];
        conv3d_knowledge.kernelH = j["kernelH"];
        conv3d_knowledge.kernelW = j["kernelW"];
        conv3d_knowledge.strideD = j["strideD"];
        conv3d_knowledge.strideH = j["strideH"];
        conv3d_knowledge.strideW = j["strideW"];
        conv3d_knowledge.dilationD = j["dilationD"];
        conv3d_knowledge.dilationH = j["dilationH"];
        conv3d_knowledge.dilationW = j["dilationW"];
        conv3d_knowledge.padHead = j["padHead"];
        conv3d_knowledge.padTail = j["padTail"];
        conv3d_knowledge.padTop = j["padTop"];
        conv3d_knowledge.padBottom = j["padBottom"];
        conv3d_knowledge.padLeft = j["padLeft"];
        conv3d_knowledge.padRight = j["padRight"];
    }

    void PrepareKnowledgeTiling(Conv3DTilingTestParamRepo& param)
    {
        auto j = nlohmann::json::parse(param.tiling_data);
        conv3d_knowledge.hoL0 = j["hoL0"];
        conv3d_knowledge.woL0 = j["woL0"];
        conv3d_knowledge.kL0 = j["kL0"];
        conv3d_knowledge.nL0 = j["nL0"];
        conv3d_knowledge.kAL1 = j["kAL1"];
        conv3d_knowledge.kBL1 = j["kBL1"];
        conv3d_knowledge.nBL1 = j["nBL1"];
        conv3d_knowledge.hoL1 = j["hoL1"];
        conv3d_knowledge.woL1 = j["woL1"];
        conv3d_knowledge.pBufferFlag = j["pBufferFlag"];
        conv3d_knowledge.iterateMNOrder = j["iterateMNOrder"];
        conv3d_knowledge.biasFullLoadFlag = j["biasFullLoadFlag"];
        conv3d_knowledge.fixpParamsFullLoadFlag = j["fixpParamsFullLoadFlag"];
        conv3d_knowledge.hf32Enable = j["hf32Enable"];
        conv3d_knowledge.hf32TransMode = j["hf32TransMode"];
        conv3d_knowledge.batchDim = j["batchDim"];
        conv3d_knowledge.nDim = j["nDim"];
        conv3d_knowledge.hoDim = j["hoDim"];
        conv3d_knowledge.doDim = j["doDim"];
        conv3d_knowledge.groupDim = j["groupDim"];
        conv3d_knowledge.isC04Flag = j["isC04Flag"];
        conv3d_knowledge.mMode = j["mMode"];
        // wyh
        conv3d_knowledge.bl1FullLoad = j["bl1FullLoad"];
        conv3d_knowledge.al1FullLoad = j["al1FullLoad"];
        conv3d_knowledge.bl1BypassFlag = j["bl1BypassFlag"];
    }

    void PrepareInfoDict(Conv3DTilingTestParamRepo& param)
    {
        auto j = nlohmann::json::parse(param.info_dict);
        conv3d_info_dict.aDtype = j["aDtype"];
        conv3d_info_dict.bDtype = j["bDtype"];
        conv3d_info_dict.cDtype = j["cDtype"];
        conv3d_info_dict.biasDtype = j["biasDtype"];
        conv3d_info_dict.aShapeN = j["aShapeN"];
        conv3d_info_dict.aShapeD = j["aShapeD"];
        conv3d_info_dict.aShapeH = j["aShapeH"];
        conv3d_info_dict.aShapeW = j["aShapeW"];
        conv3d_info_dict.bShapeN = j["bShapeN"];
        conv3d_info_dict.bShapeC = j["bShapeC"];
        conv3d_info_dict.bShapeD = j["bShapeD"];
        conv3d_info_dict.bShapeH = j["bShapeH"];
        conv3d_info_dict.bShapeW = j["bShapeW"];
        conv3d_info_dict.cShapeD = j["cShapeD"];
        conv3d_info_dict.cShapeH = j["cShapeH"];
        conv3d_info_dict.cShapeW = j["cShapeW"];
        conv3d_info_dict.aFormat = j["aFormat"];
        conv3d_info_dict.bFormat = j["bFormat"];
        conv3d_info_dict.cFormat = j["cFormat"];
        conv3d_info_dict.groups = j["groups"];
        conv3d_info_dict.strideD = j["strideD"];
        conv3d_info_dict.strideH = j["strideH"];
        conv3d_info_dict.strideW = j["strideW"];
        conv3d_info_dict.dilationD = j["dilationD"];
        conv3d_info_dict.dilationH = j["dilationH"];
        conv3d_info_dict.dilationW = j["dilationW"];
        conv3d_info_dict.padHead = j["padHead"];
        conv3d_info_dict.padTail = j["padTail"];
        conv3d_info_dict.padTop = j["padTop"];
        conv3d_info_dict.padBottom = j["padBottom"];
        conv3d_info_dict.padLeft = j["padLeft"];
        conv3d_info_dict.padRight = j["padRight"];
        conv3d_info_dict.biasFlag = j["biasFlag"];
    }

    std::string bank_path;
    const char* optype = "Conv3DV2";
    gert::TilingContext* context = nullptr;
    tuningtiling::Conv3DV2TunnerTiling conv3d_knowledge;
    tuningtiling::Conv3DV2InputArgs conv3d_info_dict;
};

TEST_P(Conv3DTilingRepo, general_cases_001)
{
    Conv3DTilingTestParamRepo param = GetParam();
    PrepareTest(param);
    tuningtiling::TuningTilingDefPtr tiling = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance(optype);
    nlohmann::json a;
    conv3d_knowledge.ToJson(a);
    tiling->FromJson(a);

    gert::StorageShape featuremap = {{conv3d_info_dict.aShapeN, conv3d_info_dict.bShapeC, conv3d_info_dict.aShapeD,
                                      conv3d_info_dict.aShapeH, conv3d_info_dict.aShapeW},
                                     {conv3d_info_dict.aShapeN, conv3d_info_dict.bShapeC, conv3d_info_dict.aShapeD,
                                      conv3d_info_dict.aShapeH, conv3d_info_dict.aShapeW}};
    gert::StorageShape weight = {{conv3d_info_dict.bShapeN, conv3d_info_dict.bShapeC, conv3d_info_dict.bShapeD,
                                  conv3d_info_dict.bShapeH, conv3d_info_dict.bShapeW},
                                 {conv3d_info_dict.bShapeN, conv3d_info_dict.bShapeC, conv3d_info_dict.bShapeD,
                                  conv3d_info_dict.bShapeH, conv3d_info_dict.bShapeW}};
    gert::StorageShape bias = {{conv3d_info_dict.bShapeN}, {conv3d_info_dict.bShapeN}};
    gert::StorageShape offset_w;
    gert::StorageShape output = {{conv3d_info_dict.aShapeN, conv3d_info_dict.bShapeN, conv3d_info_dict.cShapeD,
                                  conv3d_info_dict.cShapeH, conv3d_info_dict.cShapeW},
                                 {conv3d_info_dict.aShapeN, conv3d_info_dict.bShapeN, conv3d_info_dict.cShapeD,
                                  conv3d_info_dict.cShapeH, conv3d_info_dict.cShapeW}};
    std::vector<void*> input_shape_ref;

    bool hasBias = conv3d_info_dict.biasFlag;
    if (hasBias) {
        input_shape_ref = {&featuremap, &weight, &bias};
    } else {
        input_shape_ref = {&featuremap, &weight, nullptr};
    }

    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides_ref = {1, 1, conv3d_info_dict.strideD, conv3d_info_dict.strideH,
                                        conv3d_info_dict.strideW};
    std::vector<int64_t> pads_ref = {conv3d_info_dict.padHead,   conv3d_info_dict.padTail, conv3d_info_dict.padTop,
                                     conv3d_info_dict.padBottom, conv3d_info_dict.padLeft, conv3d_info_dict.padRight};
    std::vector<int64_t> dilations_ref = {1, 1, conv3d_info_dict.dilationD, conv3d_info_dict.dilationH,
                                          conv3d_info_dict.dilationW};

    std::string op_type = "Conv3DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    uint32_t aicoreNum = 32;
    string compile_info_string = R"({"hardware_info":
      {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub":
      true,
       "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
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
    compile_info.tilingType = op_type;
    compile_info.aicoreNum = aicoreNum;
    compile_info.socVersion = "Ascend950PR_9589";
    compile_info.shortSocVersion = "Ascend950";
    auto tilingDataPtr = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shapes_ref)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides_ref)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads_ref)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations_ref)},
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

static Conv3DTilingTestParamRepo general_cases_params_repo[] = {
    {"Conv3DV2_repo_test_0", "Conv3DV2",
     R"({"aDtype": 27, "bDtype": 27, "cDtype": 27, "biasDtype": 27, "aShapeN": 1, "aShapeD": 243, "aShapeH": 542,
     "aShapeW": 62, "bShapeN": 256, "bShapeC": 256,
"bShapeD": 3, "bShapeH": 3, "bShapeW": 3, "cShapeD": 241, "cShapeH": 540, "cShapeW": 60, "aFormat": 30, "bFormat":
30, "cFormat": 30, "groups": 1, "strideD": 1, "strideH": 1, "strideW": 1, "dilationD": 1, "dilationH": 1,
"dilationW": 1, "padHead": 0, "padTail": 0, "padTop": 0, "padBottom": 0, "padLeft": 0, "padRight": 0, "biasFlag":
true, "reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0,
"reserverdParam6": 0})",
     R"({"groups": 1, "singleCoreDo": 241, "singleCoreCo": 128, "singleCoreHo": 2025, "singleCoreWo": 0,
     "singleCoreCi": 256,
"orgDo": 241, "orgCo": 256, "orgHo": 540, "orgWo": 60, "orgCi": 256, "orgDi": 243, "orgHi": 542, "orgWi": 62,
"kernelD": 3, "kernelH": 3, "kernelW": 3, "strideD": 1, "strideH": 1, "strideW": 1, "dilationD": 1, "dilationH": 1,
"dilationW": 1, "padHead": 0, "padTail": 0, "padTop": 0, "padBottom": 0, "padLeft": 0, "padRight": 0, "hoL0": 512,
"woL0": 0, "kL0": 32, "nL0": 128, "kAL1": 288, "kBL1": 288, "nBL1": 128, "hoL1": 512, "woL1": 0, "pBufferFlag": 27,
"bl1FullLoad": 0, "al1FullLoad": 0, "bl1BypassFlag": 0, "iterateMNOrder": 0, "biasFullLoadFlag": 1,
"fixpParamsFullLoadFlag": 1, "hf32Enable": 0,
 "hf32TransMode": 0, "batchDim": 1, "nDim": 2, "hoDim": 16, "doDim": 1, "groupDim": 1, "isC04Flag": 0, "mMode":
 1})"},
    {"Conv3DV2_repo_test_1_small_1x1x1", "Conv3DV2",
     R"({"aDtype": 1, "bDtype": 1, "cDtype": 1, "biasDtype": 1, "aShapeN": 1, "aShapeD": 8, "aShapeH": 16, "aShapeW":
     16, "bShapeN": 16, "bShapeC": 8,
"bShapeD": 1, "bShapeH": 1, "bShapeW": 1, "cShapeD": 8, "cShapeH": 16, "cShapeW": 16, "aFormat": 30, "bFormat": 30,
"cFormat": 30, "groups": 1, "strideD": 1, "strideH": 1, "strideW": 1, "dilationD": 1, "dilationH": 1, "dilationW": 1,
"padHead": 0, "padTail": 0, "padTop": 0, "padBottom": 0, "padLeft": 0, "padRight": 0, "biasFlag": false,
"reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0,
"reserverdParam6": 0})",
     R"({"groups": 1, "singleCoreDo": 8, "singleCoreCo": 16, "singleCoreHo": 16, "singleCoreWo": 16, "singleCoreCi":
     8,
"orgDo": 8, "orgCo": 16, "orgHo": 16, "orgWo": 16, "orgCi": 8, "orgDi": 8, "orgHi": 16, "orgWi": 16, "kernelD": 1,
"kernelH": 1, "kernelW": 1, "strideD": 1, "strideH": 1, "strideW": 1, "dilationD": 1, "dilationH": 1, "dilationW": 1,
"padHead": 0, "padTail": 0, "padTop": 0, "padBottom": 0, "padLeft": 0, "padRight": 0, "hoL0": 16, "woL0": 16, "kL0":
8, "nL0": 16, "kAL1": 8, "kBL1": 8, "nBL1": 16, "hoL1": 16, "woL1": 16, "pBufferFlag": 11, "bl1FullLoad": 0,
"al1FullLoad": 0, "bl1BypassFlag": 0, "iterateMNOrder": 1, "biasFullLoadFlag": 1, "fixpParamsFullLoadFlag": 1,
"hf32Enable": 0,
 "hf32TransMode": 0, "batchDim": 1, "nDim": 1, "hoDim": 1, "doDim": 1, "groupDim": 1, "isC04Flag": 0, "mMode": 0})"},
    {"Conv3DV2_repo_test_2_fp32", "Conv3DV2",
     R"({"aDtype": 0, "bDtype": 0, "cDtype": 0, "biasDtype": 0, "aShapeN": 1, "aShapeD": 4, "aShapeH": 32, "aShapeW":
     32, "bShapeN": 16, "bShapeC": 4,
"bShapeD": 3, "bShapeH": 3, "bShapeW": 3, "cShapeD": 2, "cShapeH": 30, "cShapeW": 30, "aFormat": 30, "bFormat": 30,
"cFormat": 30, "groups": 1, "strideD": 1, "strideH": 1, "strideW": 1, "dilationD": 1, "dilationH": 1, "dilationW": 1,
"padHead": 0, "padTail": 0, "padTop": 0, "padBottom": 0, "padLeft": 0, "padRight": 0, "biasFlag": true,
"reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0,
"reserverdParam6": 0})",
     R"({"groups": 1, "singleCoreDo": 2, "singleCoreCo": 16, "singleCoreHo": 30, "singleCoreWo": 30, "singleCoreCi":
     4,
"orgDo": 2, "orgCo": 16, "orgHo": 30, "orgWo": 30, "orgCi": 4, "orgDi": 4, "orgHi": 32, "orgWi": 32, "kernelD": 3,
"kernelH": 3, "kernelW": 3, "strideD": 1, "strideH": 1, "strideW": 1, "dilationD": 1, "dilationH": 1, "dilationW": 1,
"padHead": 0, "padTail": 0, "padTop": 0, "padBottom": 0, "padLeft": 0, "padRight": 0, "hoL0": 30, "woL0": 30, "kL0":
4, "nL0": 16, "kAL1": 36, "kBL1": 36, "nBL1": 16, "hoL1": 30, "woL1": 30, "pBufferFlag": 0, "bl1FullLoad": 0,
"al1FullLoad": 0, "bl1BypassFlag": 0, "iterateMNOrder": 1, "biasFullLoadFlag": 1, "fixpParamsFullLoadFlag": 1,
"hf32Enable": 0,
 "hf32TransMode": 0, "batchDim": 1, "nDim": 1, "hoDim": 1, "doDim": 1, "groupDim": 1, "isC04Flag": 0, "mMode":
 0})"}};

INSTANTIATE_TEST_CASE_P(Conv3DV2, Conv3DTilingRepo, testing::ValuesIn(general_cases_params_repo));

namespace {
// ----------------------------------------------------------------------------
// Runtime-KB translation coverage. The ascend960dt build has no Conv3DV2
// runtime-kb bank, so QueryBank never succeeds and TranslateRepoTiling is not
// entered through the normal tiling entry. Drive the helpers directly.
// ----------------------------------------------------------------------------
using namespace optiling::conv_ops_tiling;

void InitRepoTilingState3D(Conv3dBaseTilingV2& sut, ConvTilingParseInfo& opInfo,
                           tuningtiling::Conv3DV2TunnerTiling& tunner)
{
    opInfo.aicoreNum = 32;
    opInfo.l1Size = 524288;
    opInfo.l0aSize = 65536;
    opInfo.l0bSize = 65536;
    opInfo.l0cSize = 262144;
    opInfo.ubSize = 196608;
    opInfo.btSize = 4096;
    opInfo.npuArch = NpuArch::DAV_9201;
    opInfo.socVersion = "Ascend960DT";
    sut.opInfo_ = &opInfo;

    sut.shapeInfo_.batch = 1;
    sut.shapeInfo_.ci = 16;
    sut.shapeInfo_.di = 4;
    sut.shapeInfo_.hi = 16;
    sut.shapeInfo_.wi = 16;
    sut.shapeInfo_.co = 16;
    sut.shapeInfo_.kd = 3;
    sut.shapeInfo_.kh = 3;
    sut.shapeInfo_.kw = 3;
    sut.shapeInfo_.dout = 2;
    sut.shapeInfo_.ho = 14;
    sut.shapeInfo_.wo = 14;
    sut.attrInfo_.groups = 1;
    sut.attrInfo_.strideD = 1;
    sut.attrInfo_.strideH = 1;
    sut.attrInfo_.strideW = 1;
    sut.attrInfo_.dilationD = 1;
    sut.attrInfo_.dilationH = 1;
    sut.attrInfo_.dilationW = 1;
    sut.attrInfo_.roundMode = 1;
    sut.descInfo_.fMapDtype = ge::DT_FLOAT16;
    sut.descInfo_.weightDtype = ge::DT_FLOAT16;
    sut.descInfo_.biasDtype = ge::DT_FLOAT16;
    sut.descInfo_.outDtype = ge::DT_FLOAT16;
    sut.descInfo_.fMapFormat = ge::FORMAT_NCDHW;
    sut.descInfo_.weightFormat = ge::FORMAT_NCDHW;
    sut.descInfo_.outFormat = ge::FORMAT_NCDHW;
    sut.flagInfo_.hasBias = true;
    sut.flagInfo_.convGroupType = ConvGroupType::NORMAL_CONV;
    sut.paramInfo_.nodeType = "Conv3DV2";
    sut.convOpsConstParams_.m0 = 16;
    sut.convOpsConstParams_.k0 = 16;
    sut.convOpsConstParams_.n0 = 16;
    sut.apiInputPlatformInfo_.ubSize = 196608;
    sut.apiInputPlatformInfo_.npuArch = NpuArch::DAV_9201;

    conv_tiling::PlatformInfo platform;
    platform.l1Size = 524288;
    platform.l0CSize = 262144;
    platform.ubSize = 196608;
    platform.l0ASize = 65536;
    platform.l0BSize = 65536;
    platform.btSize = 4096;
    platform.aivPerAic = 2;
    platform.npuArch = NpuArch::DAV_9201;
    sut.conv3dApiTiling_ = conv_tiling::Conv3dTiling(platform);

    tunner.groups = 1;
    tunner.orgCi = 16;
    tunner.orgDi = 4;
    tunner.orgHi = 16;
    tunner.orgWi = 16;
    tunner.orgDo = 2;
    tunner.orgCo = 16;
    tunner.orgHo = 14;
    tunner.orgWo = 14;
    tunner.kernelD = 3;
    tunner.kernelH = 3;
    tunner.kernelW = 3;
    tunner.khL1 = 3;
    tunner.kwL1 = 3;
    tunner.singleCoreCi = 16;
    tunner.singleCoreCo = 16;
    tunner.singleCoreHo = 14;
    tunner.singleCoreWo = 14;
    tunner.singleCoreDo = 2;
    tunner.hoL1 = 14;
    tunner.woL1 = 14;
    tunner.kAL1 = 144;
    tunner.kBL1 = 144;
    tunner.nBL1 = 16;
    tunner.hoL0 = 16;
    tunner.woL0 = 16;
    tunner.kL0 = 16;
    tunner.nL0 = 16;
    tunner.pBufferFlag = 27;
    tunner.strideD = 1;
    tunner.strideH = 1;
    tunner.strideW = 1;
    tunner.dilationD = 1;
    tunner.dilationH = 1;
    tunner.dilationW = 1;
    tunner.iterateMNOrder = 1;
    tunner.biasFullLoadFlag = 1;
    tunner.fixpParamsFullLoadFlag = 1;
    tunner.batchDim = 1;
    tunner.nDim = 1;
    tunner.hoDim = 1;
    tunner.doDim = 1;
    tunner.groupDim = 1;
    tunner.isC04Flag = 0;
    tunner.mMode = 0;
}
} // namespace

class Conv3DTilingRepoTranslate : public testing::Test {
protected:
    void SetUp() override
    {
        sut_ = std::make_shared<Conv3dBaseTilingV2>(nullptr);
        tiling_ = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance("Conv3DV2");
        ASSERT_NE(tiling_, nullptr);
        tunner_ = std::static_pointer_cast<tuningtiling::Conv3DV2TunnerTiling>(tiling_);
        ASSERT_NE(tunner_, nullptr);
        InitRepoTilingState3D(*sut_, opInfo_, *tunner_);
    }
    void TearDown() override {}

    std::shared_ptr<Conv3dBaseTilingV2> sut_;
    tuningtiling::TuningTilingDefPtr tiling_ = nullptr;
    std::shared_ptr<tuningtiling::Conv3DV2TunnerTiling> tunner_ = nullptr;
    ConvTilingParseInfo opInfo_;
};

TEST_F(Conv3DTilingRepoTranslate, translate_hw_mode_tiling)
{
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_EQ(sut_->tilingData_.groups, 1u);
    EXPECT_EQ(sut_->tilingData_.orgCi, 16u);
    EXPECT_EQ(sut_->tilingData_.kernelD, 3u);
    EXPECT_EQ(sut_->tilingData_.singleCoreBatch, 1u);
    EXPECT_EQ(sut_->tilingData_.kernelHxkernelW, 9u);
    EXPECT_EQ(sut_->tilingData_.kernelHxkernelWxkernelD, 27u);
    EXPECT_EQ(sut_->tilingData_.cinAInCore, 16u);
    EXPECT_EQ(sut_->tilingData_.mStep, 256u);
    EXPECT_EQ(sut_->tilingData_.kStep, 1u);
    EXPECT_EQ(sut_->tilingData_.nStep, 1u);
    EXPECT_EQ(sut_->tilingData_.nL1DivBlockSize, 1u);
    EXPECT_GT(sut_->tilingData_.aL1SpaceSize, 0u);
    EXPECT_FALSE(sut_->flagInfo_.mSplitModeFlag);
    EXPECT_FALSE(sut_->flagInfo_.isKernelSplit);
}

TEST_F(Conv3DTilingRepoTranslate, translate_m_mode_tiling)
{
    tunner_->mMode = 1;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_TRUE(sut_->flagInfo_.mSplitModeFlag);
    EXPECT_EQ(sut_->tilingData_.mStep, 16u);
    EXPECT_GT(sut_->tilingData_.aL1SpaceSize, 0u);
}

TEST_F(Conv3DTilingRepoTranslate, translate_kernel_split)
{
    tunner_->khL1 = 1;
    tunner_->kwL1 = 1;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_TRUE(sut_->flagInfo_.isKernelSplit);
    EXPECT_EQ(sut_->tilingData_.cinAInCore, 144u);
    EXPECT_EQ(sut_->tilingData_.khL1, 1u);
    EXPECT_EQ(sut_->tilingData_.kwL1, 1u);
}

TEST_F(Conv3DTilingRepoTranslate, translate_kernel_value_zero)
{
    tunner_->kernelH = 0;
    tunner_->kernelW = 0;
    EXPECT_FALSE(sut_->TranslateRepoTiling(tiling_));
}

TEST_F(Conv3DTilingRepoTranslate, translate_c04_wo_full_load)
{
    tunner_->isC04Flag = 1;
    tunner_->singleCoreWo = 14;
    tunner_->woL1 = 14;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_GT(sut_->tilingData_.aL1SpaceSize, 0u);
    EXPECT_EQ(sut_->tilingData_.aL1SpaceSize % 64u, 0u);
}

TEST_F(Conv3DTilingRepoTranslate, translate_opt_group)
{
    sut_->attrInfo_.groups = 2;
    sut_->shapeInfo_.ci = 32;
    sut_->shapeInfo_.co = 32;
    tunner_->groups = 2;
    tunner_->orgCi = 32;
    tunner_->orgCo = 32;
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    EXPECT_EQ(sut_->tilingData_.singleCoreGroups, sut_->optGroupInfo_.enlarge);
}

TEST_F(Conv3DTilingRepoTranslate, translate_null_tiling)
{
    tuningtiling::TuningTilingDefPtr nullTiling = nullptr;
    EXPECT_FALSE(sut_->TranslateRepoTiling(nullTiling));
}

TEST_F(Conv3DTilingRepoTranslate, get_tiling_input_args)
{
    sut_->attrInfo_.hf32Mode = 1;
    shared_ptr<void> inputArgs = nullptr;
    size_t inputArgsSize = 0;
    sut_->GetTilingInputArgs(inputArgs, inputArgsSize);
    ASSERT_NE(inputArgs, nullptr);
    EXPECT_EQ(inputArgsSize, sizeof(tuningtiling::Conv3DV2InputArgs));
    auto conv3dArgs = reinterpret_cast<tuningtiling::Conv3DV2InputArgs*>(inputArgs.get());
    EXPECT_EQ(conv3dArgs->groups, 1);
    EXPECT_EQ(conv3dArgs->bShapeC, 16);
    EXPECT_EQ(conv3dArgs->aShapeD, 4);
}

TEST_F(Conv3DTilingRepoTranslate, cache_tiling_support_and_translate)
{
    EXPECT_TRUE(sut_->CheckSupportCacheTiling());
    sut_->paramInfo_.nodeType = "QuantConv3D";
    EXPECT_FALSE(sut_->CheckSupportCacheTiling());
    sut_->paramInfo_.nodeType = "Conv3DV2";

    sut_->GetCacheTilingInputArgs();
    EXPECT_EQ(sut_->cacheInputArgs_.inputShapeD, 4);
    ASSERT_TRUE(sut_->TranslateRepoTiling(tiling_));
    sut_->GetCachedTilingData();
    sut_->tilingData_ = {};
    sut_->TranslateCachedTilingData();
    EXPECT_EQ(sut_->tilingData_.orgCi, 16u);
    EXPECT_EQ(sut_->tilingData_.kernelH, 3u);
}

// ----------------------------------------------------------------------------
// Legacy (DAV_2201) AOE knowledge-bank translation helpers are not reached by
// the ascend960dt tiling entry; drive them directly with a v1 tunner instance.
// ----------------------------------------------------------------------------
TEST_F(Conv3DTilingRepoTranslate, legacy_aoe_translate)
{
    auto def = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance(ge::AscendString("Conv3D"));
    ASSERT_NE(def, nullptr);
    auto tuner = std::static_pointer_cast<tuningtiling::Conv3DTunerTiling>(def);
    ASSERT_NE(tuner, nullptr);
    tuner->groups = 1;
    tuner->orgCi = 16;
    tuner->orgDi = 4;
    tuner->orgHi = 16;
    tuner->orgWi = 16;
    tuner->orgCo = 16;
    tuner->orgDo = 2;
    tuner->orgHo = 14;
    tuner->orgWo = 14;
    tuner->kernelD = 3;
    tuner->kernelH = 3;
    tuner->kernelW = 3;
    tuner->batchDim = 1;
    tuner->mDim = 1;
    tuner->nDim = 1;
    tuner->doDim = 1;
    tuner->groupDim = 1;

    optiling::Conv3dOpsTiling::Conv3dBaseTiling legacy(nullptr);
    legacy.shapeInfo_.batch = 1;
    legacy.flagInfo_.hasBias = true;
    ASSERT_TRUE(legacy.TranslateAoeTiling(def));
    EXPECT_EQ(legacy.tilingData_.convApiTiling.orgCi, 16u);
    EXPECT_EQ(legacy.tilingData_.convApiTiling.kernelD, 3u);
    EXPECT_EQ(legacy.tilingData_.convRunInfo.batch, 1u);
    EXPECT_EQ(legacy.tilingData_.convRunInfo.cout, 16u);

    tuningtiling::TuningTilingDefPtr nullTiling = nullptr;
    EXPECT_FALSE(legacy.TranslateAoeTiling(nullTiling));
}
