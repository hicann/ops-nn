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

using namespace ut_util;
using namespace std;
using namespace ge;

extern std::string GetTestSuiteName();
extern std::string GetTestCaseName();

static string TilingData2Str(const gert::TilingData *tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
    result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
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

class Conv3DTilingRepo: public testing::TestWithParam<Conv3DTilingTestParamRepo> {
protected:
    void SetUp() {}

    void TearDown() {}
    static void TearDownTestCase() {}
    void PrepareTest(Conv3DTilingTestParamRepo &param) {
        std::cout << "knowledge" << std::endl;
        PrepareKnowledge(param);
        std::cout << "info dict" << std::endl;
        PrepareInfoDict(param);
    }
    void PrepareKnowledge(Conv3DTilingTestParamRepo &param) {
        std::cout << param.tiling_data << std::endl;
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
        conv3d_knowledge.padHead= j["padHead"];
        conv3d_knowledge.padTail= j["padTail"];
        conv3d_knowledge.padTop= j["padTop"];
        conv3d_knowledge.padBottom= j["padBottom"];
        conv3d_knowledge.padLeft= j["padLeft"];
        conv3d_knowledge.padRight= j["padRight"];
        conv3d_knowledge.hoL0 = j["hoL0"];
        conv3d_knowledge.woL0 = j["woL0"];
        conv3d_knowledge.kL0 = j["kL0"];
        conv3d_knowledge.nL0 = j["nL0"];
        conv3d_knowledge.kAL1 = j["kAL1"];
        conv3d_knowledge.kBL1 = j["kBL1"];
        conv3d_knowledge.nBL1 = j["nBL1"];
        conv3d_knowledge.hoL1 = j["hoL1"];
        conv3d_knowledge.woL1 = j["woL1"];
        conv3d_knowledge.pBufferFlag= j["pBufferFlag"];
        conv3d_knowledge.iterateMNOrder= j["iterateMNOrder"];
        conv3d_knowledge.biasFullLoadFlag= j["biasFullLoadFlag"];
        conv3d_knowledge.fixpParamsFullLoadFlag= j["fixpParamsFullLoadFlag"];
        conv3d_knowledge.hf32Enable= j["hf32Enable"];
        conv3d_knowledge.hf32TransMode= j["hf32TransMode"];
        conv3d_knowledge.batchDim= j["batchDim"];
        conv3d_knowledge.nDim= j["nDim"];
        conv3d_knowledge.hoDim= j["hoDim"];
        conv3d_knowledge.doDim= j["doDim"];
        conv3d_knowledge.groupDim= j["groupDim"];
        conv3d_knowledge.isC04Flag= j["isC04Flag"];
        conv3d_knowledge.mMode= j["mMode"];
        //wyh
        conv3d_knowledge.bl1FullLoad = j["bl1FullLoad"];
        conv3d_knowledge.al1FullLoad = j["al1FullLoad"];
        conv3d_knowledge.bl1BypassFlag = j["bl1BypassFlag"];
    }
    void PrepareInfoDict(Conv3DTilingTestParamRepo &param) {
        std::cout << param.info_dict << std::endl;
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

TEST_P(Conv3DTilingRepo, general_cases_001) {
    Conv3DTilingTestParamRepo param = GetParam();
    std::cout << "run case " << param.case_name << std::endl;
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
    if(hasBias) {
		  input_shape_ref = {&featuremap, &weight, &bias};
	  } else {
		  input_shape_ref = {&featuremap, &weight, nullptr};
	  }

    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides_ref = {1, 1, conv3d_info_dict.strideD, conv3d_info_dict.strideH, conv3d_info_dict.strideW};
    std::vector<int64_t> pads_ref = {conv3d_info_dict.padHead, conv3d_info_dict.padTail,conv3d_info_dict.padTop,
                                conv3d_info_dict.padBottom, conv3d_info_dict.padLeft, conv3d_info_dict.padRight};
    std::vector<int64_t> dilations_ref = {1, 1, conv3d_info_dict.dilationD, conv3d_info_dict.dilationH, conv3d_info_dict.dilationW};

    std::string op_type = "Conv3DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    uint32_t aicoreNum = 32;
    string compile_info_string = R"({"hardware_info": 
      {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
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
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    auto holder = gert::TilingContextFaker().SetOpType(op_type)
                                            .NodeIoNum(3, 1)
                                            .IrInstanceNum({1, 1, 1})
                                            .InputShapes(input_shape_ref)
                                            .OutputShapes(output_shapes_ref)
                                            .CompileInfo(&compile_info)
                                            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                                            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                                            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                            .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                                            .NodeAttrs({
                                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides_ref)},
                                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads_ref)},
                                                {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations_ref)},
                                                {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                                {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}
                                                })
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
  {"Conv3DV2_repo_test_0","Conv3DV2",
R"({"aDtype": 27, "bDtype": 27, "cDtype": 27, "biasDtype": 27, "aShapeN": 1, "aShapeD": 243, "aShapeH": 542, "aShapeW": 62, "bShapeN": 256, "bShapeC": 256, 
"bShapeD": 3, "bShapeH": 3, "bShapeW": 3, "cShapeD": 241, "cShapeH": 540, "cShapeW": 60, "aFormat": 30, "bFormat": 30, "cFormat": 30, "groups": 1, 
"strideD": 1, "strideH": 1, "strideW": 1, "dilationD": 1, "dilationH": 1, "dilationW": 1, "padHead": 0, "padTail": 0, "padTop": 0, "padBottom": 0, 
"padLeft": 0, "padRight": 0, "biasFlag": true, "reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0, 
"reserverdParam6": 0})", 
R"({"groups": 1, "singleCoreDo": 241, "singleCoreCo": 128, "singleCoreHo": 2025, "singleCoreWo": 0, "singleCoreCi": 256, 
"orgDo": 241, "orgCo": 256, "orgHo": 540, "orgWo": 60, "orgCi": 256, "orgDi": 243, "orgHi": 542, "orgWi": 62, "kernelD": 3, "kernelH": 3, "kernelW": 3, 
"strideD": 1, "strideH": 1, "strideW": 1, "dilationD": 1, "dilationH": 1, "dilationW": 1, "padHead": 0, "padTail": 0, "padTop": 0, "padBottom": 0, 
"padLeft": 0, "padRight": 0, "hoL0": 512, "woL0": 0, "kL0": 32, "nL0": 128, "kAL1": 288, "kBL1": 288, "nBL1": 128, "hoL1": 512, "woL1": 0, 
"pBufferFlag": 27, "bl1FullLoad": 0, "al1FullLoad": 0, "bl1BypassFlag": 0, "iterateMNOrder": 0, "biasFullLoadFlag": 1, "fixpParamsFullLoadFlag": 1, "hf32Enable": 0,
 "hf32TransMode": 0, "batchDim": 1, "nDim": 2, "hoDim": 16, "doDim": 1, "groupDim": 1, "isC04Flag": 0, "mMode": 1})"
  }
};

INSTANTIATE_TEST_CASE_P(Conv3DV2, Conv3DTilingRepo, testing::ValuesIn(general_cases_params_repo));