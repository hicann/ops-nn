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
 * \file test_conv3d_v2_ascendc_tiling_repo.cpp
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
#include "conv/conv3d_v2/op_host/op_tiling/conv3d_base_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_utils.h"

using namespace ut_util;

namespace RuntimeKb {
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
    void SetUp() { }

    void TearDown() {
        unsetenv("TUNE_BANK_PATH");
    }
    static void TearDownTestCase() {
        Configuration::Instance().Finalize();
    }
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

TEST_P(Conv3DTilingRepo, general_cases) {
    Conv3DTilingTestParamRepo param = GetParam();
    std::cout << "run case " << param.case_name << std::endl;
    PrepareTest(param);
    Configuration::Instance().Finalize();
    bank_path = "/tmp/" + GetTestSuiteName() + "/" + GetTestCaseName();
    setenv("TUNE_BANK_PATH", bank_path.c_str(), 1);
    auto ret = SystemUtils::CreateMultiDirectory(bank_path);
    EXPECT_EQ(ret, SUCCESS);
    PlatformInfo plat;
    plat.soc_version = "Ascend910_9589";
    plat.core_num = 32;
    const std::set<std::string> kOpWhiteLists{"Conv3DV2"};
    ret = RuntimeBankManager::Instance().InitAoeOpBank(plat, kOpWhiteLists);
    EXPECT_EQ(ret, SUCCESS);
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
    uint64_t aicoreNum = 32;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", 
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, 
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288, 
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, 
                            "CORE_NUM": 32}
                            })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;
    compile_info.tilingType = op_type;
    compile_info.aicoreNum = aicoreNum;
    compile_info.shortSocVersion = "Ascend910_95";
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
                                                {"strides", ge::AnyValue::CreateFrom<std::vector<int64_t>>(strides_ref)},
                                                {"pads", ge::AnyValue::CreateFrom<std::vector<int64_t>>(pads_ref)},
                                                {"dilations", ge::AnyValue::CreateFrom<std::vector<int64_t>>(dilations_ref)},
                                                {"groups", ge::AnyValue::CreateFrom<int64_t>(1)},
                                                {"data_format", ge::AnyValue::CreateFrom<std::string>("NCHW")},
                                                {"offset_x", ge::AnyValue::CreateFrom<int64_t>(0)},
                                                {"pad_mode", ge::AnyValue::CreateFrom<std::string>("SPECIFIC")},
                                                {"enable_hf32", ge::AnyValue::CreateFrom<bool>(false)}
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

    ret = RuntimeBankManager::Instance().Update(tiling_context, optype, tiling);
    EXPECT_EQ(ret, SUCCESS);
    ret = RuntimeBankManager::Instance().Save();
    EXPECT_EQ(ret, SUCCESS);

    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    // ASSERT_EQ(tiling_data_result, param.tiling_data);
    const std::string rm_bank_dir = "rm -rf " + bank_path;
    system(rm_bank_dir.c_str());
}

static Conv3DTilingTestParamRepo general_cases_params_repo[] = {
  {
"Conv3DV2_repo_test_0","Conv3DV2",R"({"aDtype":27,"bDtype":27,"cDtype":27,"biasDtype":27,"aShapeN":1,"aShapeD":16,"aShapeH":16,"aShapeW":16,"bShapeN":1,"bShapeC":1,"bShapeD":1,"bShapeH":1,"bShapeW":1,"cShapeD":16,"cShapeH":16,"cShapeW":16,"aFormat":30,"bFormat":30,"cFormat":30,"groups":1,"strideD":1,"strideH":1,"strideW":1,"dilationD":1,"dilationH":1,"dilationW":1,"padHead":0,"padTail":0,"padTop":0,"padBottom":0,"padLeft":0,"padRight":0,"biasFlag":true,"reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0, "reserverdParam6": 0})",
R"({"groups":1,"orgCi":1,"orgDi":16,"orgHi":16,"orgWi":16,"orgDo":16,"orgCo":1,"orgHo":16,"orgWo":16,"kernelD":1,"kernelH":1,"kernelW":1,"singleCoreDo":1,"singleCoreCo":1,"singleCoreHo":128,"singleCoreWo":0,"singleCoreCi":1,"kAL1":16,"kBL1":16,"nBL1":16,"hoL1":128,"woL1":0,"hoL0":128,"woL0":0,"kL0":16,"nL0":16,"pBufferFlag":7,"strideD":1,"strideH":1,"strideW":1,"dilationD":1,"dilationH":1,"dilationW":1,"padHead":0,"padTail":0,"padTop":0,"padBottom":0,"padLeft":0,"padRight":0,"iterateMNOrder":0,"biasFullLoadFlag":1,"fixpParamsFullLoadFlag":0,"hf32Enable":0,"hf32TransMode":0,"batchDim":1,"doDim":16,"hoDim":2,"nDim":1,"groupDim":1, "mMode": 1, "isC04Flag": 0})"
  }
};

INSTANTIATE_TEST_CASE_P(Conv3DV2, Conv3DTilingRepo, testing::ValuesIn(general_cases_params_repo));

}