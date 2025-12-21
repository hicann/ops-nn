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
 * \file test_conv2d_ascendc_tiling_repo.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <stdio.h>
#include <map>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

#include "op_log.h"
#include "fusion_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "platform/platform_info.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/cube_tiling_runtime.h"
#include "op_tiling/op_tiling_util.h"
#include "test_cube_util.h"
#include "aoe/runtime_kb/common/utils/configuration.h"
#include "aoe/runtime_kb/common/utils/system_utils.h"
#include "aoe/runtime_kb/runtime_bank_manager.h"
#include "../../../op_host/op_tiling/arch35/conv2d_v2_tuning_tiling.h"
#include "../../../op_host/op_tiling/conv2d_base_tiling.h"
#include "../../../../common/op_host/op_tiling/conv_base.h"

using namespace std;
using namespace ge;

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

struct Conv2DTilingTestParamRepo {
  string case_name;
  string op_type;
  string tiling_data;
  string info_dict;
};

class Conv2DTilingRepo: public testing::TestWithParam<Conv2DTilingTestParamRepo> {
protected:
    void SetUp() { }

    void TearDown() {
        unsetenv("TUNE_BANK_PATH");
    }
    static void TearDownTestCase() {
        Configuration::Instance().Finalize();
    }
    void PrepareTest(Conv2DTilingTestParamRepo &param) {
        PrepareKnowledge(param);
        PrepareInfoDict(param);
    }
    void PrepareKnowledge(Conv2DTilingTestParamRepo &param) {
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
        conv2d_knowledge.multiNBL1 = j["multiNBL1"];
        conv2d_knowledge.hoL0 = j["hoL0"];
        conv2d_knowledge.woL0 = j["woL0"];
        conv2d_knowledge.kL0 = j["kL0"];
        conv2d_knowledge.nL0 = j["nL0"];
        conv2d_knowledge.pBufferFlag= j["pBufferFlag"];
        conv2d_knowledge.kernelHxkernelW= j["kernelHxkernelW"];
        conv2d_knowledge.cinAInCore= j["cinAInCore"];
        conv2d_knowledge.cinATailInCore= j["cinATailInCore"];
        conv2d_knowledge.oriHinxWin= j["oriHinxWin"];
        conv2d_knowledge.cinOffsetBlockInGM= j["cinOffsetBlockInGM"];
        conv2d_knowledge.mStep= j["mStep"];
        conv2d_knowledge.fmapKStride= j["fmapKStride"];
        conv2d_knowledge.nStep= j["nStep"];
        conv2d_knowledge.kStep= j["kStep"];
        conv2d_knowledge.weightKStride = j["weightKStride"];
        conv2d_knowledge.coutOffsetBlock = j["coutOffsetBlock"];
        conv2d_knowledge.cinBInCore = j["cinBInCore"];
        conv2d_knowledge.cinBTailInCore = j["cinBTailInCore"];
        conv2d_knowledge.nBL1 = j["nBL1"];
        conv2d_knowledge.nL1DivBlockSize = j["nL1DivBlockSize"];
        conv2d_knowledge.strideH= j["strideH"];
        conv2d_knowledge.strideW= j["strideW"];
        conv2d_knowledge.dilationH= j["dilationH"];
        conv2d_knowledge.dilationW= j["dilationW"];
        conv2d_knowledge.padTop= j["padTop"];
        conv2d_knowledge.padBottom= j["padBottom"];
        conv2d_knowledge.padLeft= j["padLeft"];
        conv2d_knowledge.padRight= j["padRight"];
        conv2d_knowledge.iterateMNOrder= j["iterateMNOrder"];
        conv2d_knowledge.biasFullLoadFlag= j["biasFullLoadFlag"];
        conv2d_knowledge.fixpParamsFullLoadFlag= j["fixpParamsFullLoadFlag"];
        conv2d_knowledge.offsetx = j["offsetx"];
        conv2d_knowledge.hf32Enable= j["hf32Enable"];
        conv2d_knowledge.hf32TransMode= j["hf32TransMode"];
        conv2d_knowledge.isC04Flag= j["isC04Flag"];
        conv2d_knowledge.roundMode= j["roundMode"];
        conv2d_knowledge.batchDim= j["batchDim"];
        conv2d_knowledge.hoDim= j["hoDim"];
        conv2d_knowledge.nDim = j["nDim"];
        conv2d_knowledge.nDim = j["groupDim"];
        conv2d_knowledge.mMode = j["mMode"];
    }

    void PrepareInfoDict(Conv2DTilingTestParamRepo &param) {
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

static string to_string(const std::stringstream &tiling_data) {
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

TEST_P(Conv2DTilingRepo, general_cases) {
    Conv2DTilingTestParamRepo param = GetParam();
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
    const std::set<std::string> kOpWhiteLists{"Conv2DV2"};
    ret = RuntimeBankManager::Instance().InitAoeOpBank(plat, kOpWhiteLists);
    EXPECT_EQ(ret, SUCCESS);
    tuningtiling::TuningTilingDefPtr tiling = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance(optype);
    nlohmann::json a;
    conv2d_knowledge.ToJson(a);
    tiling->FromJson(a);

    gert::StorageShape featuremap = {{conv2d_info_dict.aShapeN, conv2d_info_dict.bShapeC, 
                                        conv2d_info_dict.aShapeH, conv2d_info_dict.aShapeW},
                                        {conv2d_info_dict.aShapeN, conv2d_info_dict.bShapeC, 
                                        conv2d_info_dict.aShapeH, conv2d_info_dict.aShapeW}};
    gert::StorageShape weight = {{conv2d_info_dict.bShapeN, conv2d_info_dict.bShapeC, 
                                    conv2d_info_dict.bShapeH, conv2d_info_dict.bShapeW},
                                    {conv2d_info_dict.bShapeN, conv2d_info_dict.bShapeC, 
                                    conv2d_info_dict.bShapeH, conv2d_info_dict.bShapeW}};
    gert::StorageShape bias = {{conv2d_info_dict.bShapeN}, {conv2d_info_dict.bShapeN}};
    gert::StorageShape quantScale = {{conv2d_info_dict.bShapeN}, {conv2d_info_dict.bShapeN}};
    gert::StorageShape offset_w;
    gert::StorageShape output = {{conv2d_info_dict.aShapeN, conv2d_info_dict.bShapeN, 
                                    conv2d_info_dict.cShapeH, conv2d_info_dict.cShapeW},
                                    {conv2d_info_dict.aShapeN, conv2d_info_dict.bShapeN, 
                                    conv2d_info_dict.cShapeH, conv2d_info_dict.cShapeW}};
    std::vector<void*> input_shape_ref;

    bool quantFlag = conv2d_info_dict.aDtype == ge::DT_INT8;
    bool hasScale = false;
    bool hasBias = conv2d_info_dict.biasFlag;
    if (quantFlag) {
        if (hasBias && hasScale) {
            input_shape_ref = {&featuremap, &weight, &quantScale, &bias, nullptr};
        } else if (!hasBias && hasScale) {
            input_shape_ref = {&featuremap, &weight, &quantScale, nullptr, nullptr};
        } else if (hasBias && !hasScale) {
            input_shape_ref = {&featuremap, &weight, nullptr, &bias, nullptr};
        } else {
            input_shape_ref = {&featuremap, &weight, nullptr, nullptr, nullptr};
        }
    } else {
        if (hasBias) {
            input_shape_ref = {&featuremap, &weight, &bias, nullptr};
        } else {
            input_shape_ref = {&featuremap, &weight, nullptr, nullptr};
        }
    }

    std::vector<void*> output_shapes_ref = {&output};
    std::vector<int64_t> strides = {1, 1, conv2d_info_dict.strideH, conv2d_info_dict.strideW};
    std::vector<int64_t> pads = {conv2d_info_dict.padTop, conv2d_info_dict.padBottom, 
                                conv2d_info_dict.padLeft, conv2d_info_dict.padRight};
    std::vector<int64_t> dilations = {1, 1, conv2d_info_dict.dilationH, conv2d_info_dict.dilationW};

    std::string op_type = quantFlag ? "QuantConv2D" : "Conv2DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    uint64_t aicoreNum = 32;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
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
    compile_info.aicoreNum = aicoreNum;
    compile_info.socVersion = "Ascend910_9589";
    compile_info.shortSocVersion = "Ascend910_95";

    optiling::Conv2DTilingParseInfo quant_compile_info;
    quant_compile_info.opType = op_type;
    quant_compile_info.shortSocVersion = "Ascend910_95";

    auto tilingDataPtr = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    auto holder = quantFlag ?
                      gert::TilingContextFaker().SetOpType(op_type)
                                                .NodeIoNum(5, 1)
                                                .IrInstanceNum({1, 1, 1, 1, 1})
                                                .InputShapes(input_shape_ref)
                                                .OutputShapes(output_shapes_ref)
                                                .CompileInfo(&quant_compile_info)
                                                .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                                .NodeInputTd(0, ge::DT_INT8, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                                                .NodeInputTd(1, ge::DT_INT8, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                                                .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                                                .NodeAttrs({
                                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
                                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<std::string>("rint")}
                                                  })
                                                .TilingData(tilingDataPtr.get())
                                                .Workspace(ws_size)
                                                .Build() :
                      gert::TilingContextFaker().SetOpType(op_type)
                                                .NodeIoNum(4, 1)
                                                .IrInstanceNum({1, 1, 1, 1})
                                                .InputShapes(input_shape_ref)
                                                .OutputShapes(output_shapes_ref)
                                                .CompileInfo(&compile_info)
                                                .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                                                .NodeInputTd(0, conv2d_info_dict.aDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                                                .NodeInputTd(1, conv2d_info_dict.aDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                                                .NodeInputTd(2, conv2d_info_dict.aDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeInputTd(3, conv2d_info_dict.aDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                                                .NodeOutputTd(0, conv2d_info_dict.aDtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                                                .NodeAttrs({
                                                    {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                                    {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
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

    auto ret1 = RuntimeBankManager::Instance().Update(tiling_context, optype, tiling);
    EXPECT_EQ(ret1, SUCCESS);
    ret = RuntimeBankManager::Instance().Save();
    EXPECT_EQ(ret, SUCCESS);

    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << "===== " << tiling_key << " === " << tiling_data_result << std::endl;
    const std::string rm_bank_dir = "rm -rf " + bank_path;
    system(rm_bank_dir.c_str());
}

static Conv2DTilingTestParamRepo general_cases_params_repo[] = {
    {
    "Conv2D_repo_test_0","Conv2DV2",
    R"({"groups":1,"orgCi":18,"orgHi":2,"orgWi":32,"orgCo":32,"orgHo":2,"orgWo":32,"kernelH":1,"kernelW":1,
    "singleCoreCi":18,"singleCoreCo":32,"singleCoreHo":2,"singleCoreWo":32,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":0,"padBottom":0,"padLeft":0,"padRight":0,"biasFullLoadFlag":1,"fixpParamsFullLoadFlag":1,
    "offsetx":0,"hf32Enable":0,"hf32TransMode":0,"isC04Flag":0,"roundMode":1,"batchDim":1,"hoDim":1,"nDim":1,
    "groupDim":1,"mMode":0,"woL1":32,"woL0":32,"kAL1":32,"kBL1":32,"hoL1":2,"nBL1":32,"kL0":16,"nL0":16,"hoL0":1,
    "iterateMNOrder":0,"pBufferFlag":3,"multiNBL1":2,"kernelHxkernelW":1,"cinAInCore":32,"cinATailInCore":18,
    "oriHinxWin":64,"cinOffsetBlockInGM":2048,"mStep":32,"fmapKStride":2,"nStep":1,"kStep":1,"weightKStride":2,
    "coutOffsetBlock":18,"cinBInCore":32,"cinBTailInCore":18,"nL1DivBlockSize":2})",R"({"aDtype":1,"bDtype":1,
    "cDtype":1,"biasDtype":1,"aShapeN":1,"aShapeH":2,"aShapeW":32,"bShapeN":32,"bShapeC":18,"bShapeH":1,"bShapeW":1,
    "cShapeH":2,"cShapeW":32,"aFormat":0,"bFormat":0,"cFormat":0,"groups":1,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":0,"padBottom":0,"padLeft":0,"padRight":0,"biasFlag":true,"hf32Flag":false,
    "reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0,
    "reserverdParam6": 0})"
    }
};

INSTANTIATE_TEST_CASE_P(Conv2DV2, Conv2DTilingRepo, testing::ValuesIn(general_cases_params_repo));

}
