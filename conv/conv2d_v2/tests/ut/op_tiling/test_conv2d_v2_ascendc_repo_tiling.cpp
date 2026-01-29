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
#include <stdio.h>
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

using namespace std;
using namespace ge;
using namespace ut_util;

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
    void SetUp() {}
    void TearDown() {}
    static void TearDownTestCase() {}
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
        conv2d_knowledge.hoL0 = j["hoL0"];
        conv2d_knowledge.woL0 = j["woL0"];
        conv2d_knowledge.kL0 = j["kL0"];
        conv2d_knowledge.nL0 = j["nL0"];
        conv2d_knowledge.pBufferFlag= j["pBufferFlag"];
        conv2d_knowledge.nBL1 = j["nBL1"];
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
        conv2d_knowledge.nDim = j["nDim"];
        conv2d_knowledge.groupDim = j["groupDim"];
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

TEST_P(Conv2DTilingRepo, general_cases_001) {
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
    uint32_t aicoreNum = 32;

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
    aicoreNum = 32;
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
    compile_info.aicoreNum = aicoreNum;
    compile_info.socVersion = "Ascend950PR_9589";
    compile_info.shortSocVersion = "Ascend950";

    optiling::Conv2DTilingParseInfo quant_compile_info;
    quant_compile_info.opType = op_type;
    quant_compile_info.shortSocVersion = "Ascend950";

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

    ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
}

static Conv2DTilingTestParamRepo general_cases_params_repo[] = {
    {
    "Conv2D_repo_test_0","Conv2DV2",
    R"({"groups":1,"orgCi":1280,"orgHi":32,"orgWi":32,"orgCo":1280,"orgHo":32,"orgWo":32,"kernelH":3,"kernelW":3,
    "singleCoreCi":1280,"singleCoreCo":80,"singleCoreHo":16,"singleCoreWo":32,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":1,"padBottom":1,"padLeft":1,"padRight":1,"biasFullLoadFlag":1,"fixpParamsFullLoadFlag":1,
    "offsetx":0,"hf32Enable":0,"hf32TransMode":0,"isC04Flag":0,"roundMode":1,"batchDim":1,"hoDim":2,"woDim":1,"nDim":16,
    "groupDim":1,"mMode":0,"woL1":32,"woL0":32,"kAL1":576,"kBL1":576,"hoL1":16,"nBL1":80,"kL0":32,"nL0":80,"hoL0":16,
    "iterateMNOrder":1,"pBufferFlag":27})",R"({"aDtype":1,"bDtype":1,
    "cDtype":1,"biasDtype":1,"aShapeN":1,"aShapeH":32,"aShapeW":32,"bShapeN":1280,"bShapeC":1280,"bShapeH":3,"bShapeW":3,
    "cShapeH":32,"cShapeW":32,"aFormat":0,"bFormat":0,"cFormat":0,"groups":1,"strideH":1,"strideW":1,"dilationH":1,
    "dilationW":1,"padTop":1,"padBottom":1,"padLeft":1,"padRight":1,"biasFlag":true,"hf32Flag":false,
    "reserverdParam1": 0, "reserverdParam2": 0, "reserverdParam3": 0, "reserverdParam4": 0, "reserverdParam5": 0,
    "reserverdParam6": 0})"
    }
};

INSTANTIATE_TEST_CASE_P(Conv2DV2, Conv2DTilingRepo, testing::ValuesIn(general_cases_params_repo));