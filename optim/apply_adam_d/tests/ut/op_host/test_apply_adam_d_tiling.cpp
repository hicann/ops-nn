/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "common/utils/ut_profiling_reg.h"
#include "image_ops.h"
#include "array_ops.h"
#include "test_common.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "runtime/elewise_tiling.h"
#include "test_cube_util.h"

using namespace ge;
using namespace ut_util;

class ApplyAdamDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdamDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdamDTiling TearDown" << std::endl;
  }
};


static string TilingData2Str(const gert::TilingData *tilingDataV)
{
    auto data = tilingDataV->GetData();
    string result;
    for (size_t i = 0; i < tilingDataV->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t *>(tilingDataV->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void InitPlatForm(fe::PlatFormInfos &platformInfo, map<string, string> &socInfos,
    map<string, string> &aicoreSpec, map<string, string> &intrinsics)
{
    string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();
}

static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

static void DoTest(gert::StorageShape &var, gert::StorageShape &m, gert::StorageShape &v,
                   gert::StorageShape &beta1Power, gert::StorageShape &beta2Power, gert::StorageShape &lr,
                   gert::StorageShape &beta1, gert::StorageShape &beta2, gert::StorageShape &epsilon, gert::StorageShape &grad,
                   gert::StorageShape &var_out, gert::StorageShape &m_out, gert::StorageShape &v_out,
                   ge::DataType varDtype, ge::Format format, bool useLocking, bool useNesterov,
                   int64_t expectKey, string &expectData) {
    optiling::ElewiseCompileInfo compileInfo;
    compileInfo.isAscendC = true;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;

    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platformInfo, socInfos, aicoreSpec, intrinsics);

    std::string opType("ApplyAdamD");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(32);
    auto wsSize = reinterpret_cast<gert::ContinuousVector *>(workspaceSizeHoler.get());
    
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&var, &m, &v, &beta1Power, &beta2Power, &lr, &beta1, &beta2, &epsilon, &grad})
                      .OutputShapes({&var_out, &m_out, &v_out})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char *>(&platformInfo))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs({{"use_locking", ge::AnyValue::CreateFrom<bool>(useLocking)},
                                  {"use_nesterov", ge::AnyValue::CreateFrom<bool>(useNesterov)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    // check tiling result
    auto tilingKey = tilingContext->GetTilingKey();
    EXPECT_EQ(tilingKey, expectKey);
    auto tilingDataResult = TilingData2Str(tilingContext->GetRawTilingData());
}

#define RUN_TEST_WITH_SHAPE(...) do {\
    gert::StorageShape var = {{__VA_ARGS__}, {__VA_ARGS__}};\
    gert::StorageShape m = {{__VA_ARGS__}, {__VA_ARGS__}};\
    gert::StorageShape v = {{__VA_ARGS__}, {__VA_ARGS__}};\
    gert::StorageShape beta1Power = {{1}, {1}};\
    gert::StorageShape beta2Power = {{1}, {1}};\
    gert::StorageShape lr = {{1}, {1}};\
    gert::StorageShape beta1 = {{1}, {1}};\
    gert::StorageShape beta2 = {{1}, {1}};\
    gert::StorageShape epsilon = {{1}, {1}};\
    gert::StorageShape grad = {{__VA_ARGS__}, {__VA_ARGS__}};\
    gert::StorageShape var_out = {{__VA_ARGS__}, {__VA_ARGS__}};\
    gert::StorageShape m_out = {{__VA_ARGS__}, {__VA_ARGS__}};\
    gert::StorageShape v_out = {{__VA_ARGS__}, {__VA_ARGS__}};\
    DoTest(var, m, v,beta1Power, beta2Power, lr,beta1, beta2, epsilon, grad,\
           var_out, m_out, v_out, varDtype, dataFormat, useLocking, useNesterov,expectKey, expectData);\
} while(0)

TEST_F(ApplyAdamDTiling, apply_adam_d_tiling_1000)
{
    auto varDtype = ge::DT_FLOAT;
    auto dataFormat = ge::FORMAT_ND;
    bool useLocking = false;
    bool useNesterov = false;
    int64_t expectKey = 103;
    string expectData =
      "40001 64 245760 0 0 64 32 10 10 5 5 1 5 5 32 25 38 5 5 32 0 0 4611686019501129728 25 ";
    RUN_TEST_WITH_SHAPE(64, 10, 10, 32);
}

TEST_F(ApplyAdamDTiling, apply_adam_d_tiling_1001)
{
    auto varDtype = ge::DT_FLOAT16;
    auto dataFormat = ge::FORMAT_ND;
    bool useLocking = false;
    bool useNesterov = false;
    int64_t expectKey = 101;
    string expectData =
      "40001 64 245760 0 0 64 32 10 10 5 5 1 5 5 32 25 38 5 5 32 0 0 4611686019501129728 25 ";
    RUN_TEST_WITH_SHAPE(64, 10, 10, 32);
}

TEST_F(ApplyAdamDTiling, apply_adam_d_tiling_1002)
{
    auto varDtype = ge::DT_BF16;
    auto dataFormat = ge::FORMAT_ND;
    bool useLocking = false;
    bool useNesterov = false;
    int64_t expectKey = 102;
    string expectData =
      "40001 64 245760 0 0 64 32 10 10 5 5 1 5 5 32 25 38 5 5 32 0 0 4611686019501129728 25 ";
    RUN_TEST_WITH_SHAPE(64, 10, 10, 32);
}

TEST_F(ApplyAdamDTiling, apply_adam_d_tiling_1003)
{
    auto varDtype = ge::DT_FLOAT;
    auto dataFormat = ge::FORMAT_ND;
    bool useLocking = false;
    bool useNesterov = true;
    int64_t expectKey = 103;
    string expectData =
      "40001 64 245760 0 0 64 32 10 10 5 5 1 5 5 32 25 38 5 5 32 0 0 4611686019501129728 25 ";
    RUN_TEST_WITH_SHAPE(64, 10, 10, 32);
}

TEST_F(ApplyAdamDTiling, apply_adam_d_tiling_1004)
{
    auto varDtype = ge::DT_FLOAT16;
    auto dataFormat = ge::FORMAT_ND;
    bool useLocking = false;
    bool useNesterov = true;
    int64_t expectKey = 101;
    string expectData =
      "40001 64 245760 0 0 64 32 10 10 5 5 1 5 5 32 25 38 5 5 32 0 0 4611686019501129728 25 ";
    RUN_TEST_WITH_SHAPE(64, 10, 10, 32);
}

TEST_F(ApplyAdamDTiling, apply_adam_d_tiling_1005)
{
    auto varDtype = ge::DT_BF16;
    auto dataFormat = ge::FORMAT_ND;
    bool useLocking = false;
    bool useNesterov = true;
    int64_t expectKey = 102;
    string expectData =
      "40001 64 245760 0 0 64 32 10 10 5 5 1 5 5 32 25 38 5 5 32 0 0 4611686019501129728 25 ";
    RUN_TEST_WITH_SHAPE(64, 10, 10, 32);
}