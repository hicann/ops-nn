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
#include <fstream>
#include <vector>
#include <string>
#include "array_ops.h"
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/masked_softmax_with_rel_pos_bias_tiling.h"

using namespace std;

class MaskedSoftmaxWithRelPosBiasTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaskedSoftmaxWithRelPosBiasTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaskedSoftmaxWithRelPosBiasTiling TearDown" << std::endl;
  }
};

using namespace ge;
using namespace ut_util;
using namespace optiling;

void testTiling(gert::StorageShape &x_shape,
                gert::StorageShape &atten_mask_shape,
                gert::StorageShape &bias_shape, gert::StorageShape &y_shape,
                float scaleValue, bool hasAtten, uint32_t key) {
  map<string, string> socInfos;
  map<string, string> aicoreSpec;
  map<string, string> intrinsics;
  string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,"CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct MaskedSoftmaxWithRelPosBiasCompileInfo {};
  MaskedSoftmaxWithRelPosBiasCompileInfo compile_info;

  std::string op_type("MaskedSoftmaxWithRelPosBias");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
            nullptr);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size =
      reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);

  ge::DataType testDataType[] = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
  for (uint32_t i = 0; i < sizeof(testDataType) / sizeof(ge::DataType); i++) {
    gert::KernelRunContextHolder holder;
    if (hasAtten) {
      holder =
          gert::TilingContextFaker()
              .NodeIoNum(3, 1)
              .IrInstanceNum({1, 1, 1})
              .InputShapes({&x_shape, &atten_mask_shape, &bias_shape})
              .OutputShapes({&y_shape})
              .CompileInfo(&compile_info)
              .PlatformInfo(reinterpret_cast<char *>(&platform_info))
              .NodeInputTd(0, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeInputTd(1, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeInputTd(2, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeOutputTd(0, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeAttrs({
                  {"scale_value", Ops::NN::AnyValue::CreateFrom<float>(scaleValue)},
                  {"inner_precision_mode",
                   Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
              })
              .TilingData(param.get())
              .Workspace(ws_size)
              .SetOpType(op_type)
              .Build();
    } else {
      holder =
          gert::TilingContextFaker()
              .NodeIoNum(3, 1)
              .IrInstanceNum({1, 1, 1})
              .InputShapes({&x_shape, nullptr, &bias_shape})
              .OutputShapes({&y_shape})
              .CompileInfo(&compile_info)
              .PlatformInfo(reinterpret_cast<char *>(&platform_info))
              .NodeInputTd(0, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeInputTd(1, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeInputTd(2, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeOutputTd(0, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeAttrs({
                  {"scale_value", Ops::NN::AnyValue::CreateFrom<float>(scaleValue)},
                  {"inner_precision_mode",
                   Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
              })
              .TilingData(param.get())
              .Workspace(ws_size)
              .SetOpType(op_type)
              .Build();
    }

    gert::TilingContext *tiling_context =
        holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions); // label:"version" res:socversions
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(tiling_context),
              ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, key + 10 * i);
  }
}

void testTilingFailed(gert::StorageShape &x_shape,
                gert::StorageShape &atten_mask_shape,
                gert::StorageShape &bias_shape, gert::StorageShape &y_shape,
                float scaleValue, bool hasAtten, ge::DataType dataType) {
  map<string, string> socInfos;
  map<string, string> aicoreSpec;
  map<string, string> intrinsics;
  string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,"CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct MaskedSoftmaxWithRelPosBiasCompileInfo {};
  MaskedSoftmaxWithRelPosBiasCompileInfo compile_info;

  std::string op_type("MaskedSoftmaxWithRelPosBias");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
            nullptr);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size =
      reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);

  gert::KernelRunContextHolder holder;
  if (hasAtten) {
    holder =
        gert::TilingContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&x_shape, &atten_mask_shape, &bias_shape})
            .OutputShapes({&y_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({
                {"scale_value", Ops::NN::AnyValue::CreateFrom<float>(scaleValue)},
                {"inner_precision_mode",
                 Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
            })
            .TilingData(param.get())
            .Workspace(ws_size)
            .SetOpType(op_type)
            .Build();
  } else {
    holder =
        gert::TilingContextFaker()
            .NodeIoNum(3, 1)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&x_shape, nullptr, &bias_shape})
            .OutputShapes({&y_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char *>(&platform_info))
            .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({
                {"scale_value", Ops::NN::AnyValue::CreateFrom<float>(scaleValue)},
                {"inner_precision_mode",
                 Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
            })
            .TilingData(param.get())
            .Workspace(ws_size)
            .SetOpType(op_type)
            .Build();
  }

  gert::TilingContext *tiling_context =
      holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
      "SoCInfo", socInfos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
      "AICoreSpec", aicoreSpec);
  holder.GetContext<gert::TilingContext>()
      ->GetPlatformInfo()
      ->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
      "AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(tiling_context),
            ge::GRAPH_FAILED);
}

void testTilingShapeCheckFailed(gert::StorageShape &x_shape,
                gert::StorageShape &atten_mask_shape,
                gert::StorageShape &bias_shape, gert::StorageShape &y_shape) {
  map<string, string> socInfos;
  map<string, string> aicoreSpec;
  map<string, string> intrinsics;
  string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,"CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct MaskedSoftmaxWithRelPosBiasCompileInfo {};
  MaskedSoftmaxWithRelPosBiasCompileInfo compile_info;

  std::string op_type("MaskedSoftmaxWithRelPosBias");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
            nullptr);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size =
      reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  gert::KernelRunContextHolder holder =
      gert::TilingContextFaker()
          .NodeIoNum(3, 1)
          .IrInstanceNum({1, 1, 1})
          .InputShapes({&x_shape, &atten_mask_shape, &bias_shape})
          .OutputShapes({&y_shape})
          .CompileInfo(&compile_info)
          .PlatformInfo(reinterpret_cast<char *>(&platform_info))
          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeAttrs({
              {"scale_value", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
              {"inner_precision_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
          })
          .TilingData(param.get())
          .Workspace(ws_size)
          .SetOpType(op_type)
          .Build();
  gert::TilingContext *tiling_context =
      holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
      "SoCInfo", socInfos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
      "AICoreSpec", aicoreSpec);
  holder.GetContext<gert::TilingContext>()
      ->GetPlatformInfo()
      ->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
      "AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(tiling_context),
            ge::GRAPH_FAILED);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_50) {
  int32_t B = 48;
  int32_t W = 1;
  int32_t N = 1;
  int32_t S1 = 16;
  int32_t S2 = 16;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 5001);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 5002);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 5003);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 5004);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_51) {
  int32_t B = 48;
  int32_t W = 1;
  int32_t N = 1;
  int32_t S1 = 32;
  int32_t S2 = 31;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 5101);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 5102);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 5103);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 5104);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_52) {
  int32_t B = 1023;
  int32_t W = 4;
  int32_t N = 2;
  int32_t S1 = 16;
  int32_t S2 = 16;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 5001);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 5002);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 5003);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 5004);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_53) {
  int32_t B = 49;
  int32_t W = 3;
  int32_t N = 5;
  int32_t S1 = 4;
  int32_t S2 = 3;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 5101);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 5102);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 5103);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 5104);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_54) {
  int32_t B = 525;
  int32_t W = 3;
  int32_t N = 5;
  int32_t S1 = 4;
  int32_t S2 = 3;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 5101);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 5102);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 5103);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 5104);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, BWN_tiling_30) {
  uint32_t B = 2;
  uint32_t W = 2;
  uint32_t N = 128;
  uint32_t S1 = 8;
  uint32_t S2 = 8;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 301);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 302);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 303);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 304);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, BWN_tiling_31) {
  uint32_t B = 2;
  uint32_t W = 3;
  uint32_t N = 127;
  uint32_t S1 = 5;
  uint32_t S2 = 5;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 301);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 302);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 303);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 304);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_40) {
  int32_t B = 2;
  int32_t W = 95;
  int32_t N = 1;
  int32_t S1 = 8;
  int32_t S2 = 8;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  //因为要根据此处的tilingdata来进行kernel的UT, scaleValue固定为1.0和2.0，勿改
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 401);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 402);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 403);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 404);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_41) {
  int32_t B = 2;
  int32_t W = 96;
  int32_t N = 1;
  int32_t S1 = 8;
  int32_t S2 = 8;
  //因为要根据此处的tilingdata来进行kernel的UT, scaleValue固定为1.0和2.0，勿改
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 401);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 402);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 403);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 404);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_42) {
  int32_t B = 2;
  int32_t W = 96;
  int32_t N = 5;
  int32_t S1 = 8;
  int32_t S2 = 7;
  //因为要根据此处的tilingdata来进行kernel的UT, scaleValue固定为1.0和2.0，勿改
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 401);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 402);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 403);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 404);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_20) {
  int32_t B = 2;
  int32_t W = 1;
  int32_t N = 16;
  int32_t S1 = 144;
  int32_t S2 = 144;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 203);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 204);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_21) {
  int32_t B = 2;
  int32_t W = 1;
  int32_t N = 16;
  int32_t S1 = 144;
  int32_t S2 = 145;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 2103);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 2104);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_22) {
  int32_t B = 2;
  int32_t W = 1;
  int32_t N = 1;
  int32_t S1 = 4;
  int32_t S2 = 144;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 203);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 204);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_23) {
  int32_t B = 1;
  int32_t W = 1;
  int32_t N = 1;
  int32_t S1 = 49;
  int32_t S2 = 145;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 2103);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 2104);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_24) {
  int32_t B = 1;
  int32_t W = 1;
  int32_t N = 1;
  int32_t S1 = 1;
  int32_t S2 = 16;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 203);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 204);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_25) {
  int32_t B = 1;
  int32_t W = 1;
  int32_t N = 1;
  int32_t S1 = 1;
  int32_t S2 = 7;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 2103);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 2104);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_20_12) {
  int32_t B = 2;
  int32_t W = 3;
  int32_t N = 5;
  int32_t S1 = 36;
  int32_t S2 = 16;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  //因为要根据此处的tilingdata来进行kernel的UT, scaleValue固定为1.0和2.0，勿改
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 201);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 202);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_20_12_failed) {
  int32_t B = 2;
  int32_t W = 3;
  int32_t N = 5;
  int32_t S1 = 35;
  int32_t S2 = 6200;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTilingFailed(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, ge::DT_FLOAT);

  B = 48;
  W = 1;
  N = 1;
  S1 = 1;
  S2 = 10000;
  x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  bias_shape = {{N, S1, S2}, {N, S1, S2}};
  y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTilingFailed(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, ge::DT_FLOAT);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_21_12) {
  int32_t B = 2;
  int32_t W = 3;
  int32_t N = 5;
  int32_t S1 = 36;
  int32_t S2 = 15;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  //因为要根据此处的tilingdata来进行kernel的UT, scaleValue固定为1.0和2.0，勿改
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 201);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 202);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_2) {
  int32_t B = 1;
  int32_t W = 128;
  int32_t N = 3;
  int32_t S1 = 8;
  int32_t S2 = 8;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  //testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 3);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_3) {
  int32_t B = 20;
  int32_t W = 38;
  int32_t N = 3;
  int32_t S1 = 7;
  int32_t S2 = 7;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  //testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, tiling_4_dim) {
  int32_t B = 2;
  int32_t W = 95;
  int32_t N = 1;
  int32_t S1 = 8;
  int32_t S2 = 8;
  gert::StorageShape x_shape = {{B * W, N, S1, S2}, {B * W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
    //因为要根据此处的tilingdata来进行kernel的UT, scaleValue固定为1.0和2.0，勿改
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, true, 401);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_ShapeCheckFailed) {
  int32_t B = 20;
  int32_t W = 38;
  int32_t N = 0;
  int32_t S1 = 7;
  int32_t S2 = 7;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  //校验xshpae中有0的场景
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);

  N = 3;
  x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  atten_mask_shape = {{W, 0, S2}, {W, 0, S2}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
  atten_mask_shape = {{W, 5, S2}, {W, 5, S2}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
  atten_mask_shape = {{W, 4, S1, S2}, {W, 4, S1, S2}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
  atten_mask_shape = {{3, W, 4, S1, S2}, {3, W, 4, S1, S2}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);

  atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  bias_shape = {{N, S1, 0}, {N, S1, 0}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
  bias_shape = {{N, S1, 5}, {N, S1, 5}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
  bias_shape = {{4, N, S1, 5}, {4, N, S1, 5}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
  atten_mask_shape = {{3, 4, N, S1, S2}, {3, 4, N, S1, S2}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, input_output_ShapeCheckFailed) {
  int32_t B = 20;
  int32_t W = 38;
  int32_t N = 3;
  int32_t S1 = 7;
  int32_t S2 = 7;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2 + 1}, {B, W, N, S1, S2 + 1}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, input_output_ShapeCheckFailed2) {
  int32_t B = 20;
  int32_t W = 38;
  int32_t N = 3;
  int32_t S1 = 7;
  int32_t S2 = 7;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B * W + 1, N, S1, S2}, {B * W + 1, N, S1, S2}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, input_output_ShapeCheckFailed3) {
  int32_t B = 20;
  int32_t W = 38;
  int32_t N = 3;
  int32_t S1 = 7;
  int32_t S2 = 7;
  gert::StorageShape x_shape = {{B * W, N, S1, S2}, {B * W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B * W * N, S1, S2}, {B * W * N, S1, S2}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, input_atten_ShapeCheckFailed) {
  int32_t S1 = 14;
  int32_t S2 = 16;
  gert::StorageShape x_shape = {{2, 4, S1, S2}, {2, 4, S1, S2}};
  gert::StorageShape atten_mask_shape = {{3, S1, S2}, {3, S1, S2}};
  gert::StorageShape bias_shape = {{4, S1, S2}, {4, S1, S2}};
  gert::StorageShape y_shape = {{2, 4, S1, S2}, {2, 4, S1, S2}};
  testTilingShapeCheckFailed(x_shape, atten_mask_shape, bias_shape, y_shape);
}

//tiling失败，超过UB空间
TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_failed) {
  int32_t B = 48;
  int32_t W = 2;
  int32_t N = 3;
  int32_t S1 = 256;
  int32_t S2 = 256;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  map<string, string> socInfos;
  map<string, string> aicoreSpec;
  map<string, string> intrinsics;
  string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,"CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
    // compile info
  struct MaskedSoftmaxWithRelPosBiasCompileInfo {};
  MaskedSoftmaxWithRelPosBiasCompileInfo compile_info;

  std::string op_type("MaskedSoftmaxWithRelPosBias");
  //ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  //ASSERT_NE(param, nullptr);

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x_shape, &atten_mask_shape, &bias_shape})
                    .OutputShapes({&y_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .SetOpType(op_type)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  //ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  //ASSERT_EQ(Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_s2_1) {
  int32_t B = 4;
  int32_t W = 256;
  int32_t N = 128;
  int32_t S1 = 1;
  int32_t S2 = 1;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  //因为要根据此处的tilingdata来进行kernel的UT, scaleValue固定为1.0和2.0，勿改
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 402);
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_with_tail) {
  int32_t B = 64;
  int32_t W = 5;
  int32_t N = 1;
  int32_t S1 = 1;
  int32_t S2 = 128;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, { N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 0.125, true, 5001);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, true, 5002);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 0.125, false, 5003);
  testTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 5004);
}

void test310PTiling(gert::StorageShape &x_shape,
                gert::StorageShape &atten_mask_shape,
                gert::StorageShape &bias_shape, gert::StorageShape &y_shape,
                float scaleValue, bool hasAtten, uint32_t key) {
  string compile_info_string = R"({
        "hardware_info": {"CORE_NUM": 8, "L2_SIZE": 16777216, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536,
        "L0C_SIZE": 262144, "UB_SIZE": 262144, "BT_SIZE": 0, "cube_vector_split_bool": false,
        "ddr_read_rate": 17, "ddr_write_rate": 17, "l2_rate": 114, "l2_read_rate": 114, "l2_write_rate": 114, "l1_to_l0_a_rate": 512,
        "l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 114,
        "ub_to_ddr_rate": 17, "ub_to_l1_rate": 256, "cube_bandwidth": 0, "vector_bandwidth": 0, "vector_core_cnt": 7}})";

  map<string, string> socInfos;
  map<string, string> aicoreSpec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct MaskedSoftmaxWithRelPosBiasCompileInfo {};
  MaskedSoftmaxWithRelPosBiasCompileInfo compile_info;

  std::string op_type("MaskedSoftmaxWithRelPosBias");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
            nullptr);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size =
      reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);

  ge::DataType testDataType[] = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
  for (uint32_t i = 0; i < sizeof(testDataType) / sizeof(ge::DataType); i++) {
    gert::KernelRunContextHolder holder;
    if (hasAtten) {
      holder =
          gert::TilingContextFaker()
              .NodeIoNum(3, 1)
              .IrInstanceNum({1, 1, 1})
              .InputShapes({&x_shape, &atten_mask_shape, &bias_shape})
              .OutputShapes({&y_shape})
              .CompileInfo(&compile_info)
              .PlatformInfo(reinterpret_cast<char *>(&platform_info))
              .NodeInputTd(0, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeInputTd(1, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeInputTd(2, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeOutputTd(0, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeAttrs({
                  {"scale_value", Ops::NN::AnyValue::CreateFrom<float>(scaleValue)},
                  {"inner_precision_mode",
                   Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
              })
              .TilingData(param.get())
              .Workspace(ws_size)
              .SetOpType(op_type)
              .Build();
    } else {
      holder =
          gert::TilingContextFaker()
              .NodeIoNum(3, 1)
              .IrInstanceNum({1, 1, 1})
              .InputShapes({&x_shape, nullptr, &bias_shape})
              .OutputShapes({&y_shape})
              .CompileInfo(&compile_info)
              .PlatformInfo(reinterpret_cast<char *>(&platform_info))
              .NodeInputTd(0, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeInputTd(1, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeInputTd(2, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeOutputTd(0, testDataType[i], ge::FORMAT_ND, ge::FORMAT_ND)
              .NodeAttrs({
                  {"scale_value", Ops::NN::AnyValue::CreateFrom<float>(scaleValue)},
                  {"inner_precision_mode",
                   Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
              })
              .TilingData(param.get())
              .Workspace(ws_size)
              .SetOpType(op_type)
              .Build();
    }

    gert::TilingContext *tiling_context =
        holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    map<string, string> socversions = {{"Short_SoC_version", "Ascend310P"}, {"NpuArch", "2202"}};
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions); // label:"version" res:socversions
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(tiling_context),
              ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, key + 10 * i);
    auto workspace_size_ptr = tiling_context->GetWorkspaceSizes(1);
    ASSERT_NE(workspace_size_ptr, nullptr);
    ASSERT_EQ(workspace_size_ptr[0], 0);
  }
}

TEST_F(MaskedSoftmaxWithRelPosBiasTiling, B_tiling_310p_20) {
  int32_t B = 2;
  int32_t W = 1;
  int32_t N = 16;
  int32_t S1 = 144;
  int32_t S2 = 144;
  gert::StorageShape x_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  gert::StorageShape atten_mask_shape = {{W, S1, S2}, {W, S1, S2}};
  gert::StorageShape bias_shape = {{N, S1, S2}, {N, S1, S2}};
  gert::StorageShape y_shape = {{B, W, N, S1, S2}, {B, W, N, S1, S2}};
  test310PTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 2.0, false, 203);
  test310PTiling(x_shape, atten_mask_shape, bias_shape, y_shape, 1.0, false, 204);
}
