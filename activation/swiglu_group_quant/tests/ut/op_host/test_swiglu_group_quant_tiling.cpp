/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/swiglu_group_quant_tiling.h"
#include "../../../op_graph/swiglu_group_quant_proto.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class SwigluGroupQuantTiling : public testing::Test {
 protected:
  static void SetUpTestCase() { std::cout << "SwigluGroupQuantTiling SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "SwigluGroupQuantTiling TearDown" << std::endl; }
};

static const string COMPILE_INFO_STRING = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 40}
                           })";

// 共享的 tiling 测试环境（平台信息、compile_info、tiling 执行器与 workspace）
using OpImplPtr = decltype(gert::OpImplRegistry::GetInstance().GetOpImpl(""));
struct TilingTestFixture {
  fe::PlatFormInfos platform_info;
  optiling::SwigluGroupQuantCompileInfo compile_info;
  OpImplPtr op_impl = nullptr;
  std::map<string, string> soc_infos;
  std::map<string, string> aicore_spec;
  std::map<string, string> intrinsics;
};

static void PreparePlatform(const string &compileInfoString, fe::PlatFormInfos &platformInfo,
                            map<string, string> &socInfos, map<string, string> &aicoreSpec,
                            map<string, string> &intrinsics) {
  GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
  platformInfo.Init();
}

// 初始化 tiling 测试环境：构建 platform/compile_info，执行 tiling_parse，分配 tiling data 与 workspace
static bool InitTilingTestFixture(TilingTestFixture &fx) {
  PreparePlatform(COMPILE_INFO_STRING, fx.platform_info, fx.soc_infos, fx.aicore_spec, fx.intrinsics);
  std::string op_type("SwigluGroupQuant");
  fx.op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
  if (fx.op_impl == nullptr) {
    return false;
  }
  auto kernel_holder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()), reinterpret_cast<void*>(&fx.platform_info)})
          .Outputs({&fx.compile_info})
          .Build();
  auto* parse_ctx = kernel_holder.GetContext<gert::TilingParseContext>();
  if (!parse_ctx->GetPlatformInfo()->Init()) {
    return false;
  }
  parse_ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", fx.soc_infos);
  parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", fx.aicore_spec);
  parse_ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", fx.intrinsics);
  if (fx.op_impl->tiling_parse(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
    return false;
  }
  return true;
}

// 为 tiling context 设置平台资源
static void SetupTilingContextPlatform(gert::TilingContext* ctx, TilingTestFixture &fx) {
  ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", fx.soc_infos);
  ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", fx.aicore_spec);
  ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", fx.intrinsics);
}

// 构造本算子的属性集合（quant_mode 固定 3，clamp_limit/output_origin 可变）
static std::vector<std::pair<std::string, Ops::NN::AnyValue>> BuildNodeAttrs(float clampLimit, bool outputOrigin) {
  return {{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(27)},
          {"quant_mode", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
          {"block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
          {"round_scale", Ops::NN::AnyValue::CreateFrom<bool>(false)},
          {"clamp_limit", Ops::NN::AnyValue::CreateFrom<float>(clampLimit)},
          {"dst_type_max_finite", Ops::NN::AnyValue::CreateFrom<float>(448.0f)},
          {"output_origin", Ops::NN::AnyValue::CreateFrom<bool>(outputOrigin)}};
}

TEST_F(SwigluGroupQuantTiling, swiglu_group_quant_tiling_basic) {
  TilingTestFixture fx;
  ASSERT_TRUE(InitTilingTestFixture(fx));
  auto param = gert::TilingData::CreateCap(4096);
  auto ws_holder = gert::ContinuousVector::Create<size_t>(4096);
  auto* ws_size = reinterpret_cast<gert::ContinuousVector*>(ws_holder.get());

  gert::StorageShape x_shape = {{128, 2048}, {128, 2048}};
  gert::StorageShape y_shape = {{128, 1024}, {128, 1024}};
  gert::StorageShape y_scale_shape = {{1}, {1}};

  auto holder = gert::TilingContextFaker()
                    .SetOpType("SwigluGroupQuant")
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 0, 0, 0})
                    .InputShapes({&x_shape})
                    .OutputShapes({&y_shape, &y_scale_shape})
                    .CompileInfo(&fx.compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&fx.platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs(BuildNodeAttrs(0.0f, false))
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  SetupTilingContextPlatform(tiling_context, fx);
  EXPECT_EQ(fx.op_impl->tiling(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SwigluGroupQuantTiling, swiglu_group_quant_tiling_with_weight) {
  TilingTestFixture fx;
  ASSERT_TRUE(InitTilingTestFixture(fx));
  auto param = gert::TilingData::CreateCap(4096);
  auto ws_holder = gert::ContinuousVector::Create<size_t>(4096);
  auto* ws_size = reinterpret_cast<gert::ContinuousVector*>(ws_holder.get());

  gert::StorageShape x_shape = {{64, 2048}, {64, 2048}};
  gert::StorageShape weight_shape = {{64, 1}, {64, 1}};
  gert::StorageShape y_shape = {{64, 1024}, {64, 1024}};
  gert::StorageShape y_scale_shape = {{1}, {1}};

  auto holder = gert::TilingContextFaker()
                    .SetOpType("SwigluGroupQuant")
                    .NodeIoNum(4, 2)
                    .IrInstanceNum({1, 1, 0, 0})
                    .InputShapes({&x_shape, &weight_shape})
                    .OutputShapes({&y_shape, &y_scale_shape})
                    .CompileInfo(&fx.compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&fx.platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs(BuildNodeAttrs(0.0f, false))
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  SetupTilingContextPlatform(tiling_context, fx);
  EXPECT_EQ(fx.op_impl->tiling(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(SwigluGroupQuantTiling, swiglu_group_quant_tiling_with_clamp_origin) {
  TilingTestFixture fx;
  ASSERT_TRUE(InitTilingTestFixture(fx));
  auto param = gert::TilingData::CreateCap(4096);
  auto ws_holder = gert::ContinuousVector::Create<size_t>(4096);
  auto* ws_size = reinterpret_cast<gert::ContinuousVector*>(ws_holder.get());

  gert::StorageShape x_shape = {{128, 2048}, {128, 2048}};
  gert::StorageShape y_shape = {{128, 1024}, {128, 1024}};
  gert::StorageShape y_scale_shape = {{1}, {1}};
  gert::StorageShape y_origin_shape = {{128, 1024}, {128, 1024}};

  auto holder = gert::TilingContextFaker()
                    .SetOpType("SwigluGroupQuant")
                    .NodeIoNum(4, 3)
                    .IrInstanceNum({1, 0, 0, 0})
                    .InputShapes({&x_shape})
                    .OutputShapes({&y_shape, &y_scale_shape, &y_origin_shape})
                    .CompileInfo(&fx.compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&fx.platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs(BuildNodeAttrs(7.0f, true))
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context, nullptr);
  SetupTilingContextPlatform(tiling_context, fx);
  EXPECT_EQ(fx.op_impl->tiling(tiling_context), ge::GRAPH_SUCCESS);
}
