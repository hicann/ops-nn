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
 * \file test_max_pool_v3_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/max_pool_v3_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MaxPoolV3Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolV3Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolV3Tiling TearDown" << std::endl;
  }
};

template <typename T>
static string to_string(void* buf, size_t size) {
  std::string result;
  const T* data = reinterpret_cast<const T*>(buf);
  size_t len = size / sizeof(T);
  for (size_t i = 0; i < len; i++) {
    result += std::to_string(data[i]);
    result += " ";
  }
  return result;
}

static void ExecuteTestCase(gert::StorageShape xShape,
                            gert::StorageShape yShape, std::vector<int64_t> ksize,
                            std::vector<int64_t> strides, std::string padding_mode, std::vector<int64_t> pads,
                            std::string data_format, bool global_pooling, bool ceil_mode, ge::DataType dtype,
                            uint64_t except_tilingkey, std::string expect) {
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
  std::map<std::string, std::string> soc_version_infos = { { "Short_SoC_version", "Ascend910_95" } };

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  optiling::MaxPoolV3CompileInfo compile_info;

  std::string op_type("MaxPoolV3");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
          .Outputs({&compile_info})
          .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                          intrinsics);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&xShape})
                    .OutputShapes({&yShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                    .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                {"padding_mode", Ops::NN::AnyValue::CreateFrom<std::string>(padding_mode)},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                                {"global_pooling", Ops::NN::AnyValue::CreateFrom<bool>(global_pooling)},
                                {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceil_mode)}
                                })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
  auto tiling_key = tiling_context->GetTilingKey();
  ASSERT_EQ(tiling_key, except_tilingkey);
  auto tilingData = tiling_context->GetRawTilingData();
  ASSERT_NE(tilingData, nullptr);
  auto tiling_data_result = to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize());
  std::cout<<tiling_data_result<<std::endl;
  EXPECT_EQ(tiling_data_result, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test0) {
  gert::StorageShape xShape = {{4, 163, 1024, 600}, {4, 163, 1024, 600}};
  gert::StorageShape yShape = {{4, 163, 2, 2}, {4, 163, 2, 2}};
  std::vector<int64_t> ksize = {1, 1, 324, 457};
  std::vector<int64_t> strides = {1, 1, 858, 457};
  std::vector<int64_t> pads = {30, 30, 132, 132};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = true;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 311110;
  std::string expect = "1024 600 2 2 457 324 457 858 30 30 132 132 40 48 2608 64 30208 0 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}