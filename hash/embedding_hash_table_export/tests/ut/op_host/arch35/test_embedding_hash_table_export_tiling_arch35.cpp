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
 * \file test_embedding_hash_table_export_tiling.cpp
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
#include "../../../op_host/arch35/embedding_hash_table_export_tiling_arch35.h"

using namespace std;

struct EmbeddingHashTableExportData {
  // attrs
  string export_mode{"all"};
  bool filtered_export_flag{true};

  // inputs
  gert::StorageShape in_shape{{1}, {1}};

  // outputs
  gert::StorageShape out_shape{{20}, {20}};

  // test debug info
  string debug_info{"tiling_info:"};

  // expect
  ge::graphStatus expect_status{ge::GRAPH_FAILED};
  uint64_t expect_tiling_key{4};
  // others
  bool check_tiling_key{true};
};

class TilingEmbeddingHashTableExport : public ::testing::TestWithParam<EmbeddingHashTableExportData> {
 protected:
  void SetUp() override {
    std::cout << "TilingEmbeddingHashTableExport SetUp" << std::endl;
  }

  void TearDown() override {
    std::cout << "TilingEmbeddingHashTableExport TearDown" << std::endl;
  }
};

TEST_P(TilingEmbeddingHashTableExport, embedding_hash_table_export_tiling) {
  string compile_info_string = R"({
              "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
              "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
              "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
              "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
              "CORE_NUM": 64}
              })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();

  // compile info
  optiling::EmbeddingHashTableExportCompileInfo compile_info;

  string op_type("EmbeddingHashTableExport");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder =
      gert::KernelRunContextFaker()
          .KernelIONum(2, 1)
          .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
          .Outputs({&compile_info})
          .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                          intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

  auto test_params = GetParam();
  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder =
      gert::TilingContextFaker()
          .SetOpType("EmbeddingHashTableExport")
          .NodeIoNum(4, 4)
          .IrInstanceNum({1, 1, 1, 1})
          .InputShapes({&test_params.in_shape, &test_params.in_shape, &test_params.in_shape, &test_params.in_shape})
          .OutputShapes({&test_params.out_shape, &test_params.out_shape,&test_params.out_shape,&test_params.out_shape})
          .CompileInfo(&compile_info)
          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
          .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(1, ge::DT_UINT64, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(2, ge::DT_UINT8, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
          .NodeAttrs({{"export_mode", Ops::NN::AnyValue::CreateFrom<string>(test_params.export_mode)},
                      {"filtered_export_flag", Ops::NN::AnyValue::CreateFrom<bool>(test_params.filtered_export_flag)}})
          .TilingData(param.get())
          .Workspace(ws_size)
          .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // check tiling result
  ge::graphStatus actual_staus = tiling_func(tiling_context);
  EXPECT_EQ(actual_staus, test_params.expect_status) << test_params.debug_info;

  if (test_params.check_tiling_key) {
    auto actual_tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(actual_tiling_key, test_params.expect_tiling_key) << test_params.debug_info;
  }

}

const auto EmbeddingHashTableExportTestCases = ::testing::Values(
        EmbeddingHashTableExportData{
        "all",
        true,  // attr
        {{1}, {1}},  // input shape
        {{1024}, {1024}},  // output shape
        "embedding_hash_table_export_test_case1",  // debug info
        ge::GRAPH_SUCCESS,
        4,             // expect
        false  // others
    });

INSTANTIATE_TEST_SUITE_P(EmbeddingHashTableExportTilingCases, TilingEmbeddingHashTableExport, EmbeddingHashTableExportTestCases);