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
#include "register/op_impl_registry.h"
#include "array_ops.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace optiling {
struct MaxPoolV3CompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};
} // namespace optiling

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy_s(&tmp, sizeof(tmp), data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

template <typename T>
static string to_string(void *buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

class MaxPoolV3Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MaxPoolV3Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPoolV3Tiling TearDown" << std::endl;
  }
};

static void ExecuteTestCase(gert::StorageShape xShape,
                            gert::StorageShape yShape, std::vector<int64_t> ksize,
                            std::vector<int64_t> strides, std::string padding_mode, std::vector<int64_t> pads,
                            std::string data_format, bool global_pooling, bool ceil_mode, ge::DataType dtype,
                            uint64_t except_tilingkey, std::string expect) {
  dlog_setlevel(0, 0, 0);

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
  dlog_setlevel(0, 3, 0);
  auto tiling_data_result = to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize());
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

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test1) {
  gert::StorageShape xShape = {{4, 833, 1024, 640}, {4, 833, 1024, 640}};
  gert::StorageShape yShape = {{4, 833, 3, 3}, {4, 833, 3, 3}};
  std::vector<int64_t> ksize = {1, 1, 455, 513};
  std::vector<int64_t> strides = {1, 1, 455, 256};
  std::vector<int64_t> pads = {106, 106, 163, 163};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = true;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 311110;
  std::string expect = "1024 640 3 3 513 455 256 455 106 106 163 163 468 36 29988 64 30208 0 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test2)
{
    gert::StorageShape xShape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
    gert::StorageShape yShape = {{4, 16, 16, 16}, {4, 16, 16, 16}};
    std::vector<int64_t> ksize = {1, 1, 2, 2};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::vector<int64_t> pads = {0, 0, 0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;

    bool ceil_mode = true;
    std::string data_format = "NCHW";
    std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
    uint64_t except_tilingkey = 300001;
    std::string expect = "32 32 64 16 16 2 2 2 2 0 0 1 0 1 16 16 64 1 1 1024 256 256 1 1 2 3 ";
    ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
      dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test3)
{
    gert::StorageShape xShape = {{4, 16, 30, 30}, {4, 16, 30, 30}};
    gert::StorageShape yShape = {{4, 16, 16, 16}, {4, 16, 16, 16}};
    std::vector<int64_t> ksize = {1, 1, 2, 2};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::vector<int64_t> pads = {1, 1, 1, 1};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;

    bool ceil_mode = false;
    std::string data_format = "NCHW";
    std::string padding_mode = "CALCULATED";
    bool global_pooling = false;
    uint64_t except_tilingkey = 300002;
    std::string expect = "30 30 64 16 16 2 2 2 2 1 1 1 0 1 16 16 64 1 1 1024 256 256 1 1 2 3 ";
    ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
      dtype, except_tilingkey, expect);
}
TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test4) {
  gert::StorageShape xShape = {{1, 1, 68, 22}, {1, 1, 68, 22}};
  gert::StorageShape yShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
  std::vector<int64_t> ksize = {1, 1, 64, 16};
  std::vector<int64_t> strides = {1, 1, 2, 2};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = true;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 311110;
  std::string expect = "68 22 3 4 16 64 2 2 0 0 0 0 0 12 12 12 30208 0 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test5) {
  gert::StorageShape xShape = {{1, 1, 4, 522}, {1, 1, 4, 522}};
  gert::StorageShape yShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
  std::vector<int64_t> ksize = {1, 1, 2, 516};
  std::vector<int64_t> strides = {1, 1, 2, 2};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = true;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 311110;
  std::string expect = "4 522 2 4 516 2 2 2 0 0 0 0 0 8 8 8 30208 0 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test6) {
  gert::StorageShape xShape = {{1, 1, 68, 22}, {1, 1, 68, 22}};
  gert::StorageShape yShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
  std::vector<int64_t> ksize = {1, 1, 64, 16};
  std::vector<int64_t> strides = {1, 1, 2, 2};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = true;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 311110;
  std::string expect = "68 22 3 4 16 64 2 2 0 0 0 0 0 12 12 12 30208 0 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test7) {
  gert::StorageShape xShape = {{4, 8, 64, 32}, {4, 8, 64, 32}};
  gert::StorageShape yShape = {{4, 8, 22, 11}, {4, 8, 22, 11}};
  std::vector<int64_t> ksize = {1, 1, 3, 3};
  std::vector<int64_t> strides = {1, 1, 3, 3};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = true;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 300002;
  std::string expect = "64 32 32 22 11 3 3 3 3 0 0 1 0 1 20 11 32 2 1 2400 224 256 1 1 2 2 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test8) {
  gert::StorageShape xShape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
  gert::StorageShape yShape = {{4, 16, 33, 33}, {4, 16, 33, 33}};
  std::vector<int64_t> ksize = {1, 1, 2, 2};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = false;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 300002;
  std::string expect = "32 32 64 33 33 2 2 1 1 1 1 1 0 1 33 33 64 1 1 1360 1096 256 1 1 2 3 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test9) {
  gert::StorageShape xShape = {{4, 16, 64, 64}, {4, 16, 64, 64}};
  gert::StorageShape yShape = {{4, 16, 1, 1}, {4, 16, 1, 1}};
  std::vector<int64_t> ksize = {1, 1, 64, 64};
  std::vector<int64_t> strides = {1, 1, 1, 1};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = false;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = true;
  uint64_t except_tilingkey = 311110;
  std::string expect = "64 64 1 1 64 64 64 64 0 0 0 0 1 0 64 64 30208 1 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test10) {
  gert::StorageShape xShape = {{11, 32, 94, 304}, {11, 32, 94, 304}};
  gert::StorageShape yShape = {{11, 32, 2, 28}, {11, 32, 2, 28}};
  std::vector<int64_t> ksize = {1, 1, 25, 19};
  std::vector<int64_t> strides = {1, 1, 55, 11};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_INT64;

  bool ceil_mode = false;
  std::string data_format = "NCHW";
  std::string padding_mode = "SAME";
  bool global_pooling = false;
  uint64_t except_tilingkey = 300002;
  std::string expect = "94 304 352 2 28 19 25 11 55 0 6 11 0 1 1 28 352 2 1 7900 28 2048 3 2 1 2 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test11) {
  gert::StorageShape xShape = {{1, 2, 276, 997}, {1, 2, 276, 997}};
  gert::StorageShape yShape = {{1, 2, 5, 56}, {1, 2, 5, 56}};
  std::vector<int64_t> ksize = {1, 1, 45, 5};
  std::vector<int64_t> strides = {1, 1, 48, 18};
  std::vector<int64_t> pads = {1, 1, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_INT64;

  bool ceil_mode = false;
  std::string data_format = "NCHW";
  std::string padding_mode = "VALID";
  bool global_pooling = false;
  uint64_t except_tilingkey = 300001;
  std::string expect = "276 997 2 5 56 5 45 18 48 0 0 1 6 1 1 9 2 5 7 6840 12 1024 3 2 1 1 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test12) {
  gert::StorageShape xShape = {{16, 16, 260, 260}, {16, 16, 260, 260}};
  gert::StorageShape yShape = {{16, 16, 130, 130}, {16, 16, 130, 130}};
  std::vector<int64_t> ksize = {1, 1, 2, 2};
  std::vector<int64_t> strides = {1, 1, 2, 2};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_INT8;

  bool ceil_mode = false;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 300001;
  std::string expect = "260 260 256 130 130 2 2 2 2 0 0 8 0 1 113 130 256 2 1 65088 14720 256 0 0 1 2 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test13) {
  gert::StorageShape xShape = {{16, 16, 300, 52000}, {16, 16, 300, 52000}};
  gert::StorageShape yShape = {{16, 16, 75, 26000}, {16, 16, 75, 26000}};
  std::vector<int64_t> ksize = {1, 1, 4, 2};
  std::vector<int64_t> strides = {1, 1, 4, 2};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_INT8;

  bool ceil_mode = false;
  std::string data_format = "NCHW";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 300001;
  std::string expect = "300 52000 256 75 26000 2 4 2 4 0 0 1200 0 1 1 8176 256 75 4 65408 8192 256 0 2 1 1 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NCHW_Test14) {
  gert::StorageShape xShape = {{136, 247, 2, 2}, {136, 247, 2, 2}};
  gert::StorageShape yShape = {{136, 247, 1, 1}, {136, 247, 1, 1}};
  std::vector<int64_t> ksize = {1, 1, 52, 2};
  std::vector<int64_t> strides = {1, 1, 38, 61};
  std::vector<int64_t> pads = {0, 0, 28, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_INT16;

  bool ceil_mode = false;
  std::string data_format = "NCHW";
  std::string padding_mode = "SAME";
  bool global_pooling = false;
  uint64_t except_tilingkey = 300001;
  std::string expect = "2 2 33592 1 1 2 2 2 2 0 0 1 0 529 1 1 64 1 1 16928 8464 256 2 1 2 3 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NHWC_Test0) {
  gert::StorageShape xShape = {{1,4,256,128}, {1,4,256,128}};
  gert::StorageShape yShape = {{1, 4, 4, 128}, {1, 4, 4, 128}};
  std::vector<int64_t> ksize = {1, 1, 64, 1};
  std::vector<int64_t> strides = {1, 1, 64, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = false;
  std::string data_format = "NHWC";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 411110;
  std::string expect = "4 256 4 4 64 1 64 1 0 0 0 0 128 0 16 16 16 8192 2112 0 0 16 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NHWC_Test1) {
  gert::StorageShape xShape = {{1,4,256,128}, {1,4,256,128}};
  gert::StorageShape yShape = {{1, 4, 64, 128}, {1, 4, 64, 128}};
  std::vector<int64_t> ksize = {1, 1, 4, 1};
  std::vector<int64_t> strides = {1, 1, 4, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = false;
  std::string data_format = "NHWC";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 400001;
  std::string expect = "4 256 1 4 64 4 1 4 1 0 0 1 0 1 1 4 1 4 16 128 2048 512 1001 2 1 1 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NHWC_Test2) {
  gert::StorageShape xShape = {{1,4,256,128}, {1,4,256,128}};
  gert::StorageShape yShape = {{1, 4, 129, 128}, {1, 4, 129, 128}};
  std::vector<int64_t> ksize = {1, 1, 2, 1};
  std::vector<int64_t> strides = {1, 1, 2, 1};
  std::vector<int64_t> pads = {0, 0, 1, 1};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = false;
  std::string data_format = "NHWC";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 400002;
  std::string expect = "4 256 1 4 129 2 1 2 1 0 1 1 4 1 1 8 1 4 17 128 2048 1024 1001 2 1 1 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NHWC_Test3) {
  gert::StorageShape xShape = {{1,4,256,2}, {1,4,256,2}};
  gert::StorageShape yShape = {{1, 4, 64, 2}, {1, 4, 64, 2}};
  std::vector<int64_t> ksize = {1, 1, 4, 1};
  std::vector<int64_t> strides = {1, 1, 4, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_FLOAT;

  bool ceil_mode = false;
  std::string data_format = "NHWC";
  std::string padding_mode = "CALCULATED";
  bool global_pooling = false;
  uint64_t except_tilingkey = 400001;
  std::string expect = "4 256 1 4 64 4 1 4 1 0 0 0 4 1 1 64 1 4 1 2 512 128 0 2 1 2 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}

TEST_F(MaxPoolV3Tiling, MaxPoolV3Tiling_NHWC_Test4) {
  gert::StorageShape xShape = {{8,6912,226,8}, {8,6912,226,8}};
  gert::StorageShape yShape = {{8,147,5,8}, {8,147,5,8}};
  std::vector<int64_t> ksize = {1, 9, 4, 1};
  std::vector<int64_t> strides = {1, 47, 54, 1};
  std::vector<int64_t> pads = {0, 0, 0, 0};
  std::vector<int64_t> dilation = {1, 1, 1, 1};
  ge::DataType dtype = ge::DT_UINT8;

  bool ceil_mode = false;
  std::string data_format = "NHWC";
  std::string padding_mode = "VALID";
  bool global_pooling = false;
  uint64_t except_tilingkey = 400001;
  std::string expect = "6912 226 8 147 5 4 9 54 47 0 0 18 24 1 1 5 8 147 1 8 16416 64 1 2 1 2 ";
  ExecuteTestCase(xShape, yShape, ksize, strides, padding_mode, pads, data_format, global_pooling, ceil_mode,
    dtype, except_tilingkey, expect);
}