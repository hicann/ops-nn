/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/avg_pool_cube_tiling.h"

using namespace std;
using namespace ge;

struct AvgPool3DTilingRuntime2TestParam {
    string case_name;
    string compile_info;

    std::initializer_list<int64_t> x_shape;
    std::initializer_list<int64_t> y_shape;

    ge::Format x_format;
    ge::Format y_format;

    bool parse_result;
    bool tiling_result;

    // output
    uint32_t block_dim;
    uint64_t tiling_key;
    std::string tiling_data;
};

class AvgPool3DTilingTest : public testing::TestWithParam<AvgPool3DTilingRuntime2TestParam> {
  protected:
      static void SetUpTestCase() {
          std::cout << "AvgPool3DTilingTest SetUp" << std::endl;
      }

      static void TearDownTestCase() {
          std::cout << "AvgPool3DTilingTest TearDown" << std::endl;
      }
};

static string TilingData2Str(const gert::TilingData *tiling_data) {
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t *>(tiling_data->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }

    return result;
}

TEST_P(AvgPool3DTilingTest, general_cases) {
    AvgPool3DTilingRuntime2TestParam param = GetParam();
    std::cout << "run case " << param.case_name << std::endl;

    gert::StorageShape x_shape = {param.x_shape, param.x_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.y_shape, param.y_shape});
    std::vector<void *> output_shapes_ref(1);

    for (size_t i = 0; i < output_shapes.size(); ++i) {
        output_shapes_ref[i] = &output_shapes[i];
    }

    fe::PlatFormInfos platform_info;
    optiling::avgPool3DTilingCompileInfo::AvgPool3DCubeCompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                      .KernelIONum(2, 1)
                      .Inputs({const_cast<char *>(param.compile_info.c_str()), reinterpret_cast<void *>(&platform_info)})
                      .Outputs({&compile_info})
                      .Build();

    std::string op_type("AvgPool3D");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    if (param.parse_result) {
        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    } else {
        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_FAILED);
        return;
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes(output_shapes_ref)
                      .NodeInputTd(0, ge::DT_FLOAT16, param.x_format, param.x_format)
                      .NodeOutputTd(0, ge::DT_FLOAT16, param.y_format, param.y_format)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .TilingData(tiling_data.get())
                      .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    if (param.tiling_result) {
        ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    } else {
        ASSERT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
      return;
    }
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    ASSERT_EQ(tiling_key, param.tiling_key);
    ASSERT_EQ(block_dim, param.block_dim);
    ASSERT_EQ(tiling_data_result, param.tiling_data);
}

static AvgPool3DTilingRuntime2TestParam general_cases_params[] = {
    {"avg_pool3d_tiling_dynamic_batch_invalid_dim", R"({"_pattern": "conv3d", "push_status": 0, "fmap_c1": 233, "tiling_type": "dynamic_tiling", "repo_seeds": {"10000": 32}, "tiling_range": {"10000": [1, 35]}, "block_dim": {"10000": 32}, "_vars": {"10000": ["batch_n"]}})",
      {32, 16, 1, 56, 56}, {32, 15, 1, 56, 56, 16},
      ge::FORMAT_NDHWC, ge::FORMAT_NDC1HWC0,
      true, false, 32, 10000, "56 56 "
    },
};

INSTANTIATE_TEST_CASE_P(AvgPool3D, AvgPool3DTilingTest, testing::ValuesIn(general_cases_params));

static void Execute310PTestCase(gert::StorageShape xShape,
                            gert::StorageShape yShape, std::vector<int64_t> ksize,
                            std::vector<int64_t> strides, std::vector<int64_t> pads, bool ceil_mode, bool count_include_pad, 
                            int64_t divisor_override, std::string data_format, ge::DataType dtype,
                            uint64_t except_tilingkey, std::string expect) {
    dlog_setlevel(0, 0, 0);

    string compile_info_string = R"({
        "hardware_info": {"CORE_NUM": 8, "L2_SIZE": 16777216, "L1_SIZE": 1048576, "L0A_SIZE": 65536, "L0B_SIZE": 65536,
        "L0C_SIZE": 262144, "UB_SIZE": 262144, "BT_SIZE": 0, "cube_vector_split_bool": false,
        "ddr_read_rate": 17, "ddr_write_rate": 17, "l2_rate": 114, "l2_read_rate": 114, "l2_write_rate": 114, "l1_to_l0_a_rate": 512,
        "l1_to_l0_b_rate": 256, "l1_to_ub_rate": 128, "l0_c_to_ub_rate": 256, "ub_to_l2_rate": 114,
        "ub_to_ddr_rate": 17, "ub_to_l1_rate": 256, "cube_bandwidth": 0, "vector_bandwidth": 0, "vector_core_cnt": 7}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend310P"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::avgPool3DTilingCompileInfo::AvgPool3DCubeCompileInfo compile_info;

    std::string op_type("AvgPool3D");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
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
                                    {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                    {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceil_mode)},
                                    {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(count_include_pad)},
                                    {"divisor_override", Ops::NN::AnyValue::CreateFrom<int64_t>(divisor_override)},
                                    {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}
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
}

class AvgPool3DTiling310PTest : public testing::TestWithParam<AvgPool3DTilingRuntime2TestParam> {
  protected:
      static void SetUpTestCase() {
          std::cout << "AvgPool3DTiling310PTest SetUp" << std::endl;
      }

      static void TearDownTestCase() {
          std::cout << "AvgPool3DTiling310PTest TearDown" << std::endl;
      }
};

TEST_F(AvgPool3DTiling310PTest, AvgPool3DTiling_NCDHW_Kernel1) {
    gert::StorageShape xShape = {{20, 16, 50, 44, 31}, {20, 16, 50, 44, 31}};
    gert::StorageShape yShape = {{20, 16, 25, 22, 15}, {20, 16, 25, 22, 15}};
    std::vector<int64_t> ksize = {2};
    std::vector<int64_t> strides = {2};
    std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
    bool ceil_mode = false;
    bool count_include_pad = true;
    int64_t divisor_override = 0;
    ge::DataType dtype = ge::DT_FLOAT;

    std::string data_format = "NCDHW";
    uint64_t except_tilingkey = 50;
    std::string expect = "";
    Execute310PTestCase(xShape, yShape, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override, data_format,
        dtype, except_tilingkey, expect);
}