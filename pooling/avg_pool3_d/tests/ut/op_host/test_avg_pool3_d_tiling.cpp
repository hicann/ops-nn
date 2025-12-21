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
