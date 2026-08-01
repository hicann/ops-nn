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
 * \file test_add_rms_norm_dynamic_quant_tiling.cpp
 * \brief
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../../op_host/arch22/add_rms_norm_dynamic_quant_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AddRmsNormDynamicQuantTilingArch22 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddRmsNormDynamicQuantTilingArch22 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddRmsNormDynamicQuantTilingArch22 TearDown" << std::endl; }
};

static void ExecuteTestCase(gert::StorageShape input_shape, gert::StorageShape gamma_shape,
                            gert::StorageShape out_shape, gert::StorageShape reduce_shape, int num_inputs,
                            ge::DataType output_dtype, int expected_tiling_key, float epsilon = 0.01f,
                            ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
   "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                     "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                     "CORE_NUM": 40}
                     })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::AddRmsNormDynamicQuantCompileInfo compile_info;

    std::string op_type("AddRmsNormDynamicQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = [&]() {
        if (num_inputs == 6) {
            return gert::TilingContextFaker()
                .NodeIoNum(6, 5)
                .IrInstanceNum({1, 1, 1, 1, 1, 1})
                .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                .CompileInfo(&compile_info)
                .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)}})
                .TilingData(param.get())
                .Workspace(ws_size)
                .Build();
        } else {
            return gert::TilingContextFaker()
                .NodeIoNum(5, 5)
                .IrInstanceNum({1, 1, 1, 1, 1})
                .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                .CompileInfo(&compile_info)
                .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)}})
                .TilingData(param.get())
                .Workspace(ws_size)
                .Build();
        }
    }();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, expected_tiling_key);
}

// ========== Output INT8 ==========

TEST_F(AddRmsNormDynamicQuantTilingArch22, add_rms_norm_dynamic_quant_tiling_001)
{
    gert::StorageShape input_shape = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape reduce_shape = {{1, 1, 1}, {1, 1, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_INT8, 1);
}

TEST_F(AddRmsNormDynamicQuantTilingArch22, add_rms_norm_dynamic_quant_tiling_002)
{
    gert::StorageShape input_shape = {{1, 1, 30000}, {1, 1, 30000}};
    gert::StorageShape gamma_shape = {{30000}, {30000}};
    gert::StorageShape out_shape = {{1, 1, 30000}, {1, 1, 30000}};
    gert::StorageShape reduce_shape = {{1, 1, 1}, {1, 1, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_INT8, 3);
}

// ========== Output INT4 ==========

TEST_F(AddRmsNormDynamicQuantTilingArch22, add_rms_norm_dynamic_quant_tiling_int4_001)
{
    gert::StorageShape input_shape = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape reduce_shape = {{1, 1, 1}, {1, 1, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_INT4, 1);
}

TEST_F(AddRmsNormDynamicQuantTilingArch22, add_rms_norm_dynamic_quant_tiling_int4_002)
{
    gert::StorageShape input_shape = {{1, 1, 30000}, {1, 1, 30000}};
    gert::StorageShape gamma_shape = {{30000}, {30000}};
    gert::StorageShape out_shape = {{1, 1, 30000}, {1, 1, 30000}};
    gert::StorageShape reduce_shape = {{1, 1, 1}, {1, 1, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_INT4, 3);
}

TEST_F(AddRmsNormDynamicQuantTilingArch22, add_rms_norm_dynamic_quant_tiling_with_beta)
{
    gert::StorageShape input_shape = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape gamma_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{1, 1, 16}, {1, 1, 16}};
    gert::StorageShape reduce_shape = {{1, 1, 1}, {1, 1, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 6, ge::DT_INT8, 1);
}
