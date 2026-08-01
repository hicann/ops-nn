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
#include "../../../../op_host/arch35/add_rms_norm_dynamic_quant_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AddRmsNormDynamicQuantTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddRmsNormDynamicQuantTilingArch35 SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddRmsNormDynamicQuantTilingArch35 TearDown" << std::endl; }
};

static void ExecuteTestCase(gert::StorageShape input_shape, gert::StorageShape gamma_shape,
                            gert::StorageShape out_shape, gert::StorageShape reduce_shape, int num_inputs,
                            ge::DataType input_dtype, ge::DataType output_dtype, std::vector<bool> output_mask,
                            int64_t dst_type, int expected_tiling_key, float epsilon = 0.01f,
                            ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
 "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                   "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                   "UB_SIZE": 254976, "L2_SIZE": 33554432, "L1_SIZE": 524288,
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
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = [&]() {
        if (num_inputs == 6) {
            return gert::TilingContextFaker()
                .SetOpType(op_type)
                .NodeIoNum(6, 5)
                .IrInstanceNum({1, 1, 1, 1, 1, 1})
                .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                .CompileInfo(&compile_info)
                .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                .NodeInputTd(0, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)},
                            {"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)},
                            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dst_type)}})
                .TilingData(param.get())
                .Workspace(ws_size)
                .Build();
        } else if (num_inputs == 5) {
            return gert::TilingContextFaker()
                .SetOpType(op_type)
                .NodeIoNum(5, 5)
                .IrInstanceNum({1, 1, 1, 1, 1})
                .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                .CompileInfo(&compile_info)
                .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                .NodeInputTd(0, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)},
                            {"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)},
                            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dst_type)}})
                .TilingData(param.get())
                .Workspace(ws_size)
                .Build();
        } else {
            return gert::TilingContextFaker()
                .SetOpType(op_type)
                .NodeIoNum(4, 5)
                .IrInstanceNum({1, 1, 1, 1})
                .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                .CompileInfo(&compile_info)
                .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                .NodeInputTd(0, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, output_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, input_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)},
                            {"output_mask", Ops::NN::AnyValue::CreateFrom<std::vector<bool>>(output_mask)},
                            {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dst_type)}})
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

// ========== Input FLOAT16, output INT8 ==========

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_perf_0)
{
    gert::StorageShape input_shape = {{3, 1, 640}, {3, 1, 640}};
    gert::StorageShape gamma_shape = {{640}, {640}};
    gert::StorageShape out_shape = {{3, 1, 640}, {3, 1, 640}};
    gert::StorageShape reduce_shape = {{3, 1}, {3, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_INT8, {true, true}, 2,
                    0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_perf_1)
{
    gert::StorageShape input_shape = {{4096, 1, 6144}, {4096, 1, 6144}};
    gert::StorageShape gamma_shape = {{6144}, {6144}};
    gert::StorageShape out_shape = {{4096, 1, 6144}, {4096, 1, 6144}};
    gert::StorageShape reduce_shape = {{4096, 1}, {4096, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_INT8, {true, true}, 2,
                    0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_norm)
{
    gert::StorageShape input_shape = {{124, 1, 12699}, {124, 1, 12699}};
    gert::StorageShape gamma_shape = {{12699}, {12699}};
    gert::StorageShape out_shape = {{124, 1, 12699}, {124, 1, 12699}};
    gert::StorageShape reduce_shape = {{124, 1}, {124, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 4, ge::DT_FLOAT16, ge::DT_INT8, {true, false}, 2,
                    1);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_single_row)
{
    gert::StorageShape input_shape = {{124, 1, 14000}, {124, 1, 14000}};
    gert::StorageShape gamma_shape = {{14000}, {14000}};
    gert::StorageShape out_shape = {{124, 1, 14000}, {124, 1, 14000}};
    gert::StorageShape reduce_shape = {{124, 1}, {124, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 4, ge::DT_FLOAT16, ge::DT_INT8, {true, false}, 2,
                    2);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_split_0)
{
    gert::StorageShape input_shape = {{3, 1, 20480}, {3, 1, 20480}};
    gert::StorageShape gamma_shape = {{20480}, {20480}};
    gert::StorageShape out_shape = {{3, 1, 20480}, {3, 1, 20480}};
    gert::StorageShape reduce_shape = {{3, 1}, {3, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_INT8, {true, true}, 2,
                    3);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_large_m_small_n)
{
    gert::StorageShape input_shape = {{1000000, 1, 2}, {1000000, 1, 2}};
    gert::StorageShape gamma_shape = {{2}, {2}};
    gert::StorageShape out_shape = {{100000, 1, 2}, {1000000, 1, 2}};
    gert::StorageShape reduce_shape = {{1000000, 1}, {1000000, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_INT8, {true, true}, 2,
                    0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_m_zero_dim)
{
    gert::StorageShape input_shape = {{100, 0, 1}, {100, 0, 1}};
    gert::StorageShape gamma_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{100, 0, 1}, {100, 0, 1}};
    gert::StorageShape reduce_shape = {{100, 0}, {100, 0}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_INT8, {true, true}, 2,
                    1);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_empty_tiling_n_zero_dim)
{
    gert::StorageShape input_shape = {{100, 1, 0}, {100, 1, 0}};
    gert::StorageShape gamma_shape = {{0}, {0}};
    gert::StorageShape out_shape = {{100, 1, 0}, {100, 1, 0}};
    gert::StorageShape reduce_shape = {{100, 1}, {100, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_INT8, {true, true}, 2,
                    4);
}

// ========== Input FLOAT16, output HIFLOAT8 ==========

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_perf_hifloat8)
{
    gert::StorageShape input_shape = {{3, 1, 640}, {3, 1, 640}};
    gert::StorageShape gamma_shape = {{640}, {640}};
    gert::StorageShape out_shape = {{3, 1, 640}, {3, 1, 640}};
    gert::StorageShape reduce_shape = {{3, 1}, {3, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_HIFLOAT8, {true, true},
                    34, 0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_132_hifloat8_1)
{
    gert::StorageShape input_shape = {{4096, 1, 6144}, {4096, 1, 6144}};
    gert::StorageShape gamma_shape = {{6144}, {6144}};
    gert::StorageShape out_shape = {{4096, 1, 6144}, {4096, 1, 6144}};
    gert::StorageShape reduce_shape = {{4096, 1}, {4096, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_HIFLOAT8, {true, true},
                    34, 0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_split_hifloat8)
{
    gert::StorageShape input_shape = {{3, 1, 20480}, {3, 1, 20480}};
    gert::StorageShape gamma_shape = {{20480}, {20480}};
    gert::StorageShape out_shape = {{3, 1, 20480}, {3, 1, 20480}};
    gert::StorageShape reduce_shape = {{3, 1}, {3, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_HIFLOAT8, {true, true},
                    34, 3);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_large_m_small_n_hifloat8)
{
    gert::StorageShape input_shape = {{1000000, 1, 2}, {1000000, 1, 2}};
    gert::StorageShape gamma_shape = {{2}, {2}};
    gert::StorageShape out_shape = {{100000, 1, 2}, {1000000, 1, 2}};
    gert::StorageShape reduce_shape = {{1000000, 1}, {1000000, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_HIFLOAT8, {true, true},
                    34, 0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_zero_dim_hifloat8)
{
    gert::StorageShape input_shape = {{100, 0, 1}, {100, 0, 1}};
    gert::StorageShape gamma_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{100, 0, 1}, {100, 0, 1}};
    gert::StorageShape reduce_shape = {{100, 0}, {100, 0}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_HIFLOAT8, {true, true},
                    34, 1);
}

// ========== Input FLOAT16, output FLOAT8_E5M2 ==========

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_perf_float8_e5m2_0)
{
    gert::StorageShape input_shape = {{3, 1, 640}, {3, 1, 640}};
    gert::StorageShape gamma_shape = {{640}, {640}};
    gert::StorageShape out_shape = {{3, 1, 640}, {3, 1, 640}};
    gert::StorageShape reduce_shape = {{3, 1}, {3, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, {}, 35,
                    0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_perf_float8_e5m2_1)
{
    gert::StorageShape input_shape = {{4096, 1, 6144}, {4096, 1, 6144}};
    gert::StorageShape gamma_shape = {{6144}, {6144}};
    gert::StorageShape out_shape = {{4096, 1, 6144}, {4096, 1, 6144}};
    gert::StorageShape reduce_shape = {{4096, 1}, {4096, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, {}, 35,
                    0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_split_float8_e5m2)
{
    gert::StorageShape input_shape = {{3, 1, 20480}, {3, 1, 20480}};
    gert::StorageShape gamma_shape = {{20480}, {20480}};
    gert::StorageShape out_shape = {{3, 1, 20480}, {3, 1, 20480}};
    gert::StorageShape reduce_shape = {{3, 1}, {3, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, {}, 35,
                    3);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_large_m_small_n_float8_e5m2)
{
    gert::StorageShape input_shape = {{1000000, 1, 2}, {1000000, 1, 2}};
    gert::StorageShape gamma_shape = {{2}, {2}};
    gert::StorageShape out_shape = {{100000, 1, 2}, {1000000, 1, 2}};
    gert::StorageShape reduce_shape = {{1000000, 1}, {1000000, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, {}, 35,
                    0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_zero_dim_float8_e5m2)
{
    gert::StorageShape input_shape = {{5000, 1, 0}, {5000, 1, 0}};
    gert::StorageShape gamma_shape = {{0}, {0}};
    gert::StorageShape out_shape = {{5000, 1, 0}, {5000, 1, 0}};
    gert::StorageShape reduce_shape = {{5000, 1}, {5000, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, {}, 35,
                    4);
}

// ========== Input FLOAT16, output FLOAT8_E4M3FN ==========

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_perf_float8_e4m3fn_0)
{
    gert::StorageShape input_shape = {{3, 1, 640}, {3, 1, 640}};
    gert::StorageShape gamma_shape = {{640}, {640}};
    gert::StorageShape out_shape = {{3, 1, 640}, {3, 1, 640}};
    gert::StorageShape reduce_shape = {{3, 1}, {3, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN,
                    {true, true}, 36, 0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_perf_float8_e4m3fn_1)
{
    gert::StorageShape input_shape = {{9000, 1, 9000}, {9000, 1, 9000}};
    gert::StorageShape gamma_shape = {{9000}, {9000}};
    gert::StorageShape out_shape = {{9000, 1, 9000}, {9000, 1, 9000}};
    gert::StorageShape reduce_shape = {{9000, 1}, {9000, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN,
                    {true, true}, 36, 0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_split_float8_e4m3fn)
{
    gert::StorageShape input_shape = {{3, 1, 20480}, {3, 1, 20480}};
    gert::StorageShape gamma_shape = {{20480}, {20480}};
    gert::StorageShape out_shape = {{3, 1, 20480}, {3, 1, 20480}};
    gert::StorageShape reduce_shape = {{3, 1}, {3, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN,
                    {true, true}, 36, 3);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35,
       add_rms_norm_dynamic_quant_tiling_regbase_tiling_large_m_small_n_float8_e4m3fn)
{
    gert::StorageShape input_shape = {{1000000, 1, 2}, {1000000, 1, 2}};
    gert::StorageShape gamma_shape = {{2}, {2}};
    gert::StorageShape out_shape = {{100000, 1, 2}, {1000000, 1, 2}};
    gert::StorageShape reduce_shape = {{1000000, 1}, {1000000, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN,
                    {true, true}, 36, 0);
}

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_zero_dim_float8_e4m3fn)
{
    gert::StorageShape input_shape = {{1000000, 1, 0}, {1000000, 1, 0}};
    gert::StorageShape gamma_shape = {{0}, {0}};
    gert::StorageShape out_shape = {{1000000, 1, 0}, {1000000, 1, 0}};
    gert::StorageShape reduce_shape = {{1000000, 1}, {1000000, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 5, ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN,
                    {true, true}, 36, 4);
}

// ========== With beta input (6 inputs) ==========

TEST_F(AddRmsNormDynamicQuantTilingArch35, add_rms_norm_dynamic_quant_tiling_regbase_tiling_with_beta)
{
    gert::StorageShape input_shape = {{4096, 1, 9703}, {4096, 1, 9703}};
    gert::StorageShape gamma_shape = {{9703}, {9703}};
    gert::StorageShape out_shape = {{4096, 1, 9703}, {4096, 1, 9703}};
    gert::StorageShape reduce_shape = {{4096, 1}, {4096, 1}};

    ExecuteTestCase(input_shape, gamma_shape, out_shape, reduce_shape, 6, ge::DT_FLOAT16, ge::DT_INT8, {true, true}, 2,
                    0);
}
