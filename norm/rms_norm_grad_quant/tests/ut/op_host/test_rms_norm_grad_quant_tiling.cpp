/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "test_rms_norm_grad_quant_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class RmsNormGradQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmsNormGradQuantTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RmsNormGradQuantTilingTest TearDown" << std::endl; }
};

template <typename T>
static string to_string(void* buf, size_t size)
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

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_dx_full_load_dgamma_full_load)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};

    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                         "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64, "socVersion": "Ascend950"}
                })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::RmsNormGradQuantCompileInfo compile_info;

    std::string op_type("RmsNormGradQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(6, 2)
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
                      .NodeIoNum(6, 2)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape, &offset_x_shape})
                      .OutputShapes({&dx_out_shape, &dgamma_out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"quantMode", Ops::NN::AnyValue::CreateFrom<std::string>("static")},
                                  {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                  {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)}})
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
    EXPECT_EQ(tiling_key, 0);
    auto tilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(tilingData, nullptr);
}

static ge::graphStatus RunRmsNormGradQuantTiling(const std::vector<gert::StorageShape*>& inputShapes,
                                                 const std::vector<gert::StorageShape*>& outputShapes,
                                                 const std::vector<ge::DataType>& inputDtypes,
                                                 const std::vector<ge::DataType>& outputDtypes,
                                                 const std::vector<uint32_t>& irInstanceNum, bool divMode,
                                                 uint64_t& outTilingKey, int64_t dstType = 34)
{
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                         "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64, "socVersion": "Ascend950"}
                })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::RmsNormGradQuantCompileInfo compile_info;

    std::string op_type("RmsNormGradQuant");
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (op_impl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = op_impl->tiling;
    auto tiling_parse_func = op_impl->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(6, 2)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    if (!kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return ge::GRAPH_FAILED;
    }
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    auto parse_ret = tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());
    if (parse_ret != ge::GRAPH_SUCCESS) {
        return parse_ret;
    }

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::TilingContextFaker faker;
    faker.SetOpType(op_type)
        .NodeIoNum(6, 2)
        .IrInstanceNum(irInstanceNum)
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .TilingData(param.get())
        .Workspace(ws_size);

    for (size_t i = 0; i < inputDtypes.size(); ++i) {
        faker.NodeInputTd(static_cast<int32_t>(i), inputDtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < outputDtypes.size(); ++i) {
        faker.NodeOutputTd(static_cast<int32_t>(i), outputDtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }

    faker.NodeAttrs({{"quantMode", Ops::NN::AnyValue::CreateFrom<std::string>("static")},
                     {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(divMode)},
                     {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dstType)}});

    auto holder = faker.Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tiling_func(tiling_context);
    if (ret == ge::GRAPH_SUCCESS) {
        outTilingKey = tiling_context->GetTilingKey();
    }
    return ret;
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_empty_rows0_fp16)
{
    gert::StorageShape dy_shape = {{0, 64}, {0, 64}};
    gert::StorageShape x_shape = {{0, 64}, {0, 64}};
    gert::StorageShape rstd_shape = {{0}, {0}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{0, 64}, {0, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_big_m_fp16)
{
    gert::StorageShape dy_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape x_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape rstd_shape = {{4096}, {4096}};
    gert::StorageShape gamma_shape = {{1024}, {1024}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape dgamma_out_shape = {{1024}, {1024}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_fp16_no_optional)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_fp16_scalex_zp)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_bf16_gamma_float)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_float_all)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_dx_split)
{
    gert::StorageShape dy_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape x_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{8192}, {8192}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape dgamma_out_shape = {{8192}, {8192}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_dgamma_large)
{
    gert::StorageShape dy_shape = {{1024, 64}, {1024, 64}};
    gert::StorageShape x_shape = {{1024, 64}, {1024, 64}};
    gert::StorageShape rstd_shape = {{1024}, {1024}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{1024, 64}, {1024, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_div_mode_false)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, false, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_scalex_dtype_float)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_bf16_scalex_no_zp)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_dx_split_dgamma_large)
{
    // cols=8192 > DX_UB_FACTOR(6144) => dx SPLIT_D
    // rows=1024, cols=8192 => dgamma WITH_LARGE_ROWS
    gert::StorageShape dy_shape = {{1024, 8192}, {1024, 8192}};
    gert::StorageShape x_shape = {{1024, 8192}, {1024, 8192}};
    gert::StorageShape rstd_shape = {{1024}, {1024}};
    gert::StorageShape gamma_shape = {{8192}, {8192}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{1024, 8192}, {1024, 8192}};
    gert::StorageShape dgamma_out_shape = {{8192}, {8192}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_big_m_with_scalex_zp)
{
    gert::StorageShape dy_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape x_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape rstd_shape = {{4096}, {4096}};
    gert::StorageShape gamma_shape = {{1024}, {1024}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape dgamma_out_shape = {{1024}, {1024}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_big_m_div_mode_false)
{
    gert::StorageShape dy_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape x_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape rstd_shape = {{4096}, {4096}};
    gert::StorageShape gamma_shape = {{1024}, {1024}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape dgamma_out_shape = {{1024}, {1024}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, false, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_big_m_bf16)
{
    gert::StorageShape dy_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape x_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape rstd_shape = {{4096}, {4096}};
    gert::StorageShape gamma_shape = {{1024}, {1024}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{4096, 1024}, {4096, 1024}};
    gert::StorageShape dgamma_out_shape = {{1024}, {1024}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_empty_rows0_with_scalex_zp)
{
    gert::StorageShape dy_shape = {{0, 64}, {0, 64}};
    gert::StorageShape x_shape = {{0, 64}, {0, 64}};
    gert::StorageShape rstd_shape = {{0}, {0}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{0, 64}, {0, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_empty_rows0_bf16)
{
    gert::StorageShape dy_shape = {{0, 64}, {0, 64}};
    gert::StorageShape x_shape = {{0, 64}, {0, 64}};
    gert::StorageShape rstd_shape = {{0}, {0}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{0, 64}, {0, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_fp16_scalex_zp_int8)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_INT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey, 2),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_bf16_scalex_zp_int8)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_BF16,  ge::DT_BF16,  ge::DT_FLOAT,
                                             ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_INT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey, 2),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_fp16_scalex_no_zp_int8)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_INT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey, 2),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_dx_split_int8)
{
    gert::StorageShape dy_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape x_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{8192}, {8192}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape dgamma_out_shape = {{8192}, {8192}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_INT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey, 2),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_div_mode_false_int8)
{
    gert::StorageShape dy_shape = {{8, 64}, {8, 64}};
    gert::StorageShape x_shape = {{8, 64}, {8, 64}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 64}, {8, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_INT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, false, tilingKey, 2),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_empty_rows0_int8)
{
    gert::StorageShape dy_shape = {{0, 64}, {0, 64}};
    gert::StorageShape x_shape = {{0, 64}, {0, 64}};
    gert::StorageShape rstd_shape = {{0}, {0}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{0, 64}, {0, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_INT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey, 2),
              ge::GRAPH_SUCCESS);
}

// Coverage: empty_tiling.cpp CalcUsedCoreNumGamma multi-core path (cols > 8192)
// and CalcTilingDataDgamma UB split path
TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_empty_large_cols)
{
    gert::StorageShape dy_shape = {{0, 16384}, {0, 16384}};
    gert::StorageShape x_shape = {{0, 16384}, {0, 16384}};
    gert::StorageShape rstd_shape = {{0}, {0}};
    gert::StorageShape gamma_shape = {{16384}, {16384}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{0, 16384}, {0, 16384}};
    gert::StorageShape dgamma_out_shape = {{16384}, {16384}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

// Coverage: big_m_tiling.cpp DgammaDoTilingStg0 tail block fold path (basicBlockLoopTailBlock != 0)
// and FindNearestPower2 value > 4 branch
// rows=16384, cols=64 => big M path, mToProcessTailBlock large enough for fold
TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_big_m_large_rows_tail_fold)
{
    gert::StorageShape dy_shape = {{16384, 64}, {16384, 64}};
    gert::StorageShape x_shape = {{16384, 64}, {16384, 64}};
    gert::StorageShape rstd_shape = {{16384}, {16384}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{16384, 64}, {16384, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

// Coverage: regbase_tiling.cpp CalcUsedCoreNumGamma multi-colset path
// cols=1024 => cols_sets = 1024/8 = 128 > 64 cores => isMultiColset=true
// Also covers colsPerCoreDG_ adjustment branch (line 437)
TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_multi_colset)
{
    gert::StorageShape dy_shape = {{8, 1024}, {8, 1024}};
    gert::StorageShape x_shape = {{8, 1024}, {8, 1024}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{1024}, {1024}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{8, 1024}, {8, 1024}};
    gert::StorageShape dgamma_out_shape = {{1024}, {1024}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

// Coverage: regbase_tiling.cpp CalcTilingDataDgamma WITH_LARGE_ROWS path
// rows=4096, cols=64 => UB can't full-load all rows => WITH_LARGE_ROWS
// But not big enough for BigM (rows < cols*2 or < MFACTOR*CORE/2 is not met, needs check)
TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_dgamma_large_rows_split)
{
    // rows=256, cols=64 => not big_m (256 < 64*2=128 fails, 256 >= 128 true, but 256 < 64*64/2=2048)
    // Actually BigM: rows >= cols*2 AND rows >= MFACTOR*aivCoreNum/2 = 64*64/2=2048 => 256 < 2048 => no BigM
    // UB check: inputUbSize = 2*(ceil(256*4)*8) + ceil(256*4) = 2*8192 + 1024 = 17408
    //           outputUbSize = ceil(257*4)*8 = 8224
    //           total = 2*(17408+8224) = 51264, UB=245760 => fits => FULL_LOAD
    // Need larger rows: rows=8192, cols=64
    // BigM check: 8192 >= 128 and 8192 >= 2048 => BigM kicks in (priority 500 > 1000)
    // To avoid BigM: rows must be < cols*2 or < MFACTOR*CORE/2
    // Use rows=2000, cols=64: 2000 < 2048 => no BigM
    // UB: inputUbSizeForOneColSet = ceil(2000*4)*8 = 8000*8=64000, dy same = 64000, rstd=8000
    //     total input = 64000+64000+8000 = 136000
    //     output = ceil(2001*4)*8 = 8004*8=64032
    //     2*(136000+64032) = 400064, UB=245760 => 400064 > 245760 => WITH_LARGE_ROWS!
    gert::StorageShape dy_shape = {{2000, 64}, {2000, 64}};
    gert::StorageShape x_shape = {{2000, 64}, {2000, 64}};
    gert::StorageShape rstd_shape = {{2000}, {2000}};
    gert::StorageShape gamma_shape = {{64}, {64}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{2000, 64}, {2000, 64}};
    gert::StorageShape dgamma_out_shape = {{64}, {64}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_regbase_dx_full_load_2)
{
    // rows=256, cols=64 => not big_m (256 < 64*2=128 fails, 256 >= 128 true, but 256 < 64*64/2=2048)
    // Actually BigM: rows >= cols*2 AND rows >= MFACTOR*aivCoreNum/2 = 64*64/2=2048 => 256 < 2048 => no BigM
    // UB check: inputUbSize = 2*(ceil(256*4)*8) + ceil(256*4) = 2*8192 + 1024 = 17408
    //           outputUbSize = ceil(257*4)*8 = 8224
    //           total = 2*(17408+8224) = 51264, UB=245760 => fits => FULL_LOAD
    // Need larger rows: rows=8192, cols=64
    // BigM check: 8192 >= 128 and 8192 >= 2048 => BigM kicks in (priority 500 > 1000)
    // To avoid BigM: rows must be < cols*2 or < MFACTOR*CORE/2
    // Use rows=2000, cols=64: 2000 < 2048 => no BigM
    // UB: inputUbSizeForOneColSet = ceil(2000*4)*8 = 8000*8=64000, dy same = 64000, rstd=8000
    //     total input = 64000+64000+8000 = 136000
    //     output = ceil(2001*4)*8 = 8004*8=64032
    //     2*(136000+64032) = 400064, UB=245760 => 400064 > 245760 => WITH_LARGE_ROWS!
    gert::StorageShape dy_shape = {{2, 16239, 9, 2, 2, 2}, {2, 16239, 9, 2, 2, 2}};
    gert::StorageShape x_shape = {{2, 16239, 9, 2, 2, 2}, {2, 16239, 9, 2, 2, 2}};
    gert::StorageShape rstd_shape = {{2, 16239, 9, 2}, {2, 16239, 9, 2}};
    gert::StorageShape gamma_shape = {{2, 2}, {2, 2}};
    gert::StorageShape scales_x_shape = {{1}, {1}};
    gert::StorageShape offset_x_shape = {{1}, {1}};
    gert::StorageShape dx_out_shape = {{2, 16239, 9, 2, 2, 2}, {2, 16239, 9, 2, 2, 2}};
    gert::StorageShape dgamma_out_shape = {{2, 2}, {2, 2}};

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_SUCCESS);
}

// Shape validation tests for CheckInputsShape
static void BuildValidShapes(gert::StorageShape& dy_shape, gert::StorageShape& x_shape, gert::StorageShape& rstd_shape,
                             gert::StorageShape& gamma_shape, gert::StorageShape& scales_x_shape,
                             gert::StorageShape& dx_out_shape, gert::StorageShape& dgamma_out_shape)
{
    dy_shape = {{8, 64}, {8, 64}};
    x_shape = {{8, 64}, {8, 64}};
    rstd_shape = {{8}, {8}};
    gamma_shape = {{64}, {64}};
    scales_x_shape = {{1}, {1}};
    dx_out_shape = {{8, 64}, {8, 64}};
    dgamma_out_shape = {{64}, {64}};
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_dy_dim_invalid)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    dy_shape = {{64}, {64}}; // dimNum=1, invalid

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_x_dim_invalid)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    x_shape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}; // dimNum=9, invalid

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_rstd_dim_invalid)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    rstd_shape = {{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}}; // dimNum=8, invalid

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_gamma_dim_invalid)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    gamma_shape = {{}, {}}; // dimNum=0, invalid

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_scalesx_dim_invalid)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    scales_x_shape = {{1, 1}, {1, 1}}; // dimNum=2, invalid

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_offsetx_dim_invalid)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    gert::StorageShape offset_x_shape = {{1, 1}, {1, 1}}; // dimNum=2, invalid

    std::vector<gert::StorageShape*> inputs = {&dy_shape,    &x_shape,        &rstd_shape,
                                               &gamma_shape, &scales_x_shape, &offset_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                             ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 1};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_dimnum_relation_invalid)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    rstd_shape = {{8, 8}, {8, 8}}; // x.dimNum(2) != rstd.dimNum(2) + gamma.dimNum(1)

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_rstd_prefix_mismatch)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    x_shape = {{4, 64}, {4, 64}}; // rstd[0]=8 != x[0]=4

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_gamma_suffix_mismatch)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    gamma_shape = {{32}, {32}}; // gamma[0]=32 != x[1]=64

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_dxout_mismatch)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    dx_out_shape = {{8, 32}, {8, 32}}; // dxOut != dy

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradQuantTilingTest, rms_norm_grad_quant_shape_dgammaout_mismatch)
{
    gert::StorageShape dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape;
    BuildValidShapes(dy_shape, x_shape, rstd_shape, gamma_shape, scales_x_shape, dx_out_shape, dgamma_out_shape);
    dgamma_out_shape = {{32}, {32}}; // dgammaOut != gamma

    std::vector<gert::StorageShape*> inputs = {&dy_shape, &x_shape, &rstd_shape, &gamma_shape, &scales_x_shape};
    std::vector<gert::StorageShape*> outputs = {&dx_out_shape, &dgamma_out_shape};
    std::vector<ge::DataType> inputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                                             ge::DT_FLOAT};
    std::vector<ge::DataType> outputDtypes = {ge::DT_HIFLOAT8, ge::DT_FLOAT};
    std::vector<uint32_t> irInstanceNum = {1, 1, 1, 1, 1, 0};

    uint64_t tilingKey = 0;
    EXPECT_EQ(RunRmsNormGradQuantTiling(inputs, outputs, inputDtypes, outputDtypes, irInstanceNum, true, tilingKey),
              ge::GRAPH_FAILED);
}