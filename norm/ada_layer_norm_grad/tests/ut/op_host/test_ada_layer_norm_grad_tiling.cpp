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
#include "array_ops.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "norm/ada_layer_norm_grad/op_host/ada_layer_norm_grad_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

struct AdaLayerNormGradCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
    int64_t blockSize = 0;
    int64_t vlFp32 = 0;
    bool isRegBase = false;
};

class AdaLayerNormGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaLayerNormGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaLayerNormGradTiling TearDown" << std::endl; }
};

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_000)
{
    gert::StorageShape input0_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape input1_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape input2_shape = {{4, 16}, {4, 16}};
    gert::StorageShape input3_shape = {{4, 16}, {4, 16}};
    gert::StorageShape input4_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape input5_shape = {{4096}, {4096}};
    gert::StorageShape input6_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape output1_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape output2_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape output3_shape = {{4096}, {4096}};
    gert::StorageShape output4_shape = {{4096}, {4096}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaLayerNormGradCompileInfo compile_info;

    std::string op_type("AdaLayerNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaLayerNormGrad")
                      .NodeIoNum(7, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape,
                                    &input5_shape, &input6_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape, &output3_shape, &output4_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 411);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
}

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_001)
{
    gert::StorageShape input0_shape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape input1_shape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape input2_shape = {{4, 16}, {4, 16}};
    gert::StorageShape input3_shape = {{4, 16}, {4, 16}};
    gert::StorageShape input4_shape = {{4, 128}, {4, 128}};
    gert::StorageShape input5_shape = {{128}, {128}};
    gert::StorageShape input6_shape = {{128}, {128}};
    gert::StorageShape output0_shape = {{4, 16, 128}, {4, 16, 128}};
    gert::StorageShape output1_shape = {{4, 128}, {4, 128}};
    gert::StorageShape output2_shape = {{4, 128}, {4, 128}};
    gert::StorageShape output3_shape = {{128}, {128}};
    gert::StorageShape output4_shape = {{128}, {128}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaLayerNormGradCompileInfo compile_info;

    std::string op_type("AdaLayerNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaLayerNormGrad")
                      .NodeIoNum(7, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape,
                                    &input5_shape, &input6_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape, &output3_shape, &output4_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 211);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
}

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_002)
{
    gert::StorageShape input0_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape input1_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape input2_shape = {{4, 16}, {4, 16}};
    gert::StorageShape input3_shape = {{4, 16}, {4, 16}};
    gert::StorageShape input4_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape input5_shape = {{4096}, {4096}};
    gert::StorageShape input6_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape output1_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape output2_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape output3_shape = {{4096}, {4096}};
    gert::StorageShape output4_shape = {{4096}, {4096}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaLayerNormGradCompileInfo compile_info;

    std::string op_type("AdaLayerNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaLayerNormGrad")
                      .NodeIoNum(7, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape,
                                    &input5_shape, &input6_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape, &output3_shape, &output4_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 412);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
}

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_003)
{
    gert::StorageShape input0_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape input1_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape input2_shape = {{4, 16}, {4, 16}};
    gert::StorageShape input3_shape = {{4, 16}, {4, 16}};
    gert::StorageShape input4_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape input5_shape = {{4096}, {4096}};
    gert::StorageShape input6_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{4, 16, 4096}, {4, 16, 4096}};
    gert::StorageShape output1_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape output2_shape = {{4, 4096}, {4, 4096}};
    gert::StorageShape output3_shape = {{4096}, {4096}};
    gert::StorageShape output4_shape = {{4096}, {4096}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 48}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaLayerNormGradCompileInfo compile_info;

    std::string op_type("AdaLayerNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaLayerNormGrad")
                      .NodeIoNum(7, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape,
                                    &input5_shape, &input6_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape, &output3_shape, &output4_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 414);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 48);
}

static constexpr uint64_t ASCEND_950_UB_SIZE = 245760;
static constexpr uint64_t ASCEND_950_CORE_NUM = 64;

static string Build950CompileInfoString(uint64_t ubSize = ASCEND_950_UB_SIZE)
{
    return R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": )" +
           std::to_string(ubSize) +
           R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": )" +
           std::to_string(ASCEND_950_CORE_NUM) + R"(, "socVersion": "Ascend950"}
                          })";
}

static void SetAdaLayerNormGradPlatformResource(const string& compileInfoString, fe::PlatFormInfos& platformInfo)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> version;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics, version);

    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    platformInfo.SetPlatformRes("version", version);
}

// 950 (regbase / arch35) tiling runner: isRegBase=true selects AdaLayerNormGradArch35Tiling.
// Returns the tiling status, tiling key, block dim and workspace size of the selected template.
static void RunAdaLayerNormGrad950Tiling(gert::StorageShape& input0_shape, gert::StorageShape& input1_shape,
                                         gert::StorageShape& input2_shape, gert::StorageShape& input3_shape,
                                         gert::StorageShape& input4_shape, gert::StorageShape& input5_shape,
                                         gert::StorageShape& input6_shape, gert::StorageShape& output0_shape,
                                         gert::StorageShape& output1_shape, gert::StorageShape& output2_shape,
                                         gert::StorageShape& output3_shape, gert::StorageShape& output4_shape,
                                         ge::DataType dataType, ge::DataType gammaType, uint32_t tilingKeyExpect,
                                         uint32_t blockDimExpect, uint64_t workspaceExpect, uint64_t ubSize,
                                         ge::graphStatus expectStatus = ge::GRAPH_SUCCESS)
{
    auto compile_info_string = std::make_shared<string>(Build950CompileInfoString(ubSize));
    std::vector<char> compile_info_buf(compile_info_string->begin(), compile_info_string->end());
    compile_info_buf.push_back('\0');

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    AdaLayerNormGradCompileInfo compile_info;

    std::string op_type("AdaLayerNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({compile_info_buf.data(), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    SetAdaLayerNormGradPlatformResource(*compile_info_string,
                                        *kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo());

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaLayerNormGrad")
                      .NodeIoNum(7, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape,
                                    &input5_shape, &input6_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape, &output3_shape, &output4_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, gammaType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, gammaType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, gammaType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, gammaType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    SetAdaLayerNormGradPlatformResource(*compile_info_string, *tiling_context->GetPlatformInfo());

    ASSERT_EQ(tiling_func(tiling_context), expectStatus);
    if (expectStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, tilingKeyExpect);
    ASSERT_EQ(tiling_context->GetBlockDim(), blockDimExpect);
    ASSERT_EQ(ws_size->GetSize(), 1U);
    const size_t* wsData = reinterpret_cast<const size_t*>(ws_size->GetData());
    // workspace = sys workspace (from lib api) + usrWorkspaceSize (blockNum*colAlignV*workspaceNum*4)
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tiling_context->GetPlatformInfo());
    ASSERT_EQ(wsData[0], static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()) + workspaceExpect);
}

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_950_cutn_batch_split)
{
    gert::StorageShape input0_shape = {{64, 1, 133}, {64, 1, 133}};
    gert::StorageShape input1_shape = {{64, 1, 133}, {64, 1, 133}};
    gert::StorageShape input2_shape = {{64, 1, 1}, {64, 1, 1}};
    gert::StorageShape input3_shape = {{64, 1, 1}, {64, 1, 1}};
    gert::StorageShape input4_shape = {{64, 133}, {64, 133}};
    gert::StorageShape input5_shape = {{133}, {133}};
    gert::StorageShape input6_shape = {{133}, {133}};
    gert::StorageShape output0_shape = {{64, 1, 133}, {64, 1, 133}};
    gert::StorageShape output1_shape = {{64, 133}, {64, 133}};
    gert::StorageShape output2_shape = {{64, 133}, {64, 133}};
    gert::StorageShape output3_shape = {{133}, {133}};
    gert::StorageShape output4_shape = {{133}, {133}};

    // batch-split (batch 64 >= coreNum 64) + cutN => tier1, fp32 => key 110, workspace = blockNum*colAlignV*2*4
    RunAdaLayerNormGrad950Tiling(input0_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape,
                                 input6_shape, output0_shape, output1_shape, output2_shape, output3_shape,
                                 output4_shape, ge::DT_FLOAT, ge::DT_FLOAT, 110, 64, 64 * 136 * 2 * 4,
                                 ASCEND_950_UB_SIZE);
}

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_950_cutn_row_split)
{
    gert::StorageShape input0_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input1_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape input2_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input3_shape = {{40, 1, 1}, {40, 1, 1}};
    gert::StorageShape input4_shape = {{40, 133}, {40, 133}};
    gert::StorageShape input5_shape = {{133}, {133}};
    gert::StorageShape input6_shape = {{133}, {133}};
    gert::StorageShape output0_shape = {{40, 1, 133}, {40, 1, 133}};
    gert::StorageShape output1_shape = {{40, 133}, {40, 133}};
    gert::StorageShape output2_shape = {{40, 133}, {40, 133}};
    gert::StorageShape output3_shape = {{133}, {133}};
    gert::StorageShape output4_shape = {{133}, {133}};

    // batch 40 < coreNum 64, row 40 < coreNum 64 => blockNum=row=40, cutN row-split => tier2 => key 210,
    // workspace = blockNum*colAlignV*(2+2*batch)*4
    RunAdaLayerNormGrad950Tiling(input0_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape,
                                 input6_shape, output0_shape, output1_shape, output2_shape, output3_shape,
                                 output4_shape, ge::DT_FLOAT, ge::DT_FLOAT, 210, 40, 40 * 136 * (2 + 2 * 40) * 4,
                                 ASCEND_950_UB_SIZE);
}

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_950_cutd_batch_split)
{
    gert::StorageShape input0_shape = {{64, 1, 4096}, {64, 1, 4096}};
    gert::StorageShape input1_shape = {{64, 1, 4096}, {64, 1, 4096}};
    gert::StorageShape input2_shape = {{64, 1, 1}, {64, 1, 1}};
    gert::StorageShape input3_shape = {{64, 1, 1}, {64, 1, 1}};
    gert::StorageShape input4_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape input5_shape = {{4096}, {4096}};
    gert::StorageShape input6_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{64, 1, 4096}, {64, 1, 4096}};
    gert::StorageShape output1_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape output2_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape output3_shape = {{4096}, {4096}};
    gert::StorageShape output4_shape = {{4096}, {4096}};

    // large col 4096 forces cutD (cutN cannot host a row), batch-split => tier3 => key 310,
    // workspace = blockNum*colAlignV*2*4
    RunAdaLayerNormGrad950Tiling(input0_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape,
                                 input6_shape, output0_shape, output1_shape, output2_shape, output3_shape,
                                 output4_shape, ge::DT_FLOAT, ge::DT_FLOAT, 310, 64, 64 * 4096 * 2 * 4,
                                 ASCEND_950_UB_SIZE);
}

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_950_cutd_row_split)
{
    gert::StorageShape input0_shape = {{1, 1, 4096}, {1, 1, 4096}};
    gert::StorageShape input1_shape = {{1, 1, 4096}, {1, 1, 4096}};
    gert::StorageShape input2_shape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape input3_shape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape input4_shape = {{1, 4096}, {1, 4096}};
    gert::StorageShape input5_shape = {{4096}, {4096}};
    gert::StorageShape input6_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{1, 1, 4096}, {1, 1, 4096}};
    gert::StorageShape output1_shape = {{1, 4096}, {1, 4096}};
    gert::StorageShape output2_shape = {{1, 4096}, {1, 4096}};
    gert::StorageShape output3_shape = {{4096}, {4096}};
    gert::StorageShape output4_shape = {{4096}, {4096}};

    // batch 1, row 1 < coreNum 64 => blockNum=1, large col 4096 cutD => tier4 => key 410,
    // workspace = blockNum*colAlignV*(2+2*1)*4
    RunAdaLayerNormGrad950Tiling(input0_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape,
                                 input6_shape, output0_shape, output1_shape, output2_shape, output3_shape,
                                 output4_shape, ge::DT_FLOAT, ge::DT_FLOAT, 410, 1, 1 * 4096 * (2 + 2 * 1) * 4,
                                 ASCEND_950_UB_SIZE);
}

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_950_cutd_dinner_length_reject)
{
    gert::StorageShape input0_shape = {{64, 1, 4096}, {64, 1, 4096}};
    gert::StorageShape input1_shape = {{64, 1, 4096}, {64, 1, 4096}};
    gert::StorageShape input2_shape = {{64, 1, 1}, {64, 1, 1}};
    gert::StorageShape input3_shape = {{64, 1, 1}, {64, 1, 1}};
    gert::StorageShape input4_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape input5_shape = {{4096}, {4096}};
    gert::StorageShape input6_shape = {{4096}, {4096}};
    gert::StorageShape output0_shape = {{64, 1, 4096}, {64, 1, 4096}};
    gert::StorageShape output1_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape output2_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape output3_shape = {{4096}, {4096}};
    gert::StorageShape output4_shape = {{4096}, {4096}};

    // UB too small for even a single cutD chunk => dInnerLength < B32_ALIGN_FACTOR => reject
    RunAdaLayerNormGrad950Tiling(input0_shape, input1_shape, input2_shape, input3_shape, input4_shape, input5_shape,
                                 input6_shape, output0_shape, output1_shape, output2_shape, output3_shape,
                                 output4_shape, ge::DT_FLOAT, ge::DT_FLOAT, 0, 0, 0, 1024, ge::GRAPH_FAILED);
}

// Direct construction of the arch35 tiling class so the defensive rejection branches
// (row<1 / col<1 / blockNum<1) can be exercised without the base shape-attribute gate.
class TestAdaLayerNormGradArch35Tiling : public optiling::AdaLayerNormGradArch35Tiling {
public:
    using optiling::AdaLayerNormGradArch35Tiling::AdaLayerNormGradArch35Tiling;
    ge::graphStatus RunDoOpTiling() { return DoOpTiling(); }
    optiling::ParamsAdaLayerNormGrad& Params() { return commonParams; }
};

TEST_F(AdaLayerNormGradTiling, ada_layer_norm_grad_tiling_950_arch35_reject_row_col_blocknum)
{
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    auto compile_info_string = std::make_shared<string>(Build950CompileInfoString());
    SetAdaLayerNormGradPlatformResource(*compile_info_string, platform_info);
    AdaLayerNormGradCompileInfo compile_info;

    // Valid arch35 input contract so DoOpTiling's scale / rstd / mean shape checks pass and the
    // row<1 / col<1 defensive branches are actually reached (the faker still needs real input tensors
    // or GetInputShape(index) dereferences a null input pointer).
    gert::StorageShape input0_shape = {{1, 1, 133}, {1, 1, 133}};
    gert::StorageShape input1_shape = {{1, 1, 133}, {1, 1, 133}};
    gert::StorageShape input2_shape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape input3_shape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape input4_shape = {{1, 133}, {1, 133}};
    gert::StorageShape input5_shape = {{133}, {133}};
    gert::StorageShape input6_shape = {{133}, {133}};
    gert::StorageShape output0_shape = {{1, 1, 133}, {1, 1, 133}};
    gert::StorageShape output1_shape = {{1, 133}, {1, 133}};
    gert::StorageShape output2_shape = {{1, 133}, {1, 133}};
    gert::StorageShape output3_shape = {{133}, {133}};
    gert::StorageShape output4_shape = {{133}, {133}};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("AdaLayerNormGrad")
                      .NodeIoNum(7, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&input0_shape, &input1_shape, &input2_shape, &input3_shape, &input4_shape,
                                    &input5_shape, &input6_shape})
                      .OutputShapes({&output0_shape, &output1_shape, &output2_shape, &output3_shape, &output4_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    SetAdaLayerNormGradPlatformResource(*compile_info_string, *tiling_context->GetPlatformInfo());

    TestAdaLayerNormGradArch35Tiling tiling(tiling_context);
    auto& params = tiling.Params();
    params.isRegBase = true;
    params.dyDtype = ge::DataType::DT_FLOAT;
    params.coreNum = ASCEND_950_CORE_NUM;
    params.ubSizePlatForm = ASCEND_950_UB_SIZE;
    params.batchSize = 1;
    params.seqSize = 1;

    // row < 1 rejection
    params.rowSize = 0;
    params.colSize = 133;
    EXPECT_EQ(tiling.RunDoOpTiling(), ge::GRAPH_FAILED);

    // col < 1 rejection
    params.rowSize = 1;
    params.colSize = 0;
    EXPECT_EQ(tiling.RunDoOpTiling(), ge::GRAPH_FAILED);

    // blockNum < 1 guard: with row>=1/col>=1 the core-split always yields blockNum>=1 (batch-split or
    // row-split both keep blockNum>=CeilDiv(1,..)), so the only shape that can drive blockNum to 0 is
    // row=0, which is already rejected by the row<1 guard above; exercising the same guard path here.
    params.rowSize = 0;
    params.colSize = 133;
    EXPECT_EQ(tiling.RunDoOpTiling(), ge::GRAPH_FAILED);
}
