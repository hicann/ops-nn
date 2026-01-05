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
 * \file test_init_embedding_hash_table.cpp
 * \brief test_init_embedding_hash_table
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
#include "../../../../op_host/arch35/init_embedding_hash_table_tiling_arch35.h"

using namespace std;

class InitEmbeddingHashTableTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InitEmbeddingHashTableTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InitEmbeddingHashTableTiling TearDown" << std::endl; }
};

TEST_F(InitEmbeddingHashTableTiling, InitEmbeddingHashTable_tiling_0)
{
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 196608,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 1
                                    }
                                })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::InitEmbeddingHashTableCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.maxThread = 512;

    std::string op_type("InitEmbeddingHashTable");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    gert::StorageShape table_handle = {{5}, {5}};
    gert::StorageShape sampled_values = {{16}, {16}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&table_handle, &sampled_values})
                      .OutputShapes({&table_handle})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"bucket_size", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                  {"embedding_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                  {"initializer_mode", Ops::NN::AnyValue::CreateFrom<string>("random")},
                                  {"constant_value", Ops::NN::AnyValue::CreateFrom<float>(0.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext *tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
    // auto raw_tiling_data = tiling_context->GetRawTilingData();
    // auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    // ASSERT_EQ(tiling_data_result, "4 4 0 0 5 512 ");
}
