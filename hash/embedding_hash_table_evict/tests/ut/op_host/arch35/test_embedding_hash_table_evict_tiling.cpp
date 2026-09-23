/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_embedding_hash_table_evict_tiling.cpp
 * \brief embedding_hash_table_evict tiling test
 */

#include <iostream>
#include <map>
#include <string>
#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/embedding_hash_table_evict_tiling_arch35.h"

using namespace ge;
using namespace ut_util;

namespace {
constexpr size_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;

std::string TilingData2Str(const gert::TilingData* tilingData)
{
    std::string result;
    for (size_t i = 0; i < tilingData->GetDataSize(); i += sizeof(int32_t)) {
        result += std::to_string((reinterpret_cast<const int32_t*>(tilingData->GetData())[i / sizeof(int32_t)]));
        result += " ";
    }
    return result;
}

class EmbeddingHashTableEvictTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EmbeddingHashTableEvict Tiling Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "EmbeddingHashTableEvict Tiling Test TearDown" << std::endl; }
};

TEST_F(EmbeddingHashTableEvictTilingTest, tiling_succeed)
{
    std::string compileInfoString = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::EvictCompileInfo compileInfo;

    std::string opType("EmbeddingHashTableEvict");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    auto* parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    ASSERT_TRUE(parseContext->GetPlatformInfo()->Init());
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* workspaceSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    ASSERT_NE(param, nullptr);

    gert::StorageShape tableHandleShape = {{5}, {5}};
    gert::StorageShape keysShape = {{128}, {128}};
    gert::StorageShape dummyOutputShape = {{1}, {1}};

    auto holder = gert::TilingContextFaker()
                      .SetOpType("EmbeddingHashTableEvict")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 0})
                      .InputShapes({&tableHandleShape, &keysShape})
                      .OutputShapes({&dummyOutputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"table_cap", Ops::NN::AnyValue::CreateFrom<int64_t>(100)},
                          {"embedding_dim", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                          {"init_mode", Ops::NN::AnyValue::CreateFrom<std::string>("constant")},
                          {"const_val", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                      })
                      .TilingData(param.get())
                      .Workspace(workspaceSize)
                      .Build();

    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), 0);
    ASSERT_EQ(tilingContext->GetBlockDim(), 1);
    auto actualWorkspaceSizes = tilingContext->GetWorkspaceSizes(1);
    ASSERT_NE(actualWorkspaceSizes, nullptr);
    ASSERT_EQ(actualWorkspaceSizes[0], ASCENDC_TOOLS_WORKSPACE);
    ASSERT_EQ(TilingData2Str(tilingContext->GetRawTilingData()), "100 0 8 0 0 0 0 0 128 0 512 0 ");
}
} // namespace
