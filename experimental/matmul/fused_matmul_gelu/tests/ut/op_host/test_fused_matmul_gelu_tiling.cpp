/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "../../../op_host/fused_matmul_gelu_tiling.h"

using namespace std;
using namespace ge;

namespace {
constexpr uint64_t SYS_WORKSPACE_BYTES = static_cast<uint64_t>(16 * 1024 * 1024);
constexpr uint64_t EXPECT_BUF_SIZE = 188416;
constexpr uint64_t EXPECT_ELEMS_PER_VEC_LOOP = 5888;
constexpr uint64_t EXPECT_MATMUL_WORKSPACE_SIZE = 512;
constexpr uint64_t EXPECT_CUBE_CORE_ALIGNED = 8;

struct FusedMatmulGeluTilingTestParam {
    string caseName;
    ge::DataType dtype;
    initializer_list<int64_t> xShape;
    initializer_list<int64_t> weightShape;
    initializer_list<int64_t> biasShape;
    initializer_list<int64_t> yShape;
    bool hasBias;
    int64_t approximate;
    ge::graphStatus expectStatus;
    uint64_t expectTilingKey;
    uint64_t expectM;
    uint64_t expectK;
    uint64_t expectN;
    uint64_t expectTotalElement;
    uint64_t expectHasBias;
    uint64_t expectApproximate;
};

static string CompileInfoString()
{
    return R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                 "Intrinsic_fix_pipe_l0c2out": false,
                                 "Intrinsic_data_move_l12ub": true,
                                 "Intrinsic_data_move_l0c2ub": true,
                                 "Intrinsic_data_move_out2l1_nd2nz": false,
                                 "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                 "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                 "CORE_NUM": 40, "vector_core_cnt": 40,
                                 "socVersion": "Ascend910B"}})";
}

static vector<uint64_t> ReadFirstU64Fields(const gert::TilingData* tilingData, size_t fieldNum)
{
    vector<uint64_t> fields;
    if (tilingData == nullptr || tilingData->GetData() == nullptr) {
        return fields;
    }

    const auto* data = reinterpret_cast<const uint64_t*>(tilingData->GetData());
    for (size_t i = 0; i < fieldNum; ++i) {
        fields.push_back(data[i]);
    }
    return fields;
}
} // namespace

class FusedMatmulGeluTilingTest : public testing::TestWithParam<FusedMatmulGeluTilingTestParam> {};

TEST_P(FusedMatmulGeluTilingTest, tiling_cases)
{
    auto param = GetParam();

    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape weightShape = {param.weightShape, param.weightShape};
    gert::StorageShape biasShape = {param.biasShape, param.biasShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};

    string opType("FusedMatmulGelu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoString = CompileInfoString();

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    optiling::FusedMatmulGeluCompileInfo compileInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType.c_str())
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &weightShape, param.hasBias ? &biasShape : nullptr})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"approximate", Ops::NN::AnyValue::CreateFrom<int64_t>(param.approximate)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);

    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), param.expectStatus);
    if (param.expectStatus != ge::GRAPH_SUCCESS) {
        return;
    }

    EXPECT_EQ(tilingContext->GetTilingKey(), param.expectTilingKey);
    EXPECT_EQ(tilingContext->GetBlockDim(), 1U);

    auto fields = ReadFirstU64Fields(tilingContext->GetRawTilingData(), 15);
    ASSERT_EQ(fields.size(), 15UL);

    EXPECT_EQ(fields[0], param.expectM);
    EXPECT_EQ(fields[1], param.expectK);
    EXPECT_EQ(fields[2], param.expectN);
    EXPECT_EQ(fields[3], param.expectTotalElement);
    EXPECT_EQ(fields[4], EXPECT_BUF_SIZE);
    EXPECT_EQ(fields[5], 1UL); // cubeCoreNum
    EXPECT_EQ(fields[6], 2UL); // vecCoreNum
    EXPECT_EQ(fields[7], 0UL); // vecTasksPerCore
    EXPECT_EQ(fields[8], 1UL); // vecTasksTailCore
    EXPECT_EQ(fields[9], EXPECT_ELEMS_PER_VEC_LOOP);
    EXPECT_EQ(fields[10], param.expectHasBias);
    EXPECT_EQ(fields[11], param.expectApproximate);
    EXPECT_EQ(fields[12], EXPECT_MATMUL_WORKSPACE_SIZE);
    EXPECT_EQ(fields[13], EXPECT_CUBE_CORE_ALIGNED);

    auto workspaceSizes = tilingContext->GetWorkspaceSizes(1);
    ASSERT_NE(workspaceSizes, nullptr);
    EXPECT_EQ(workspaceSizes[0], SYS_WORKSPACE_BYTES + EXPECT_MATMUL_WORKSPACE_SIZE);
}

static FusedMatmulGeluTilingTestParam cases[] = {
    {"fp16_bias_tanh_success",
     ge::DT_FLOAT16,
     {2, 4},
     {3, 4},
     {3},
     {2, 3},
     true,
     1,
     ge::GRAPH_SUCCESS,
     1,
     2,
     4,
     3,
     6,
     1,
     1},

    {"bf16_bias_tanh_success",
     ge::DT_BF16,
     {2, 4},
     {3, 4},
     {3},
     {2, 3},
     true,
     1,
     ge::GRAPH_SUCCESS,
     1,
     2,
     4,
     3,
     6,
     1,
     1},

    {"fp16_no_bias_tanh_success",
     ge::DT_FLOAT16,
     {2, 4},
     {3, 4},
     {},
     {2, 3},
     false,
     1,
     ge::GRAPH_SUCCESS,
     1,
     2,
     4,
     3,
     6,
     0,
     1},

    {"invalid_weight_k_mismatch",
     ge::DT_FLOAT16,
     {2, 4},
     {3, 5},
     {3},
     {2, 3},
     true,
     1,
     ge::GRAPH_FAILED,
     0,
     0,
     0,
     0,
     0,
     0,
     0},

    {"invalid_bias_shape", ge::DT_FLOAT16, {2, 4}, {3, 4}, {4}, {2, 3}, true, 1, ge::GRAPH_FAILED, 0, 0, 0, 0, 0, 0, 0},

    {"invalid_output_shape",
     ge::DT_FLOAT16,
     {2, 4},
     {3, 4},
     {3},
     {2, 4},
     true,
     1,
     ge::GRAPH_FAILED,
     0,
     0,
     0,
     0,
     0,
     0,
     0},

    {"invalid_approximate",
     ge::DT_FLOAT16,
     {2, 4},
     {3, 4},
     {3},
     {2, 3},
     true,
     2,
     ge::GRAPH_FAILED,
     0,
     0,
     0,
     0,
     0,
     0,
     0},
};

INSTANTIATE_TEST_CASE_P(FusedMatmulGelu, FusedMatmulGeluTilingTest, testing::ValuesIn(cases));
