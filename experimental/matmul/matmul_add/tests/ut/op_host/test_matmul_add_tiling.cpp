/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_common.h"

using namespace ge;
using namespace std;

namespace {

constexpr uint64_t TILING_KEY_FP16 = 10;
constexpr uint64_t TILING_KEY_BF16 = 30;

const char* kCompileInfo = R"({
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
        "CORE_NUM": 48
    }
})";

ge::graphStatus RunTiling(
    ge::DataType dtype,
    gert::StorageShape aShape,
    gert::StorageShape bShape,
    gert::StorageShape biasShape,
    gert::StorageShape yShape,
    uint64_t* tilingKey)
{
    std::string opType("MatmulAdd");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    int32_t compileInfo = 0;

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
        .SetOpType("MatmulAdd")
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&aShape, &bShape, &biasShape})
        .OutputShapes({&yShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();

    auto* context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = opImpl->tiling(context);
    if (tilingKey != nullptr) {
        *tilingKey = context->GetTilingKey();
    }
    return ret;
}

} // namespace

class MatmulAddTilingTest : public testing::Test {};

TEST_F(MatmulAddTilingTest, tiling_with_bias_fp16_success)
{
    uint64_t tilingKey = 0;
    EXPECT_EQ(RunTiling(
        ge::DT_FLOAT16,
        {{64, 128}, {64, 128}},
        {{128, 32}, {128, 32}},
        {{32}, {32}},
        {{64, 32}, {64, 32}},
        &tilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, TILING_KEY_FP16);
}

TEST_F(MatmulAddTilingTest, tiling_without_bias_bf16_success)
{
    uint64_t tilingKey = 0;
    EXPECT_EQ(RunTiling(
        ge::DT_BF16,
        {{32, 64}, {32, 64}},
        {{64, 16}, {64, 16}},
        {{}, {}},
        {{32, 16}, {32, 16}},
        &tilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, TILING_KEY_BF16);
}

TEST_F(MatmulAddTilingTest, k_mismatch_returns_failed)
{
    EXPECT_NE(RunTiling(
        ge::DT_FLOAT16,
        {{64, 128}, {64, 128}},
        {{127, 32}, {127, 32}},
        {{32}, {32}},
        {{64, 32}, {64, 32}},
        nullptr), ge::GRAPH_SUCCESS);
}

TEST_F(MatmulAddTilingTest, bias_length_mismatch_returns_failed)
{
    EXPECT_NE(RunTiling(
        ge::DT_FLOAT16,
        {{64, 128}, {64, 128}},
        {{128, 32}, {128, 32}},
        {{31}, {31}},
        {{64, 32}, {64, 32}},
        nullptr), ge::GRAPH_SUCCESS);
}

TEST_F(MatmulAddTilingTest, unsupported_dtype_returns_failed)
{
    EXPECT_NE(RunTiling(
        ge::DT_FLOAT,
        {{64, 128}, {64, 128}},
        {{128, 32}, {128, 32}},
        {{32}, {32}},
        {{64, 32}, {64, 32}},
        nullptr), ge::GRAPH_SUCCESS);
}
