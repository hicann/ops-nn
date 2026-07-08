/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../../../op_host/glu_grad_tiling.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"

using namespace ge;
using namespace std;
using namespace ut_util;

namespace {
constexpr int64_t TILING_KEY_SMALL_SHAPE_FP16 = 1;
constexpr int64_t TILING_KEY_BIG_SHAPE_BF16 = 283;

struct TilingCase {
    gert::StorageShape gradOutShape;
    gert::StorageShape selfShape;
    gert::StorageShape outShape;
    ge::DataType dtype;
    int64_t dim;
    int64_t expectedTilingKey;
};

std::string GetCompileInfoString()
{
    return R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}})";
}

ge::graphStatus RunTiling(const TilingCase& tilingCase, int64_t* actualTilingKey = nullptr)
{
    auto compileInfoString = GetCompileInfoString();
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::GluGradCompileInfo compileInfo;

    const string opType("GLUGrad");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    auto parseContext = kernelHolder.GetContext<gert::TilingParseContext>();
    EXPECT_TRUE(parseContext->GetPlatformInfo()->Init());
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({const_cast<gert::StorageShape*>(&tilingCase.gradOutShape),
                          const_cast<gert::StorageShape*>(&tilingCase.selfShape)})
                      .OutputShapes({const_cast<gert::StorageShape*>(&tilingCase.outShape)})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(tilingCase.dim)}})
                      .NodeInputTd(0, tilingCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, tilingCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, tilingCase.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tilingFunc(context);
    if (actualTilingKey != nullptr && ret == ge::GRAPH_SUCCESS) {
        *actualTilingKey = context->GetTilingKey();
    }
    return ret;
}
} // namespace

class GluGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GluGradTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GluGradTilingTest TearDown" << std::endl;
    }
};

TEST_F(GluGradTilingTest, tiling_small_shape)
{
    TilingCase tilingCase{{{1, 80, 1280}, {1, 80, 1280}},
        {{1, 80, 2560}, {1, 80, 2560}},
        {{1, 80, 2560}, {1, 80, 2560}},
        ge::DT_FLOAT16,
        -1,
        TILING_KEY_SMALL_SHAPE_FP16};
    int64_t actualTilingKey = 0;
    EXPECT_EQ(RunTiling(tilingCase, &actualTilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(actualTilingKey, tilingCase.expectedTilingKey);
}

TEST_F(GluGradTilingTest, tiling_big_shape)
{
    TilingCase tilingCase{{{1, 4, 10240}, {1, 4, 10240}},
        {{1, 8, 10240}, {1, 8, 10240}},
        {{1, 8, 10240}, {1, 8, 10240}},
        ge::DT_BF16,
        1,
        TILING_KEY_BIG_SHAPE_BF16};
    int64_t actualTilingKey = 0;
    EXPECT_EQ(RunTiling(tilingCase, &actualTilingKey), ge::GRAPH_SUCCESS);
    EXPECT_EQ(actualTilingKey, tilingCase.expectedTilingKey);
}

TEST_F(GluGradTilingTest, tiling_invalid_split_dim)
{
    TilingCase tilingCase{{{2, 2}, {2, 2}},
        {{2, 5}, {2, 5}},
        {{2, 5}, {2, 5}},
        ge::DT_FLOAT,
        -1,
        0};
    EXPECT_EQ(RunTiling(tilingCase), ge::GRAPH_FAILED);
}

TEST_F(GluGradTilingTest, tiling_unsupported_dtype)
{
    TilingCase tilingCase{{{2, 2}, {2, 2}},
        {{2, 4}, {2, 4}},
        {{2, 4}, {2, 4}},
        ge::DT_INT32,
        -1,
        0};
    EXPECT_EQ(RunTiling(tilingCase), ge::GRAPH_FAILED);
}
