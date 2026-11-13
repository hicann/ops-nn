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
#include <map>
#include <string>

#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_kernel/arch35/fake_quant_with_min_max_vars_tiling_data.h"

namespace {
struct FakeQuantWithMinMaxVarsCompileInfo {};

const char* kCompileInfo = R"({"hardware_info": {
    "CORE_NUM": 48, "vector_core_cnt": 48, "UB_SIZE": 196608,
    "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072
}})";

struct TilingCase {
    gert::StorageShape shape;
    int64_t numBits;
    bool narrowRange;
    bool defaultAttrs;
    int32_t numCore;
    int32_t blockFactor;
    int32_t blockTailFactor;
    int32_t baseLen;
};

// min/max are per-tensor scalar inputs: the kernel unconditionally copies a
// fixed 4-byte float from each of them, so any shape other than (1,) would
// read out of bounds or take an undefined value and must be rejected by tiling.
struct InvalidScalarCase {
    gert::StorageShape xShape;
    gert::StorageShape minShape;
    gert::StorageShape maxShape;
};
} // namespace

class FakeQuantWithMinMaxVarsTilingTest : public testing::TestWithParam<TilingCase> {};

class FakeQuantWithMinMaxVarsTilingRejectTest : public testing::TestWithParam<InvalidScalarCase> {};

TEST_P(FakeQuantWithMinMaxVarsTilingTest, tiling_data)
{
    auto testCase = GetParam();
    fe::PlatFormInfos platformInfo;
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);
    ASSERT_TRUE(platformInfo.Init());
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxVars");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    FakeQuantWithMinMaxVarsCompileInfo compileInfo;
    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(2, 1)
                           .Inputs({const_cast<char*>(kCompileInfo), &platformInfo})
                           .Outputs({&compileInfo})
                           .Build();
    ASSERT_EQ(impl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape minShape = {{1}, {1}};
    gert::StorageShape maxShape = {{1}, {1}};
    gert::StorageShape yShape = testCase.shape;
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspaceHolder, nullptr);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto faker = gert::TilingContextFaker();
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&testCase.shape, &minShape, &maxShape})
        .OutputShapes({&yShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(tilingData.get())
        .Workspace(workspace);
    if (!testCase.defaultAttrs) {
        faker.NodeAttrs({
            {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(testCase.numBits)},
            {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(testCase.narrowRange)},
        });
    }
    auto holder = faker.Build();
    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->tiling(context), ge::GRAPH_SUCCESS);

    auto raw = context->GetRawTilingData();
    ASSERT_NE(raw, nullptr);
    ASSERT_GE(raw->GetDataSize(), sizeof(FakeQuantWithMinMaxVarsTilingData));
    auto td = reinterpret_cast<const FakeQuantWithMinMaxVarsTilingData*>(raw->GetData());
    ASSERT_NE(td, nullptr);
    const int64_t totalLength = testCase.shape.GetStorageShape().GetShapeSize();
    EXPECT_EQ(td->totalLength, totalLength);
    EXPECT_EQ(td->dim0, 1);
    EXPECT_EQ(td->dim1, totalLength);
    EXPECT_EQ(td->dim2, 1);
    EXPECT_EQ(td->numBits, testCase.numBits);
    EXPECT_EQ(td->narrowRange, testCase.narrowRange);
    EXPECT_EQ(td->numCore, testCase.numCore);
    EXPECT_EQ(context->GetBlockDim(), testCase.numCore);
    // MODE=0, BUFFER_MODE=1, ROUND_MODE=0, HAS_ZP=0 → key = 4. ROUND_MODE is
    // selected as 0 (CAST_FLOOR) to match the kernel's hardcoded rounding.
    EXPECT_EQ(context->GetTilingKey(), 4U);
    EXPECT_EQ(context->GetWorkspaceNum(), 1U);
    ASSERT_NE(context->GetWorkspaceSizes(1), nullptr);
    EXPECT_EQ(context->GetWorkspaceSizes(1)[0], 0U);
    EXPECT_EQ(td->blockFactor, testCase.blockFactor);
    EXPECT_EQ(td->blockTailFactor, testCase.blockTailFactor);
    EXPECT_EQ(td->baseLen, testCase.baseLen);
    EXPECT_EQ(static_cast<int64_t>(td->blockFactor) * (td->numCore - 1) + td->blockTailFactor, totalLength);
    EXPECT_FALSE(td->hasZeroPoint);
    EXPECT_EQ(td->axis, 0);
    EXPECT_EQ(td->sqrtMode, 0);
    if (totalLength != 0) {
        EXPECT_EQ(td->blockAxis, 1);
        EXPECT_EQ(td->ubAxis, 1);
        EXPECT_EQ(td->blockUnion, 1);
        EXPECT_EQ(td->baseN, 1);
        EXPECT_EQ(td->roundMode, 0); // synced to the kernel's CAST_FLOOR rounding
        EXPECT_EQ(td->baseLen % 32, 0);
        EXPECT_LE(static_cast<int64_t>(td->baseLen) * 20 + 160 + 2048, 196608);
    }
}

INSTANTIATE_TEST_SUITE_P(ShapesAndAttrs, FakeQuantWithMinMaxVarsTilingTest,
                         testing::Values(TilingCase{{{1}, {1}}, 8, false, false, 1, 1, 1, 32},
                                         TilingCase{{{8}, {8}}, 8, false, false, 1, 8, 8, 32},
                                         TilingCase{{{32}, {32}}, 8, false, false, 1, 32, 32, 32},
                                         TilingCase{{{33}, {33}}, 8, false, false, 2, 32, 1, 32},
                                         TilingCase{{{2, 3, 4096}, {2, 3, 4096}}, 8, false, false, 48, 512, 512, 512},
                                         TilingCase{{{1537}, {1537}}, 8, false, false, 25, 64, 1, 64},
                                         TilingCase{{{1048576}, {1048576}}, 8, false, false, 48, 21856, 21344, 9696},
                                         TilingCase{{{256}, {256}}, 2, false, false, 8, 32, 32, 32},
                                         TilingCase{{{256}, {256}}, 16, false, false, 8, 32, 32, 32},
                                         TilingCase{{{256}, {256}}, 8, true, false, 8, 32, 32, 32},
                                         TilingCase{{{256}, {256}}, 8, false, true, 8, 32, 32, 32},
                                         TilingCase{{{0, 4}, {0, 4}}, 8, false, false, 1, 0, 0, 0},
                                         TilingCase{{{0}, {0}}, 16, true, false, 1, 0, 0, 0},
                                         // Empty x with valid min/max still goes through the
                                         // empty-input fast path and succeeds — the success
                                         // counterpart of the empty-x reject case below.
                                         TilingCase{{{0}, {0}}, 8, false, false, 1, 0, 0, 0}));

// Tiling must reject min/max inputs whose shape is not (1,) before any kernel
// launch: an empty or multi-element scalar tensor is read by a fixed 4-byte
// DataCopyPad in the kernel, which is out of bounds for shape (0,) and takes
// an undefined element for multi-element shapes.
// The check must also fire for an empty x: it runs before the empty-input
// early exit, so an empty x combined with an invalid min/max is rejected
// instead of silently taking the fast path.
TEST_P(FakeQuantWithMinMaxVarsTilingRejectTest, tiling_reject_invalid_min_max_shape)
{
    auto testCase = GetParam();
    fe::PlatFormInfos platformInfo;
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);
    ASSERT_TRUE(platformInfo.Init());
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetCoreNumByCoreType("AICore");
    platformInfo.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxVars");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling_parse, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    FakeQuantWithMinMaxVarsCompileInfo compileInfo;
    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(2, 1)
                           .Inputs({const_cast<char*>(kCompileInfo), &platformInfo})
                           .Outputs({&compileInfo})
                           .Build();
    ASSERT_EQ(impl->tiling_parse(parseHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape yShape = testCase.xShape;
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspaceHolder, nullptr);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto faker = gert::TilingContextFaker();
    faker.NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&testCase.xShape, &testCase.minShape, &testCase.maxShape})
        .OutputShapes({&yShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(tilingData.get())
        .Workspace(workspace);
    auto holder = faker.Build();
    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_NE(impl->tiling(context), ge::GRAPH_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(InvalidMinMaxShape, FakeQuantWithMinMaxVarsTilingRejectTest,
                         testing::Values(InvalidScalarCase{{{8}, {8}}, {{0}, {0}}, {{1}, {1}}},
                                         InvalidScalarCase{{{8}, {8}}, {{1}, {1}}, {{0}, {0}}},
                                         InvalidScalarCase{{{8}, {8}}, {{2}, {2}}, {{1}, {1}}},
                                         InvalidScalarCase{{{8}, {8}}, {{1}, {1}}, {{2}, {2}}},
                                         // Empty x with an invalid min: the scalar shape check must
                                         // fire before the empty-input early exit, otherwise this
                                         // combination would silently succeed.
                                         InvalidScalarCase{{{0}, {0}}, {{0}, {0}}, {{1}, {1}}}));
