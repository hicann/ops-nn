/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_adaptive_avg_pool2d_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "../../../op_kernel/arch35/adaptive_avg_pool2d_grad_struct.h"

using namespace std;
using namespace ge;
using namespace AdaptiveAvgPool2dGradOp;

struct AdaptiveAvgPool2dGradCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

static const string TEST_OP_TYPE = "AdaptiveAvgPool2dGrad";

static const string COMPILE_INFO_STRING = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
                                                       "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                       "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                       "CORE_NUM": 40}
                                     })";

class AdaptiveAvgPool2dGradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveAvgPool2dGradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool2dGradTilingTest TearDown" << std::endl; }
};

static void SetPlatformResource(
    gert::TilingParseContext* parseContext, map<string, string>& socInfos, map<string, string>& aicoreSpec,
    map<string, string>& intrinsics)
{
    ASSERT_NE(parseContext, nullptr);
    ASSERT_NE(parseContext->GetPlatformInfo(), nullptr);
    ASSERT_TRUE(parseContext->GetPlatformInfo()->Init());
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

static void SetPlatformResource(
    gert::TilingContext* tilingContext, map<string, string>& socInfos, map<string, string>& aicoreSpec,
    map<string, string>& intrinsics)
{
    ASSERT_NE(tilingContext, nullptr);
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

static void PrepareAdaptiveAvgPool2dGradCompileInfo(
    AdaptiveAvgPool2dGradCompileInfo& compileInfo, fe::PlatFormInfos& platformInfo)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), socInfos, aicoreSpec, intrinsics);

    platformInfo.Init();

    auto kernelHolder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()), reinterpret_cast<void*>(&platformInfo)})
            .Outputs({&compileInfo})
            .Build();

    SetPlatformResource(kernelHolder.GetContext<gert::TilingParseContext>(), socInfos, aicoreSpec, intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    ASSERT_EQ(opImpl->tiling_parse(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(compileInfo.coreNum, 40u);
    EXPECT_EQ(compileInfo.ubSize, 196608u);
}

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& shape)
{
    gert::StorageShape storageShape;
    for (auto dim : shape) {
        storageShape.MutableOriginShape().AppendDim(dim);
    }
    for (auto dim : shape) {
        storageShape.MutableStorageShape().AppendDim(dim);
    }
    return storageShape;
}

static void ExecuteAdaptiveAvgPool2dGradTilingCase(
    const std::vector<int64_t>& yGradShapeVec, const std::vector<int64_t>& xShapeVec, ge::DataType dtype,
    ge::Format format, uint64_t* actualTilingKey, uint64_t* actualBlockDim)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    AdaptiveAvgPool2dGradCompileInfo compileInfo;
    PrepareAdaptiveAvgPool2dGradCompileInfo(compileInfo, platformInfo);

    gert::StorageShape yGradShape = MakeStorageShape(yGradShapeVec);
    gert::StorageShape xShape = MakeStorageShape(xShapeVec);
    gert::StorageShape xGradShape = MakeStorageShape(xShapeVec);

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    ASSERT_NE(tilingData, nullptr);
    ASSERT_NE(workspace, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(TEST_OP_TYPE)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, dtype, format, ge::Format::FORMAT_RESERVED)
                      .NodeInputTd(1, dtype, format, ge::Format::FORMAT_RESERVED)
                      .NodeOutputTd(0, dtype, format, ge::Format::FORMAT_RESERVED)
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(xShapeVec)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);

    SetPlatformResource(tilingContext, socInfos, aicoreSpec, intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);

    ASSERT_EQ(opImpl->tiling(tilingContext), ge::GRAPH_SUCCESS);
    EXPECT_GT(tilingContext->GetBlockDim(), 0u);

    if (actualTilingKey != nullptr) {
        *actualTilingKey = tilingContext->GetTilingKey();
    }
    if (actualBlockDim != nullptr) {
        *actualBlockDim = tilingContext->GetBlockDim();
    }
}

static void ExecuteAdaptiveAvgPool2dGradTilingKeyCase(
    const std::vector<int64_t>& yGradShapeVec, const std::vector<int64_t>& xShapeVec, ge::DataType dtype,
    ge::Format format, uint64_t expectTilingKey)
{
    uint64_t actualTilingKey = 0;
    uint64_t actualBlockDim = 0;
    ExecuteAdaptiveAvgPool2dGradTilingCase(yGradShapeVec, xShapeVec, dtype, format, &actualTilingKey, &actualBlockDim);
    EXPECT_EQ(actualTilingKey, expectTilingKey);
    EXPECT_GT(actualBlockDim, 0u);
}

static void ExecuteAdaptiveAvgPool2dGradNotSmallKernelCase(
    const std::vector<int64_t>& yGradShapeVec, const std::vector<int64_t>& xShapeVec, ge::DataType dtype,
    ge::Format format)
{
    uint64_t actualTilingKey = 0;
    uint64_t actualBlockDim = 0;
    ExecuteAdaptiveAvgPool2dGradTilingCase(yGradShapeVec, xShapeVec, dtype, format, &actualTilingKey, &actualBlockDim);

    EXPECT_NE(actualTilingKey, GET_TPL_TILING_KEY(TPL_SMALL_KERNEL, TPL_INT32, 0));
    EXPECT_NE(actualTilingKey, GET_TPL_TILING_KEY(TPL_SMALL_KERNEL, TPL_INT64, 0));
    EXPECT_GT(actualBlockDim, 0u);
}

// ============ Basic Register / Parse Tests ============

TEST_F(AdaptiveAvgPool2dGradTilingTest, op_registered)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, tiling_parse_success)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    AdaptiveAvgPool2dGradCompileInfo compileInfo;

    auto kernelHolder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()), reinterpret_cast<void*>(&platformInfo)})
            .Outputs({&compileInfo})
            .Build();

    SetPlatformResource(kernelHolder.GetContext<gert::TilingParseContext>(), socInfos, aicoreSpec, intrinsics);

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(TEST_OP_TYPE.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    ASSERT_EQ(opImpl->tiling_parse(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    EXPECT_EQ(compileInfo.coreNum, 40u);
    EXPECT_EQ(compileInfo.ubSize, 196608u);
}

// ============ Original Simt / Big Kernel Template Tests ============

TEST_F(AdaptiveAvgPool2dGradTilingTest, tiling_key_simt_kernel_nchw_fp32)
{
    ExecuteAdaptiveAvgPool2dGradTilingKeyCase(
        {1, 4, 8, 8}, {1, 4, 16, 16}, ge::DT_FLOAT, ge::FORMAT_NCHW, GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, TPL_INT32, 0));
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, tiling_key_simt_kernel_chw_fp16)
{
    ExecuteAdaptiveAvgPool2dGradTilingKeyCase(
        {4, 8, 8}, {4, 16, 16}, ge::DT_FLOAT16, ge::FORMAT_NCL, GET_TPL_TILING_KEY(TPL_SIMT_KERNEL, TPL_INT32, 0));
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, tiling_key_big_kernel_nchw_fp32_large_w_window)
{
    ExecuteAdaptiveAvgPool2dGradTilingKeyCase(
        {1, 128, 16, 1}, {1, 128, 16, 512}, ge::DT_FLOAT, ge::FORMAT_NCHW,
        GET_TPL_TILING_KEY(TPL_BIG_KERNEL, TPL_INT32, 0));
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, tiling_key_big_kernel_nchw_fp16_large_w_window)
{
    ExecuteAdaptiveAvgPool2dGradTilingKeyCase(
        {1, 128, 16, 1}, {1, 128, 16, 1024}, ge::DT_FLOAT16, ge::FORMAT_NCHW,
        GET_TPL_TILING_KEY(TPL_BIG_KERNEL, TPL_INT32, 0));
}

TEST_F(AdaptiveAvgPool2dGradTilingTest, tiling_key_big_kernel_chw_fp32_large_w_window)
{
    ExecuteAdaptiveAvgPool2dGradTilingKeyCase(
        {128, 16, 1}, {128, 16, 512}, ge::DT_FLOAT, ge::FORMAT_NCL, GET_TPL_TILING_KEY(TPL_BIG_KERNEL, TPL_INT32, 0));
}

// ============ Small Kernel Positive Branch Coverage ============

struct SmallKernelPositiveCase {
    const char* name;
    std::vector<int64_t> yGradShape;
    std::vector<int64_t> xShape;
    ge::DataType dtype;
    ge::Format format;
    uint64_t expectTilingKey;
};

TEST_F(AdaptiveAvgPool2dGradTilingTest, small_kernel_positive_branch_matrix)
{
    const uint64_t smallInt32Key = GET_TPL_TILING_KEY(TPL_SMALL_KERNEL, TPL_INT32, 0);
    const uint64_t smallInt64Key = GET_TPL_TILING_KEY(TPL_SMALL_KERNEL, TPL_INT64, 0);

    std::vector<SmallKernelPositiveCase> cases = {
        // TrySplitNC success, fp32 branch, NC exactly aligned by highAxisInner.
        {"fp32_try_split_nc_success_high_axis_tail_aligned",
         {1, 2560, 4, 8},
         {1, 2560, 5, 7},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // TrySplitNC success, fp32 branch, highAxisTail != highAxisInner.
        {"fp32_try_split_nc_success_high_axis_tail_unaligned",
         {1, 2603, 4, 8},
         {1, 2603, 5, 7},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // TrySplitNC success, non-fp32 branch, inputW hits INPUTW_BFLOAT_THRESHOLD boundary by fp16.
        {"fp16_try_split_nc_success_input_w_boundary_16",
         {1, 5131, 4, 16},
         {1, 5131, 2, 7},
         ge::DT_FLOAT16,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // TrySplitNC fails by target core number, SplitUnalignHW only needs H dynamic adjustment.
        {"fp32_split_h_to_limit_and_tail_aligned",
         {1, 128, 7, 19},
         {1, 128, 31, 23},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // H dynamic adjustment returns with hOutputTail not equal to hOutputInner.
        {"fp32_split_h_non_aligned_tail",
         {1, 128, 7, 19},
         {1, 128, 53, 23},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // H reaches LIMIT, target core still insufficient, then enters W dynamic adjustment.
        {"fp32_split_h_then_split_w_non_aligned_tail",
         {1, 128, 5, 19},
         {1, 128, 11, 29},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // W dynamic adjustment, but final wOutputTail aligned.
        {"fp32_split_h_then_split_w_aligned_tail",
         {1, 128, 5, 19},
         {1, 128, 11, 30},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // Split W plus highAxisTail unaligned.
        {"fp32_split_w_high_axis_tail_unaligned",
         {1, 193, 5, 17},
         {1, 193, 7, 23},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // Non-fp32 branch, SplitUnalignHW, W branch.
        {"fp16_split_w_branch", {1, 257, 5, 16}, {1, 257, 11, 31}, ge::DT_FLOAT16, ge::FORMAT_NCHW, smallInt32Key},

        // CHW / FORMAT_NCL small kernel path, fp32 branch.
        {"chw_fp32_small_kernel_split_hw", {257, 7, 19}, {257, 13, 29}, ge::DT_FLOAT, ge::FORMAT_NCL, smallInt32Key},

        // CHW / FORMAT_NCL small kernel path, non-fp32 branch.
        {"chw_fp16_small_kernel_split_hw", {521, 5, 16}, {521, 17, 31}, ge::DT_FLOAT16, ge::FORMAT_NCL, smallInt32Key},

        // yGrad spatial is larger than x spatial, covers DoBufferCalculate input-side larger path.
        {"fp32_ygrad_spatial_larger_than_x_spatial",
         {1, 256, 97, 113},
         {1, 256, 17, 19},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // kernelH * kernelW is close to KERNEL_SIZE_MAX but still less than 256.
        {"fp32_kernel_product_255_still_small_kernel",
         {1, 2560, 3, 8},
         {1, 2560, 43, 129},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // Out data count > MAX_INT32, covers GetTilingKey INT64 branch.
        {"fp32_small_kernel_int64_tiling_key",
         {1, 32768, 397, 113},
         {1, 32768, 541, 139},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt64Key},

        // Random non-regular shape, fp32, both H/W need search.
        {"fp32_random_irregular_shape_01",
         {1, 384, 9, 37},
         {1, 384, 71, 83},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // Random non-regular shape, fp16, both H/W not powers of two.
        {"fp16_random_irregular_shape_02",
         {1, 640, 13, 29},
         {1, 640, 41, 67},
         ge::DT_FLOAT16,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // NC barely above threshold, highAxisOuter small, forces HW split strongly.
        {"fp32_nc_barely_above_threshold",
         {1, 129, 6, 23},
         {1, 129, 19, 47},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        // fp16 NC barely above threshold, else branch with W exactly 16.
        {"fp16_nc_barely_above_threshold",
         {1, 129, 7, 16},
         {1, 129, 23, 37},
         ge::DT_FLOAT16,
         ge::FORMAT_NCHW,
         smallInt32Key},
    };

    for (const auto& item : cases) {
        SCOPED_TRACE(item.name);
        ExecuteAdaptiveAvgPool2dGradTilingKeyCase(
            item.yGradShape, item.xShape, item.dtype, item.format, item.expectTilingKey);
    }
}

// ============ Small Kernel IsCapable False Branch Coverage ============

struct SmallKernelNegativeCase {
    const char* name;
    std::vector<int64_t> yGradShape;
    std::vector<int64_t> xShape;
    ge::DataType dtype;
    ge::Format format;
};

TEST_F(AdaptiveAvgPool2dGradTilingTest, small_kernel_capability_false_branch_matrix)
{
    std::vector<SmallKernelNegativeCase> cases = {
        // kernelH * kernelW >= KERNEL_SIZE_MAX, small kernel must reject.
        {"reject_kernel_product_equal_256", {1, 256, 4, 8}, {1, 256, 61, 121}, ge::DT_FLOAT, ge::FORMAT_NCHW},

        // baseData.inputNCSize < HIGH_THRESHOLD.
        {"reject_nc_less_than_128", {1, 127, 4, 8}, {1, 127, 5, 7}, ge::DT_FLOAT, ge::FORMAT_NCHW},

        // gradInputW * gradInputH < WINSIZE_THRESHOLD.
        {"reject_grad_input_window_area_less_than_16", {1, 128, 1, 15}, {1, 128, 3, 17}, ge::DT_FLOAT, ge::FORMAT_NCHW},

        // fp32 branch: gradInputW < INPUTW_FLOAT_THRESHOLD.
        {"reject_fp32_input_w_less_than_8", {1, 128, 4, 7}, {1, 128, 7, 11}, ge::DT_FLOAT, ge::FORMAT_NCHW},

        // non-fp32 branch: gradInputW < INPUTW_BFLOAT_THRESHOLD.
        {"reject_fp16_input_w_less_than_16", {1, 128, 4, 15}, {1, 128, 7, 23}, ge::DT_FLOAT16, ge::FORMAT_NCHW},

        // Initial LIMIT/LIMIT buffer calculation exceeds available UB.
        {"reject_initial_limit_buffer_exceeds_ub", {1, 128, 409, 2049}, {1, 128, 7, 17}, ge::DT_FLOAT, ge::FORMAT_NCHW},

        // CHW format, NC below threshold.
        {"reject_chw_nc_less_than_128", {63, 11, 19}, {63, 23, 31}, ge::DT_FLOAT, ge::FORMAT_NCL},

        // CHW format, fp16 width below non-fp32 threshold.
        {"reject_chw_fp16_input_w_less_than_16", {256, 7, 15}, {256, 19, 23}, ge::DT_FLOAT16, ge::FORMAT_NCL},

        // Irregular large kernel on H only, should not hit small kernel.
        {"reject_large_h_kernel_only", {1, 320, 2, 31}, {1, 320, 600, 37}, ge::DT_FLOAT, ge::FORMAT_NCHW},

        // Irregular large kernel on W only, should not hit small kernel.
        {"reject_large_w_kernel_only", {1, 320, 11, 2}, {1, 320, 17, 600}, ge::DT_FLOAT, ge::FORMAT_NCHW},
    };

    for (const auto& item : cases) {
        SCOPED_TRACE(item.name);
        ExecuteAdaptiveAvgPool2dGradNotSmallKernelCase(item.yGradShape, item.xShape, item.dtype, item.format);
    }
}

// ============ Additional Irregular Small Kernel Smoke Tests ============

TEST_F(AdaptiveAvgPool2dGradTilingTest, small_kernel_more_irregular_shapes_smoke)
{
    const uint64_t smallInt32Key = GET_TPL_TILING_KEY(TPL_SMALL_KERNEL, TPL_INT32, 0);

    std::vector<SmallKernelPositiveCase> cases = {
        {"fp32_irregular_prime_like_01",
         {1, 211, 11, 23},
         {1, 211, 37, 59},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        {"fp32_irregular_prime_like_02",
         {1, 337, 17, 31},
         {1, 337, 43, 71},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        {"fp32_irregular_prime_like_03",
         {1, 719, 19, 41},
         {1, 719, 73, 97},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        {"fp16_irregular_prime_like_04",
         {1, 333, 11, 17},
         {1, 333, 31, 43},
         ge::DT_FLOAT16,
         ge::FORMAT_NCHW,
         smallInt32Key},

        {"fp16_irregular_prime_like_05",
         {1, 777, 13, 19},
         {1, 777, 47, 61},
         ge::DT_FLOAT16,
         ge::FORMAT_NCHW,
         smallInt32Key},

        {"fp32_non_power_of_two_wide_input",
         {1, 191, 8, 47},
         {1, 191, 29, 211},
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         smallInt32Key},

        {"fp16_non_power_of_two_wide_input",
         {1, 263, 9, 53},
         {1, 263, 35, 199},
         ge::DT_FLOAT16,
         ge::FORMAT_NCHW,
         smallInt32Key},

        {"fp32_tall_shape", {1, 401, 23, 9}, {1, 401, 151, 17}, ge::DT_FLOAT, ge::FORMAT_NCHW, smallInt32Key},

        {"fp16_tall_shape", {1, 529, 29, 16}, {1, 529, 173, 31}, ge::DT_FLOAT16, ge::FORMAT_NCHW, smallInt32Key},

        {"chw_fp32_irregular_01", {383, 13, 23}, {383, 47, 89}, ge::DT_FLOAT, ge::FORMAT_NCL, smallInt32Key},

        {"chw_fp16_irregular_02", {769, 17, 29}, {769, 67, 101}, ge::DT_FLOAT16, ge::FORMAT_NCL, smallInt32Key},
    };

    for (const auto& item : cases) {
        SCOPED_TRACE(item.name);
        ExecuteAdaptiveAvgPool2dGradTilingKeyCase(
            item.yGradShape, item.xShape, item.dtype, item.format, item.expectTilingKey);
    }
}