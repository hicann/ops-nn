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
 * \file test_max_pool3d_grad_ndhwc_big_tiling.cpp
 * \brief tiling UT for the MaxPool3DGrad NDHWC big-kernel template (input split, overlap-only).
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../../../op_host/arch35/max_pool3d_grad_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;
using namespace Pool3DGradNameSpace;

class MaxPool3DGradNdhwcBigTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPool3DGradNdhwcBigTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MaxPool3DGradNdhwcBigTiling TearDown" << std::endl; }
};

static void RunBigTilingCase(gert::StorageShape& xShape, gert::StorageShape& yShape, gert::StorageShape& dxShape,
                             std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrList,
                             std::vector<int64_t> ksize, std::vector<int64_t> strides, ge::DataType dataType,
                             uint64_t expectTilingKey, bool expectBig = true, int64_t expectArgmaxBytes = -1,
                             int64_t expectTotalBlocks = -1, int64_t expectCInner = -1, int64_t expectCTail = -1,
                             int64_t expectInputBytes = -1, int64_t expectGradBytes = -1,
                             int64_t expectOutputBytes = -1, int64_t expectDOuter = -1, int64_t expectDInner = -1,
                             int64_t expectHOuter = -1, int64_t expectHInner = -1, int64_t expectWOuter = -1,
                             int64_t expectWInner = -1)
{
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string COMPILE_INFO_STRING = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 64}})";
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), socInfos, aicoreSpec, intrinsics);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::Tiling4Pool3DGradCompileInfo compileInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()),
                                     reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_EQ(gert::OpImplRegistry::GetInstance()
                  .GetOpImpl("MaxPool3DGrad")
                  ->tiling_parse(kernelHolder.GetContext<gert::KernelContext>()),
              ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DGrad")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &yShape, &yShape})
                      .OutputShapes({&dxShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dataType, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(1, dataType, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(2, dataType, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeOutputTd(0, dataType, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeAttrs(attrList)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    std::map<std::string, std::string> versionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", versionInfos);

    EXPECT_EQ(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->tiling(tilingContext), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tilingContext->GetTilingKey(), expectTilingKey);
    ASSERT_GT(tilingContext->GetBlockDim(), 0);
    if (!expectBig) {
        return;
    }
    const auto* tilingData = tilingContext->GetTilingData<Pool3DGradNDHWCTilingData>();
    ASSERT_NE(tilingData, nullptr);

    // ---- 输入参数（公式自洽断言用）----
    const int64_t nX = static_cast<int64_t>(xShape.GetStorageShape().GetDim(0));
    const int64_t dX = static_cast<int64_t>(xShape.GetStorageShape().GetDim(1));
    const int64_t hX = static_cast<int64_t>(xShape.GetStorageShape().GetDim(2));
    const int64_t wX = static_cast<int64_t>(xShape.GetStorageShape().GetDim(3));
    const int64_t cX = static_cast<int64_t>(xShape.GetStorageShape().GetDim(4));
    const int64_t dP = static_cast<int64_t>(yShape.GetStorageShape().GetDim(1));
    const int64_t hP = static_cast<int64_t>(yShape.GetStorageShape().GetDim(2));
    const int64_t wP = static_cast<int64_t>(yShape.GetStorageShape().GetDim(3));
    ASSERT_EQ(ksize.size(), 5U);
    ASSERT_EQ(strides.size(), 5U);
    const int64_t kD = ksize[1], kH = ksize[2], kW = ksize[3];
    const int64_t sD = strides[1], sH = strides[2], sW = strides[3];
    const int64_t kv = kD * kH * kW;
    const int64_t bytes = (dataType == ge::DT_FLOAT) ? 4 : 2;
    const int64_t alignNum = (dataType == ge::DT_FLOAT) ? 8 : 16; // max(32B/bytes, 32B/4B idx)

    EXPECT_EQ(tilingData->base.highAxisInner, 1);
    EXPECT_EQ(tilingData->cDim, cX);
    EXPECT_EQ(tilingData->isBigKernel, 1);
    EXPECT_GE(tilingData->base.normalCoreProcessNum, 1);

    // ---- 三轴 inner/outer/tail 自洽（input 切分）----
    const int64_t dI = tilingData->base.dOutputInner;
    const int64_t hI = tilingData->base.hOutputInner;
    const int64_t wI = tilingData->base.wOutputInner;
    const int64_t dO = tilingData->base.dOutputOuter;
    const int64_t hO = tilingData->base.hOutputOuter;
    const int64_t wO = tilingData->base.wOutputOuter;
    EXPECT_EQ(dO, (dX + dI - 1) / dI);
    EXPECT_EQ(tilingData->base.dOutputTail, dX - (dO - 1) * dI);
    EXPECT_EQ(hO, (hX + hI - 1) / hI);
    EXPECT_EQ(tilingData->base.hOutputTail, hX - (hO - 1) * hI);
    EXPECT_EQ(wO, (wX + wI - 1) / wI);
    EXPECT_EQ(tilingData->base.wOutputTail, wX - (wO - 1) * wI);

    // ---- buffer 公式自洽 ----
    const int64_t cAligned = (tilingData->cOutputInner + alignNum - 1) / alignNum * alignNum;
    const int64_t covD = (dI + kD - 1 + sD - 1) / sD;
    const int64_t covH = (hI + kH - 1 + sH - 1) / sH;
    const int64_t covW = (wI + kW - 1 + sW - 1) / sW;
    // 单窗口口径为上界; 超 UB 时按前向可用预算封顶（kernel 按 maxCount_ 分块合并）, 下限一个 W 位行
    EXPECT_LE(tilingData->base.inputBufferSize, kD * kH * kW * cAligned * bytes);
    EXPECT_GE(tilingData->base.inputBufferSize, cAligned * bytes);
    EXPECT_EQ(tilingData->base.outputBufferSize, dI * hI * wI * cAligned * 4);
    // 统一 tiling: grad/argmax 恒为重算区间（RF）口径（原不切分档的精确 pooled 计数已随
    // TryPlaneSplit 移除, 大小 kernel 共用 DoBufferCalculate 的 RF 公式）
    EXPECT_EQ(tilingData->base.gradBufferSize, covD * covH * covW * cAligned * bytes);
    EXPECT_EQ(tilingData->base.argmaxBufferSize, covD * covH * covW * tilingData->cOutputInner * 4);

    // ---- block 数自洽：totalBlocks = nX × cOuter × dO × hO × wO ----
    const int64_t totalBlocks = nX * tilingData->cOutputOuter * dO * hO * wO;
    EXPECT_EQ(tilingData->base.normalCoreProcessNum * (tilingData->base.usedCoreNum - 1) +
                  tilingData->base.tailCoreProcessNum,
              totalBlocks);
    EXPECT_EQ(static_cast<int64_t>(tilingContext->GetBlockDim()), tilingData->base.usedCoreNum);

    // ---- 逐 case 锚点 ----
    if (expectTotalBlocks > 0) {
        EXPECT_EQ(totalBlocks, expectTotalBlocks);
    }
    if (expectCInner > 0) {
        EXPECT_EQ(tilingData->cOutputInner, expectCInner);
        EXPECT_EQ(tilingData->cOutputOuter, (cX + expectCInner - 1) / expectCInner);
    }
    if (expectCTail > 0) {
        EXPECT_EQ(tilingData->cOutputTail, expectCTail);
    }
    if (expectInputBytes > 0) {
        EXPECT_EQ(tilingData->base.inputBufferSize, expectInputBytes);
    }
    if (expectGradBytes > 0) {
        EXPECT_EQ(tilingData->base.gradBufferSize, expectGradBytes);
    }
    if (expectOutputBytes > 0) {
        EXPECT_EQ(tilingData->base.outputBufferSize, expectOutputBytes);
    }
    if (expectArgmaxBytes >= 0) {
        EXPECT_EQ(tilingData->base.argmaxBufferSize, expectArgmaxBytes);
    }
    if (expectDOuter > 0) {
        EXPECT_EQ(dO, expectDOuter);
    }
    if (expectDInner > 0) {
        EXPECT_EQ(dI, expectDInner);
    }
    if (expectHOuter > 0) {
        EXPECT_EQ(hO, expectHOuter);
    }
    if (expectHInner > 0) {
        EXPECT_EQ(hI, expectHInner);
    }
    if (expectWOuter > 0) {
        EXPECT_EQ(wO, expectWOuter);
    }
    if (expectWInner > 0) {
        EXPECT_EQ(wI, expectWInner);
    }
}

static std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeAttrs(std::vector<int64_t> ksize,
                                                                        std::vector<int64_t> strides,
                                                                        std::string padding, std::vector<int64_t> pads)
{
    return {{"ksize", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(ksize)},
            {"strides", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(strides)},
            {"padding", Ops::NN::AnyValue::CreateFrom<string>(padding)},
            {"pads", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>(pads)},
            {"data_format", Ops::NN::AnyValue::CreateFrom<string>("NDHWC")}};
}

// (6,6,6) 三轴等维：默认启发式贪心 → (d=6, h=6, w=2)，即 dI=hI=1、wI=3
// grad=6*6*8*32*2=18432；argmax=6*6*8*32*4=36864；input=216*32*2=13824(单窗×2)
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_overlap_small_c_fp16)
{
    gert::StorageShape xShape = {{1, 6, 6, 6, 32}, {1, 6, 6, 6, 32}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 32}, {1, 1, 1, 1, 32}};
    gert::StorageShape dxShape = {{1, 6, 6, 6, 32}, {1, 6, 6, 6, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 50688, 36, 32, 32, 13824, 25344, 768, 6, 1, 6, 1,
                     1, 6);
}

TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_overlap_stride2_fp16)
{
    // (6,6,6) s=2：covD=covH=3、covW=4；grad=1152；argmax=2304
    gert::StorageShape xShape = {{1, 6, 6, 6, 16}, {1, 6, 6, 6, 16}};
    gert::StorageShape yShape = {{1, 3, 3, 3, 16}, {1, 3, 3, 3, 16}};
    gert::StorageShape dxShape = {{1, 6, 6, 6, 16}, {1, 6, 6, 6, 16}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 2, 2, 2, 1}, "SAME", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 2, 2, 2, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 3456, 36, 16, 16, 6912, 1728, 384, 6, 1, 6, 1, 1,
                     6);
}

// N=64 铺满核：blocksBefore ≥ coreNum → 不切分（DoBufferCalculate 整平面口径）
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_multicore_n64_fp16)
{
    gert::StorageShape xShape = {{64, 6, 6, 6, 32}, {64, 6, 6, 6, 32}};
    gert::StorageShape yShape = {{64, 1, 1, 1, 32}, {64, 1, 1, 1, 32}};
    gert::StorageShape dxShape = {{64, 6, 6, 6, 32}, {64, 6, 6, 6, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 108416, 192, 32, 32, 13824, 54208, 9216, 3, 2, 1,
                     6, 1, 6);
}

// C=40 非对齐：cAligned=48、argmax 按 cOutputInner=40 计
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_unaligned_c40_fp16)
{
    gert::StorageShape xShape = {{1, 6, 6, 6, 40}, {1, 6, 6, 6, 40}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 40}, {1, 1, 1, 1, 40}};
    gert::StorageShape dxShape = {{1, 6, 6, 6, 40}, {1, 6, 6, 6, 40}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 63360, 36, 40, 40, 20736, 38016, 1152, 6, 1, 6, 1,
                     1, 6);
}

// pad=3 + stride=2：(8,8,8) → 贪心 (d=8, h=8, w=1)；covD=covH=3、covW=7
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_pad3_stride2_fp16)
{
    gert::StorageShape xShape = {{1, 8, 8, 8, 32}, {1, 8, 8, 8, 32}};
    gert::StorageShape yShape = {{1, 5, 5, 5, 32}, {1, 5, 5, 5, 32}};
    gert::StorageShape dxShape = {{1, 8, 8, 8, 32}, {1, 8, 8, 8, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 2, 2, 2, 1}, "CALCULATED", {3, 3, 3, 3, 3, 3});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 2, 2, 2, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 8064, 64, 32, 32, 13824, 4032, 1024, 8, 1, 8, 1,
                     1, 8);
}

// P0-44 后 non-overlap 由大 kernel 认领（原 reject_non_overlap_falls_to_simt 过时）
TEST_F(MaxPool3DGradNdhwcBigTiling, non_overlap_k_eq_s_cube_claimed_by_big_kernel)
{
    gert::StorageShape xShape = {{1, 12, 12, 12, 32}, {1, 12, 12, 12, 32}};
    gert::StorageShape yShape = {{1, 2, 2, 2, 32}, {1, 2, 2, 2, 32}};
    gert::StorageShape dxShape = {{1, 12, 12, 12, 32}, {1, 12, 12, 12, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 6, 6, 6, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 6, 6, 6, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 512, 72, 32, 32, 13824, 256, 3072, 12, 1, 3, 4, 2,
                     6);
}

// C 轴 tile（cInner=64）+ 单窗口 input 预算后, UT 平台（UB=196608）亦可容纳 → BIG 认领
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_c256_ut_platform_claimed_fp16)
{
    gert::StorageShape xShape = {{1, 6, 6, 6, 256}, {1, 6, 6, 6, 256}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 256}, {1, 1, 1, 1, 256}};
    gert::StorageShape dxShape = {{1, 6, 6, 6, 256}, {1, 6, 6, 6, 256}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 101376, 144, 64, 64, 27648, 50688, 1536, 6, 1, 6,
                     1, 1, 6);
}

// ==================== C 轴 tile（cX > vReg/bytes） ====================
// UT 平台 UB=196608（fp16 chunk=128, fp32 chunk=64；cAligned 按 16/8 元素对齐）

// (5,5,7) C=256 fp16：升级循环到三轴全切 inner=1（totalTry≈141）
//   covD=5 covH=5 covW=7 → argmax=5*5*7*128*4=89600；input=175*128*2=44800(×2)
//   total=89600+89600=179200 ≤ availableUb → BIG；blocks=2×5×5×7=350
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_c_tile_c256_escalate_fp16)
{
    gert::StorageShape xShape = {{1, 5, 5, 7, 256}, {1, 5, 5, 7, 256}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 256}, {1, 1, 1, 1, 256}};
    gert::StorageShape dxShape = {{1, 5, 5, 7, 256}, {1, 5, 5, 7, 256}};
    auto attrs = MakeAttrs({1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 83200, 100, 64, 64, 22400, 41600, 1792, 5, 1, 5,
                     1, 1, 7);
}

TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_c_tile_c192_tail64_bf16)
{
    gert::StorageShape xShape = {{1, 5, 5, 7, 192}, {1, 5, 5, 7, 192}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 192}, {1, 1, 1, 1, 192}};
    gert::StorageShape dxShape = {{1, 5, 5, 7, 192}, {1, 5, 5, 7, 192}};
    auto attrs = MakeAttrs({1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, ge::DT_BF16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 83200, 75, 64, 64, 22400, 41600, 1792, 5, 1, 5, 1,
                     1, 7);
}

// fp32 chunk=64：(5,5,7) 默认启发式即过门 → (w=7, d=5, h=1)；blocks=2×5×1×7=70
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_c_tile_c128_fp32)
{
    gert::StorageShape xShape = {{1, 5, 5, 7, 128}, {1, 5, 5, 7, 128}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 128}, {1, 1, 1, 1, 128}};
    gert::StorageShape dxShape = {{1, 5, 5, 7, 128}, {1, 5, 5, 7, 128}};
    auto attrs = MakeAttrs({1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 83200, 50, 64, 64, 44800, 83200, 1792, 5, 1, 5, 1,
                     1, 7);
}

TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_c_tile_c300_tail44_fp16)
{
    gert::StorageShape xShape = {{1, 5, 5, 7, 300}, {1, 5, 5, 7, 300}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 300}, {1, 1, 1, 1, 300}};
    gert::StorageShape dxShape = {{1, 5, 5, 7, 300}, {1, 5, 5, 7, 300}};
    auto attrs = MakeAttrs({1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 83200, 125, 64, 44, 22400, 41600, 1792, 5, 1, 5,
                     1, 1, 7);
}

// C tile + pad + stride2：覆盖窗口数收缩（covW=3），预算充裕，默认启发式过门
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_c_tile_c256_pad_stride2_fp16)
{
    gert::StorageShape xShape = {{1, 5, 5, 7, 256}, {1, 5, 5, 7, 256}};
    gert::StorageShape yShape = {{1, 1, 1, 2, 256}, {1, 1, 1, 2, 256}};
    gert::StorageShape dxShape = {{1, 5, 5, 7, 256}, {1, 5, 5, 7, 256}};
    auto attrs = MakeAttrs({1, 5, 5, 6, 1}, {1, 1, 1, 2, 1}, "CALCULATED", {0, 0, 0, 0, 1, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 5, 5, 6, 1}, {1, 1, 1, 2, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 76800, 50, 128, 128, 38400, 38400, 3584, 5, 1, 5,
                     1, 1, 7);
}

TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_c_tile_c256_n2_multicore_fp16)
{
    gert::StorageShape xShape = {{2, 5, 5, 7, 256}, {2, 5, 5, 7, 256}};
    gert::StorageShape yShape = {{2, 1, 1, 1, 256}, {2, 1, 1, 1, 256}};
    gert::StorageShape dxShape = {{2, 5, 5, 7, 256}, {2, 5, 5, 7, 256}};
    auto attrs = MakeAttrs({1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 83200, 200, 64, 64, 22400, 41600, 1792, 5, 1, 5,
                     1, 1, 7);
}

// cX == chunk 边界：cOutputOuter == 1
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_single_tile_c128_fp16)
{
    gert::StorageShape xShape = {{1, 5, 5, 7, 128}, {1, 5, 5, 7, 128}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 128}, {1, 1, 1, 1, 128}};
    gert::StorageShape dxShape = {{1, 5, 5, 7, 128}, {1, 5, 5, 7, 128}};
    auto attrs = MakeAttrs({1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 83200, 50, 64, 64, 22400, 41600, 1792, 5, 1, 5, 1,
                     1, 7);
}

// cX < chunk：cAligned=112、argmax 按 cX=100 计；默认启发式 (w=7,d=5,h=2) 即过门
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_single_tile_c100_fp16)
{
    gert::StorageShape xShape = {{1, 5, 5, 7, 100}, {1, 5, 5, 7, 100}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 100}, {1, 1, 1, 1, 100}};
    gert::StorageShape dxShape = {{1, 5, 5, 7, 100}, {1, 5, 5, 7, 100}};
    auto attrs = MakeAttrs({1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 5, 5, 7, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 83200, 50, 64, 36, 22400, 41600, 1792, 5, 1, 5, 1,
                     1, 7);
}

// ==================== D 轴深切分（input D 远大于 pooled D 家族） ====================
// (1,16,6,6,32) k=(6,6,6) s=1 → pooled=(11,1,1)：贪心 (d=16, h=4, w=1)
//   dI=1, hI=2, wI=6；covD=6 covH=7 covW=11；blocks=16×3×1=48
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_plane_split_pd_fp16)
{
    gert::StorageShape xShape = {{1, 16, 6, 6, 32}, {1, 16, 6, 6, 32}};
    gert::StorageShape yShape = {{1, 11, 1, 1, 32}, {1, 11, 1, 1, 32}};
    gert::StorageShape dxShape = {{1, 16, 6, 6, 32}, {1, 16, 6, 6, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 50688, 96, 32, 32, 13824, 25344, 768, 16, 1, 6, 1,
                     1, 6);
}

TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_plane_split_pd_fp32)
{
    gert::StorageShape xShape = {{1, 16, 6, 6, 8}, {1, 16, 6, 6, 8}};
    gert::StorageShape yShape = {{1, 11, 1, 1, 8}, {1, 11, 1, 1, 8}};
    gert::StorageShape dxShape = {{1, 16, 6, 6, 8}, {1, 16, 6, 6, 8}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 12672, 96, 8, 8, 6912, 12672, 192, 16, 1, 6, 1, 1,
                     6);
}

// (1,12,6,6,32) pooled=(7,1,1)：贪心 (d=12, h=6, w=1)；covD=6 covH=6 covW=11
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_plane_split_off_shallow_pd)
{
    gert::StorageShape xShape = {{1, 12, 6, 6, 32}, {1, 12, 6, 6, 32}};
    gert::StorageShape yShape = {{1, 7, 1, 1, 32}, {1, 7, 1, 1, 32}};
    gert::StorageShape dxShape = {{1, 12, 6, 6, 32}, {1, 12, 6, 6, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 50688, 72, 32, 32, 13824, 25344, 768, 12, 1, 6, 1,
                     1, 6);
}

// (1,32,6,6,32)：贪心 (d=32, h=2, w=1) 直接过门（无需升级）；blocks=32×2=64
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_deep_d_split_fp16)
{
    gert::StorageShape xShape = {{1, 32, 6, 6, 32}, {1, 32, 6, 6, 32}};
    gert::StorageShape yShape = {{1, 27, 1, 1, 32}, {1, 27, 1, 1, 32}};
    gert::StorageShape dxShape = {{1, 32, 6, 6, 32}, {1, 32, 6, 6, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 67584, 64, 32, 32, 13824, 33792, 2304, 32, 1, 2,
                     3, 1, 6);
}

// (1,32,6,6,128)：最小预算 2×216×128×4=221184 > UT availableUb → SIMT
// （ST 平台 253952 仍 BIG）
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_c128_ut_platform_claimed_fp16)
{
    gert::StorageShape xShape = {{1, 32, 6, 6, 128}, {1, 32, 6, 6, 128}};
    gert::StorageShape yShape = {{1, 27, 1, 1, 128}, {1, 27, 1, 1, 128}};
    gert::StorageShape dxShape = {{1, 32, 6, 6, 128}, {1, 32, 6, 6, 128}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 101376, 384, 64, 64, 27648, 50688, 1536, 32, 1, 6,
                     1, 1, 6);
}

// ==================== H/W 宽切分 + 三轴混合 ====================

// (1,6,24,24,32) pooled=(1,19,19)：贪心 (h=24, w=3, d=1)；hI=1 wI=8 dI=6
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_hw_split_wide_h_w_fp16)
{
    gert::StorageShape xShape = {{1, 6, 24, 24, 32}, {1, 6, 24, 24, 32}};
    gert::StorageShape yShape = {{1, 1, 19, 19, 32}, {1, 1, 19, 19, 32}};
    gert::StorageShape dxShape = {{1, 6, 24, 24, 32}, {1, 6, 24, 24, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 133632, 144, 32, 32, 13824, 66816, 3072, 6, 1, 24,
                     1, 1, 24);
}

// (1,16,16,16,32) pooled=(11,11,11)：升级循环 → (d=16, h=8, w=1)；dI=1 hI=2 wI=16
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_tri_axis_split_fp16)
{
    gert::StorageShape xShape = {{1, 16, 16, 16, 32}, {1, 16, 16, 16, 32}};
    gert::StorageShape yShape = {{1, 11, 11, 11, 32}, {1, 11, 11, 11, 32}};
    gert::StorageShape dxShape = {{1, 16, 16, 16, 32}, {1, 16, 16, 16, 32}};
    auto attrs = MakeAttrs({1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 6, 6, 6, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 112896, 128, 32, 32, 13824, 56448, 4096, 16, 1, 8,
                     2, 1, 16);
}

TEST_F(MaxPool3DGradNdhwcBigTiling, reject_small_ksize_falls_to_simt)
{
    gert::StorageShape xShape = {{1, 6, 6, 6, 32}, {1, 6, 6, 6, 32}};
    gert::StorageShape yShape = {{1, 5, 5, 5, 32}, {1, 5, 5, 5, 32}};
    gert::StorageShape dxShape = {{1, 6, 6, 6, 32}, {1, 6, 6, 6, 32}};
    auto attrs = MakeAttrs({1, 2, 2, 2, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    // 联调合一版: kv<128 本模板拒绝后落到 NDHWCSmallKernelTiling
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 2, 2, 2, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), false);
}

TEST_F(MaxPool3DGradNdhwcBigTiling, reject_huge_ksize_falls_to_simt)
{
    gert::StorageShape xShape = {{1, 16, 16, 16, 32}, {1, 16, 16, 16, 32}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 32}, {1, 1, 1, 1, 32}};
    gert::StorageShape dxShape = {{1, 16, 16, 16, 32}, {1, 16, 16, 16, 32}};
    auto attrs = MakeAttrs({1, 16, 16, 16, 1}, {1, 1, 1, 1, 1}, "CALCULATED", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 16, 16, 16, 1}, {1, 1, 1, 1, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 1, 1, 0, 0), false);
}

// ==================== MP3G 最小化阶梯探针（临时·只打印不断言） ====================
static void RunBigTilingProbe(int64_t dX, const char* tag)
{
    const int64_t hX = 80, wX = 52, cX = 8;
    const int64_t dP = (dX - 4) / 16 + 1;
    const int64_t hP = (80 - 4) / 10 + 1;
    const int64_t wP = (52 - 8) / 6 + 1;
    gert::StorageShape xShape = {{1, dX, hX, wX, cX}, {1, dX, hX, wX, cX}};
    gert::StorageShape yShape = {{1, dP, hP, wP, cX}, {1, dP, hP, wP, cX}};
    gert::StorageShape dxShape = {{1, dX, hX, wX, cX}, {1, dX, hX, wX, cX}};
    auto attrs = MakeAttrs({1, 4, 4, 8, 1}, {1, 16, 10, 6, 1}, "VALID", {0, 0, 0, 0, 0, 0});

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string COMPILE_INFO_STRING = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
    "CORE_NUM": 64}})";
    GetPlatFormInfos(COMPILE_INFO_STRING.c_str(), socInfos, aicoreSpec, intrinsics);
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::Tiling4Pool3DGradCompileInfo compileInfo;
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(COMPILE_INFO_STRING.c_str()),
                                     reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_EQ(gert::OpImplRegistry::GetInstance()
                  .GetOpImpl("MaxPool3DGrad")
                  ->tiling_parse(kernelHolder.GetContext<gert::KernelContext>()),
              ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPool3DGrad")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &yShape, &yShape})
                      .OutputShapes({&dxShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeAttrs(attrs)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    std::map<std::string, std::string> versionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", versionInfos);

    auto st = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3DGrad")->tiling(tilingContext);
    ASSERT_EQ(st, ge::GRAPH_SUCCESS);
    const auto* t = tilingContext->GetTilingData<Pool3DGradNDHWCTilingData>();
    ASSERT_NE(t, nullptr);
    const int64_t totalBlocks = t->base.dOutputOuter * t->base.hOutputOuter * t->base.wOutputOuter * t->cOutputOuter;
    std::cout << "[LADDER][" << tag << "] d=" << dX << " key=" << tilingContext->GetTilingKey()
              << " big=" << t->isBigKernel << " blockDim=" << tilingContext->GetBlockDim() << " inner(d,h,w)=("
              << t->base.dOutputInner << "," << t->base.hOutputInner << "," << t->base.wOutputInner << ")"
              << " outer=(" << t->base.dOutputOuter << "," << t->base.hOutputOuter << "," << t->base.wOutputOuter << ")"
              << " c(in/out)=(" << t->cOutputInner << "/" << t->cOutputOuter << ")"
              << " blocks=" << totalBlocks << " core(normal/tail/used)=(" << t->base.normalCoreProcessNum << "/"
              << t->base.tailCoreProcessNum << "/" << t->base.usedCoreNum << ")"
              << " buf(in/gr/am/y)=(" << t->base.inputBufferSize << "/" << t->base.gradBufferSize << "/"
              << t->base.argmaxBufferSize << "/" << t->base.outputBufferSize << ")" << std::endl;
}

TEST_F(MaxPool3DGradNdhwcBigTiling, mp3g_minimize_ladder_probe)
{
    RunBigTilingProbe(392, "A1");
    RunBigTilingProbe(196, "A2");
    RunBigTilingProbe(100, "A3");
    RunBigTilingProbe(52, "A4");
    RunBigTilingProbe(36, "A5");
    RunBigTilingProbe(20, "A6");
    RunBigTilingProbe(17, "A7");
}

// ==================== P0-44: non-overlap 门控放开 ====================
// 大 kernel 数学（PStart/PEnd、逐窗 DMA、四级 Scatter）对 k<=s 天然成立。
// 放开前这些形状被 IsCapable 拒收后落到小 kernel tiling 的 useBigKernel
// 预算分支（单窗 inputBufferSize），小 kernel 前向装整 tile 越界
// （mp3g_*_anc_g1 系列 error 341/82）；放开后由大 kernel 正确认领。

// k == s（g1_09/g1_10 形态）: kv=128, C=8, fp16 —— 应被大 kernel 认领
TEST_F(MaxPool3DGradNdhwcBigTiling, non_overlap_k_eq_s_claimed_by_big_kernel)
{
    // x=(2,13,13,65,8), k=(4,4,8), s=(4,4,8) → pooled=(2,3,3,8,8) (VALID 语义)
    gert::StorageShape xShape = {{2, 13, 13, 65, 8}, {2, 13, 13, 65, 8}};
    gert::StorageShape yShape = {{2, 3, 3, 8, 8}, {2, 3, 3, 8, 8}};
    gert::StorageShape dxShape = {{2, 13, 13, 65, 8}, {2, 13, 13, 65, 8}};
    auto attrs = MakeAttrs({1, 4, 4, 8, 1}, {1, 4, 4, 8, 1}, "VALID", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 4, 4, 8, 1}, {1, 4, 4, 8, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 0, 0), true, 2048, 288, 64, 8, 16384, 1024, 32768, 4, 4, 4, 4,
                     9, 8);
}

// k < s（g1_11/g1_12 形态）: kv=128, C=8, fp16 —— 应被大 kernel 认领
TEST_F(MaxPool3DGradNdhwcBigTiling, non_overlap_k_lt_s_claimed_by_big_kernel)
{
    // x=(2,13,13,65,8), k=(4,4,8), s=(7,6,11) → pooled=(2,2,2,6,8) (VALID 语义)
    gert::StorageShape xShape = {{2, 13, 13, 65, 8}, {2, 13, 13, 65, 8}};
    gert::StorageShape yShape = {{2, 2, 2, 6, 8}, {2, 2, 2, 6, 8}};
    gert::StorageShape dxShape = {{2, 13, 13, 65, 8}, {2, 13, 13, 65, 8}};
    auto attrs = MakeAttrs({1, 4, 4, 8, 1}, {1, 7, 6, 11, 1}, "VALID", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 4, 4, 8, 1}, {1, 7, 6, 11, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 0, 0), true, 2048, 72, 64, 8, 16384, 1024, 118272, 2, 7, 3, 6,
                     6, 11);
}

// k == s + 大 C（g1_10 精确形态）: kv=128, C=64, fp16 —— 大 kernel 认领,
// inputBufferSize 必须为单窗口语义（4*4*8*64*2=16384B），不得出现小 kernel 的
// 整 tile 期望跨度公式（P0-43 错配形态即 16KB 队列装 ~1.3MB tile）
TEST_F(MaxPool3DGradNdhwcBigTiling, non_overlap_k_eq_s_big_c_budget_single_window)
{
    // x=(2,13,13,65,64), k=(4,4,8), s=(4,4,8) → pooled=(2,3,3,8,64) (VALID 语义)
    gert::StorageShape xShape = {{2, 13, 13, 65, 64}, {2, 13, 13, 65, 64}};
    gert::StorageShape yShape = {{2, 3, 3, 8, 64}, {2, 3, 3, 8, 64}};
    gert::StorageShape dxShape = {{2, 13, 13, 65, 64}, {2, 13, 13, 65, 64}};
    auto attrs = MakeAttrs({1, 4, 4, 8, 1}, {1, 4, 4, 8, 1}, "VALID", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 4, 4, 8, 1}, {1, 4, 4, 8, 1}, ge::DT_FLOAT16,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 0, 0), true, 2048, 288, 64, 64, 16384, 1024, 32768, 4, 4, 4, 4,
                     9, 8);
}

// 超 UB 预算封顶认领: kv=512 fp32, 整窗 input×2 超 UT availableUb → input 按预算封顶,
// 窗口元素数(512×32) > maxCount_(halved) → kernel 走 split 分块合并
TEST_F(MaxPool3DGradNdhwcBigTiling, big_kernel_input_halved_split_engaged_fp32)
{
    // kv=512, C=64, s=2（batch=64 < 256）: 整窗 input×2=262368 超 UT availableUb
    // → input 封顶至 79232 认领; 窗口元素 512×64 > maxCount_ 19808 → split 生效
    gert::StorageShape xShape = {{1, 9, 9, 9, 64}, {1, 9, 9, 9, 64}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 64}, {1, 1, 1, 1, 64}};
    gert::StorageShape dxShape = {{1, 9, 9, 9, 64}, {1, 9, 9, 9, 64}};
    auto attrs = MakeAttrs({1, 8, 8, 8, 1}, {1, 2, 2, 2, 1}, "VALID", {0, 0, 0, 0, 0, 0});
    RunBigTilingCase(xShape, yShape, dxShape, attrs, {1, 8, 8, 8, 1}, {1, 2, 2, 2, 1}, ge::DT_FLOAT,
                     GET_TPL_TILING_KEY(TPL_INT32, 0, 1, 1, 0), true, 32768, 81, 64, 64, 79232, 32768, 2304, 9, 1, 9, 1,
                     1, 9);
}
