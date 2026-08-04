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
 * \file test_normalize_bbox_tiling.cpp
 * \brief NormalizeBBox Tiling UT (iteration-1 core path)
 *
 * Coverage:
 *   - Core path: batch>1 -> splitMode==1 (split by batch); batch==1 -> splitMode==0 (split by num)
 *   - TilingKey: reversed_box 0/1 select different keys (GET_TPL_TILING_KEY)
 *   - reversed layout: num read from dim2, normal layout: num read from dim1
 *   - 5 shape/attr validation branches + dtype guard return GRAPH_FAILED
 *   - TilingParse (platform info -> compile info) prepare path
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../op_kernel/arch35/normalize_bbox_tiling_data.h"

using namespace ge;
using namespace std;

namespace {
// Mirror of optiling::NormalizeBBoxCompileInfo POD layout (int32 + uint64, default pack 8 -> size 16).
struct NormalizeBBoxCompileInfoStub {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

constexpr int32_t kCoreNum = 64;
constexpr uint64_t kUbSize = 245760; // 240 KB (Ascend950)

struct TilingRunResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    NormalizeBBoxTilingData tilingData{};
    bool tilingDataValid = false;
};

// Build a StorageShape from a dim vector (origin == storage).
gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t d : dims) {
        shape.MutableStorageShape().AppendDim(d);
        shape.MutableOriginShape().AppendDim(d);
    }
    return shape;
}

// Drive the full 7-step tiling with a pre-filled compile info (GetPlatformInfo reads coreNum/ubSize
// straight from CompileInfo, so we don't depend on the platform API returning a core count here).
// coreNum/ubSize are overridable so we can exercise the platform-info guard (0 values) and the
// small-UB tileLen clamp branch (ubSize <= UB_RESERVE or floor-aligned tileLen < blockAlign).
TilingRunResult RunTiling(const std::vector<int64_t>& boxesDims, const std::vector<int64_t>& shapeHwDims,
                          bool reversedBox, ge::DataType boxesDtype, int32_t coreNum = kCoreNum,
                          uint64_t ubSize = kUbSize)
{
    TilingRunResult result;

    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("NormalizeBBox");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return result;
    }
    auto tilingFunc = opImpl->tiling;

    NormalizeBBoxCompileInfoStub compileInfo;
    compileInfo.totalCoreNum = coreNum;
    compileInfo.ubSizePlatForm = ubSize;

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    auto param = gert::TilingData::CreateCap(4096);
    if (param == nullptr) {
        return result;
    }
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    gert::StorageShape boxesShape = MakeShape(boxesDims);
    gert::StorageShape shapeHwShape = MakeShape(shapeHwDims);
    gert::StorageShape yShape = MakeShape(boxesDims);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("NormalizeBBox")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, boxesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, boxesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reversed_box", Ops::NN::AnyValue::CreateFrom<bool>(reversedBox)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr) {
        return result;
    }

    result.status = tilingFunc(tilingContext);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.tilingKey = tilingContext->GetTilingKey();
        const auto* raw = tilingContext->GetRawTilingData();
        if (raw != nullptr && raw->GetDataSize() >= sizeof(NormalizeBBoxTilingData)) {
            result.tilingData = *reinterpret_cast<const NormalizeBBoxTilingData*>(raw->GetData());
            result.tilingDataValid = true;
        }
    }
    return result;
}
} // namespace

class NormalizeBBoxTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NormalizeBBoxTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "NormalizeBBoxTiling TearDown" << std::endl; }
};

// ---- Core path: batch>1 -> splitMode==1 (split by batch) ------------------------------------
TEST_F(NormalizeBBoxTiling, tiling_multi_batch_split_by_batch)
{
    auto r = RunTiling({64, 1024, 4}, {64, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 1u); // batch > 1
    EXPECT_EQ(r.tilingData.batch, 64u);
    EXPECT_EQ(r.tilingData.num, 1024u);
    EXPECT_EQ(r.tilingData.coordNum, 4u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 64u); // min(batch, coreNum) = min(64,64)
    EXPECT_GT(r.tilingData.tileLen, 0u);
}

// batch>1 but batch < coreNum: usedCoreNum clamps to batch
TEST_F(NormalizeBBoxTiling, tiling_multi_batch_small_batch)
{
    auto r = RunTiling({2, 8, 4}, {2, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 1u);
    EXPECT_EQ(r.tilingData.batch, 2u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 2u); // min(2, 64)
}

// ---- Core path: batch==1 -> splitMode==0 (split by num) -------------------------------------
TEST_F(NormalizeBBoxTiling, tiling_single_batch_split_by_num)
{
    auto r = RunTiling({1, 1024, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u); // batch == 1
    EXPECT_EQ(r.tilingData.batch, 1u);
    EXPECT_EQ(r.tilingData.num, 1024u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 64u); // min(num, coreNum) = min(1024,64)
}

// batch==1, num < coreNum: single-core fallback path
TEST_F(NormalizeBBoxTiling, tiling_single_batch_tiny_num)
{
    auto r = RunTiling({1, 1, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 1u); // min(1, 64)
}

// ---- reversed layout (batch,4,num): num read from dim2 --------------------------------------
TEST_F(NormalizeBBoxTiling, tiling_reversed_layout_num_from_dim2)
{
    auto r = RunTiling({2, 4, 8}, {2, 3}, /*reversedBox=*/true, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.batch, 2u);
    EXPECT_EQ(r.tilingData.num, 8u);       // reversed -> num = dim2
    EXPECT_EQ(r.tilingData.splitMode, 1u); // batch > 1
}

// ---- TilingKey: reversed_box 0/1 must select different keys ---------------------------------
TEST_F(NormalizeBBoxTiling, tiling_key_differs_by_reversed_box)
{
    auto normal = RunTiling({2, 8, 4}, {2, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    auto reversed = RunTiling({2, 4, 8}, {2, 3}, /*reversedBox=*/true, ge::DT_FLOAT);
    ASSERT_EQ(normal.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(reversed.status, ge::GRAPH_SUCCESS);
    // reversedBox is the only TPL axis carried in the key -> the two keys must differ.
    EXPECT_NE(normal.tilingKey, reversed.tilingKey);
}

// ---- fp16 dtype path (blockAlign / tileLen computed with sizeof(half)) ----------------------
TEST_F(NormalizeBBoxTiling, tiling_fp16_dtype)
{
    auto r = RunTiling({4, 256, 4}, {4, 3}, /*reversedBox=*/false, ge::DT_FLOAT16);
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.batch, 4u);
    EXPECT_EQ(r.tilingData.num, 256u);
    EXPECT_GT(r.tilingData.tileLen, 0u);
}

// ============================================================================================
// Validation branches (GetShapeAttrsInfo) -> tiling returns GRAPH_FAILED
// ============================================================================================

// 1. boxes rank out of [2, 8]
TEST_F(NormalizeBBoxTiling, tiling_fail_boxes_rank_out_of_range)
{
    auto r = RunTiling({4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// 2. shape_hw not 2-D
TEST_F(NormalizeBBoxTiling, tiling_fail_shape_hw_not_2d)
{
    auto r = RunTiling({2, 8, 4}, {2, 3, 1}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// 3. shape_hw second dim != 3
TEST_F(NormalizeBBoxTiling, tiling_fail_shape_hw_dim1_not_3)
{
    auto r = RunTiling({2, 8, 4}, {2, 2}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// 4a. normal layout boxes[2] != 4
TEST_F(NormalizeBBoxTiling, tiling_fail_normal_coord_not_4)
{
    auto r = RunTiling({2, 8, 3}, {2, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// 4b. reversed layout boxes[1] != 4
TEST_F(NormalizeBBoxTiling, tiling_fail_reversed_coord_not_4)
{
    auto r = RunTiling({2, 3, 8}, {2, 3}, /*reversedBox=*/true, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// 5. batch mismatch between boxes and shape_hw
TEST_F(NormalizeBBoxTiling, tiling_fail_batch_mismatch)
{
    auto r = RunTiling({2, 8, 4}, {3, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// ============================================================================================
// TilingParse prepare path (platform info -> NormalizeBBoxCompileInfo) + tiling end-to-end
// ============================================================================================
TEST_F(NormalizeBBoxTiling, tiling_parse_prepare_and_run)
{
    std::string opType("NormalizeBBox");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    ASSERT_NE(opImpl, nullptr);
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;
    ASSERT_NE(tilingFunc, nullptr);
    ASSERT_NE(tilingParseFunc, nullptr);

    std::string compileInfoString = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                          "Intrinsic_fix_pipe_l0c2out": false,
                                                          "Intrinsic_data_move_l12ub": true,
                                                          "Intrinsic_data_move_l0c2ub": true,
                                                          "Intrinsic_data_move_out2l1_nd2nz": false,
                                                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                          "CORE_NUM": 64}
                                       })";
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> socVersionInfos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    NormalizeBBoxCompileInfoStub compileInfo;

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs(
                                {const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    auto parseCtx = kernelHolder.GetContext<gert::TilingParseContext>();
    parseCtx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseCtx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parseCtx->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.totalCoreNum, 0);
    EXPECT_GT(compileInfo.ubSizePlatForm, 0u);

    // run tiling with the parsed compile info
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    gert::StorageShape boxesShape = MakeShape({8, 64, 4});
    gert::StorageShape shapeHwShape = MakeShape({8, 3});
    gert::StorageShape yShape = MakeShape({8, 64, 4});

    auto holder = gert::TilingContextFaker()
                      .SetOpType("NormalizeBBox")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&boxesShape, &shapeHwShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reversed_box", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

// ############################################################################################
// Iteration-2: Tiling branch coverage for the integrated blockAlign segment model +
// dual split-mode across all four (dtype x layout) keys + boundaries.
//
// Reference (normalize_bbox_regbase_tiling.cpp::DoOpTiling, coreNum = kCoreNum = 64):
//   blockAlign (frames per 32B):  alignFrames = reversedBox ? 32/dsize : 32/(coordNum*dsize)
//       normal  fp32 -> 2   |  normal  fp16 -> 4
//       reversed fp32 -> 8  |  reversed fp16 -> 16
//   split-by-num (batch==1):
//       perCoreRaw  = CeilDiv(num, coreNum)
//       numPerCore  = CeilDiv(perCoreRaw, alignFrames) * alignFrames   (align up -> 32B start)
//       usedCoreNum = CeilDiv(num, numPerCore)                         (front-full / tail-last)
//       front-full invariants: numBigCore == usedCoreNum, tailNumCore == numPerCore
//   split-by-batch (batch>1, dtype/layout agnostic):
//       usedCoreNum  = min(batch, coreNum)
//       batchPerCore = CeilDiv(batch, usedCoreNum)
//       bigCoreNum   = batch - usedCoreNum * (batchPerCore - 1)
//       tailBatchNum = batchPerCore - 1
//   tileLen: UB(=kUbSize 245760) budget floored to 256B repeat then capped at MAX_TILE_LEN 8192
//            -> deterministically 8192 for both fp16/fp32 under this stub.
// ############################################################################################

// ---- blockAlign segment boundary: normal fp32 (alignFrames = 2) -----------------------------
// num=300 -> perCoreRaw=ceil(300/64)=5 -> numPerCore=ceil(5/2)*2=6 -> usedCoreNum=ceil(300/6)=50
TEST_F(NormalizeBBoxTiling, tiling_blockalign_normal_fp32)
{
    auto r = RunTiling({1, 300, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.num, 300u);
    EXPECT_EQ(r.tilingData.numPerCore, 6u); // aligned up from raw 5 to a multiple of 2
    EXPECT_EQ(r.tilingData.numPerCore % 2u, 0u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 50u);
    // front-full / tail-last model invariants
    EXPECT_EQ(r.tilingData.numBigCore, r.tilingData.usedCoreNum);
    EXPECT_EQ(r.tilingData.tailNumCore, r.tilingData.numPerCore);
}

// ---- blockAlign segment boundary: normal fp16 (alignFrames = 4) -----------------------------
// num=300 -> perCoreRaw=5 -> numPerCore=ceil(5/4)*4=8 -> usedCoreNum=ceil(300/8)=38
TEST_F(NormalizeBBoxTiling, tiling_blockalign_normal_fp16)
{
    auto r = RunTiling({1, 300, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT16);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.numPerCore, 8u); // multiple of 4 (fp16 normal frame is 8B)
    EXPECT_EQ(r.tilingData.numPerCore % 4u, 0u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 38u);
    EXPECT_EQ(r.tilingData.numBigCore, r.tilingData.usedCoreNum);
}

// ---- blockAlign segment boundary: reversed fp32 (alignFrames = 8) ---------------------------
// reversed layout {1,4,num}; num=300 -> perCoreRaw=5 -> numPerCore=ceil(5/8)*8=8 -> used=ceil(300/8)=38
TEST_F(NormalizeBBoxTiling, tiling_blockalign_reversed_fp32)
{
    auto r = RunTiling({1, 4, 300}, {1, 3}, /*reversedBox=*/true, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.num, 300u);      // reversed -> num from dim2
    EXPECT_EQ(r.tilingData.numPerCore, 8u); // reversed row aligns per element -> 8 (32/4B)
    EXPECT_EQ(r.tilingData.numPerCore % 8u, 0u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 38u);
}

// ---- blockAlign segment boundary: reversed fp16 (alignFrames = 16) --------------------------
// reversed {1,4,num}; num=300 -> perCoreRaw=5 -> numPerCore=ceil(5/16)*16=16 -> used=ceil(300/16)=19
TEST_F(NormalizeBBoxTiling, tiling_blockalign_reversed_fp16)
{
    auto r = RunTiling({1, 4, 300}, {1, 3}, /*reversedBox=*/true, ge::DT_FLOAT16);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.numPerCore, 16u); // reversed fp16 element align -> 16 (32/2B)
    EXPECT_EQ(r.tilingData.numPerCore % 16u, 0u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 19u);
}

// ---- blockAlign differentiation: dtype changes the segment alignment (same num) -------------
// Locks that fp32 and fp16 pick different numPerCore for identical num in num-split.
TEST_F(NormalizeBBoxTiling, tiling_blockalign_dtype_differentiates_segment)
{
    auto fp32 = RunTiling({1, 300, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    auto fp16 = RunTiling({1, 300, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT16);
    ASSERT_EQ(fp32.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(fp16.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(fp32.tilingData.numPerCore, 6u);
    EXPECT_EQ(fp16.tilingData.numPerCore, 8u);
    EXPECT_NE(fp32.tilingData.numPerCore, fp16.tilingData.numPerCore);
    // reversed layout widens the alignment further vs normal for the same dtype
    auto rev32 = RunTiling({1, 4, 300}, {1, 3}, /*reversedBox=*/true, ge::DT_FLOAT);
    ASSERT_EQ(rev32.status, ge::GRAPH_SUCCESS);
    EXPECT_GT(rev32.tilingData.numPerCore, fp32.tilingData.numPerCore); // 8 > 6
}

// ---- large-scale single-batch num-split: all 64 cores engaged, balanced -------------------
// num=100000 -> perCoreRaw=ceil(100000/64)=1563 -> numPerCore=ceil(1563/2)*2=1564 -> used=64
TEST_F(NormalizeBBoxTiling, tiling_num_split_large_scale_fullcore)
{
    auto r = RunTiling({1, 100000, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 64u); // clamped to full core count
    EXPECT_EQ(r.tilingData.numPerCore, 1564u);
    EXPECT_EQ(r.tilingData.numPerCore % 2u, 0u);
    // usedCoreNum*numPerCore >= num (front-full covers all frames, last core clamps the rest)
    EXPECT_GE(r.tilingData.usedCoreNum * r.tilingData.numPerCore, r.tilingData.num);
}

// ============================================================================================
// Dual split-mode: split-by-batch (splitMode==1) load balancing across dtype/layout keys
// ============================================================================================

// batch=100 (>coreNum): 36 big cores *2 + 28 small cores *1 = 100
TEST_F(NormalizeBBoxTiling, tiling_batch_split_loadbalance_over_corenum)
{
    auto r = RunTiling({100, 8, 4}, {100, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 1u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 64u);
    EXPECT_EQ(r.tilingData.batchPerCore, 2u);
    EXPECT_EQ(r.tilingData.bigCoreNum, 36u);
    EXPECT_EQ(r.tilingData.tailBatchNum, 1u);
    // load conservation: bigCoreNum*batchPerCore + (used-bigCoreNum)*tailBatchNum == batch
    uint64_t small = r.tilingData.usedCoreNum - r.tilingData.bigCoreNum;
    EXPECT_EQ(r.tilingData.bigCoreNum * r.tilingData.batchPerCore + small * r.tilingData.tailBatchNum,
              r.tilingData.batch);
}

// batch=130 reversed fp16: batchPerCore=3, 2 big cores *3 + 62 small *2 = 130
TEST_F(NormalizeBBoxTiling, tiling_batch_split_loadbalance_reversed_fp16)
{
    auto r = RunTiling({130, 4, 8}, {130, 3}, /*reversedBox=*/true, ge::DT_FLOAT16);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 1u);
    EXPECT_EQ(r.tilingData.num, 8u); // reversed -> dim2
    EXPECT_EQ(r.tilingData.usedCoreNum, 64u);
    EXPECT_EQ(r.tilingData.batchPerCore, 3u);
    EXPECT_EQ(r.tilingData.bigCoreNum, 2u);
    EXPECT_EQ(r.tilingData.tailBatchNum, 2u);
    uint64_t small = r.tilingData.usedCoreNum - r.tilingData.bigCoreNum;
    EXPECT_EQ(r.tilingData.bigCoreNum * r.tilingData.batchPerCore + small * r.tilingData.tailBatchNum,
              r.tilingData.batch);
}

// batch-split params are independent of dtype/layout key (only reversedBox changes num source)
TEST_F(NormalizeBBoxTiling, tiling_batch_split_params_key_agnostic)
{
    auto nf32 = RunTiling({100, 8, 4}, {100, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    auto nf16 = RunTiling({100, 8, 4}, {100, 3}, /*reversedBox=*/false, ge::DT_FLOAT16);
    auto rf32 = RunTiling({100, 4, 8}, {100, 3}, /*reversedBox=*/true, ge::DT_FLOAT);
    auto rf16 = RunTiling({100, 4, 8}, {100, 3}, /*reversedBox=*/true, ge::DT_FLOAT16);
    ASSERT_EQ(nf32.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(nf16.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(rf32.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(rf16.status, ge::GRAPH_SUCCESS);
    // identical batch split (usedCoreNum / batchPerCore / bigCoreNum) across all 4 keys
    for (const auto* r : {&nf16, &rf32, &rf16}) {
        EXPECT_EQ(r->tilingData.usedCoreNum, nf32.tilingData.usedCoreNum);
        EXPECT_EQ(r->tilingData.batchPerCore, nf32.tilingData.batchPerCore);
        EXPECT_EQ(r->tilingData.bigCoreNum, nf32.tilingData.bigCoreNum);
        EXPECT_EQ(r->tilingData.splitMode, 1u);
    }
}

// ============================================================================================
// TilingKey space {dtype compile axis} x {reversedBox TPL}: host key carries only reversedBox
// ============================================================================================
TEST_F(NormalizeBBoxTiling, tiling_key_dtype_axis_not_in_host_key)
{
    // Same reversedBox, different dtype -> same host TilingKey (dtype resolved by DTYPE_* binary axis)
    auto k0_f32 = RunTiling({2, 8, 4}, {2, 3}, /*reversedBox=*/false, ge::DT_FLOAT);   // K0
    auto k2_f16 = RunTiling({2, 8, 4}, {2, 3}, /*reversedBox=*/false, ge::DT_FLOAT16); // K2
    auto k1_f32 = RunTiling({2, 4, 8}, {2, 3}, /*reversedBox=*/true, ge::DT_FLOAT);    // K1
    auto k3_f16 = RunTiling({2, 4, 8}, {2, 3}, /*reversedBox=*/true, ge::DT_FLOAT16);  // K3
    ASSERT_EQ(k0_f32.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(k2_f16.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(k1_f32.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(k3_f16.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(k0_f32.tilingKey, k2_f16.tilingKey); // reversedBox=0 -> one key regardless of dtype
    EXPECT_EQ(k1_f32.tilingKey, k3_f16.tilingKey); // reversedBox=1 -> one key regardless of dtype
    EXPECT_NE(k0_f32.tilingKey, k1_f32.tilingKey); // reversedBox flips the key
}

// ============================================================================================
// Boundaries: empty tensor / minimal num / tail non-aligned / tileLen budget
// ============================================================================================

// empty tensor num==0, batch==1 -> fast path: splitMode=0, usedCoreNum=1, split params zeroed
TEST_F(NormalizeBBoxTiling, tiling_empty_tensor_num0_single_batch)
{
    auto r = RunTiling({1, 0, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.num, 0u);
    EXPECT_EQ(r.tilingData.splitMode, 0u); // batch==1
    EXPECT_EQ(r.tilingData.usedCoreNum, 1u);
    EXPECT_EQ(r.tilingData.numPerCore, 0u);
    EXPECT_EQ(r.tilingData.batchPerCore, 0u);
    EXPECT_EQ(r.tilingData.numBigCore, 0u);
}

// empty tensor num==0, batch>1 -> fast path: splitMode=1 (batch>1) but usedCoreNum clamps to 1
TEST_F(NormalizeBBoxTiling, tiling_empty_tensor_num0_multi_batch)
{
    auto r = RunTiling({4, 0, 4}, {4, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.batch, 4u);
    EXPECT_EQ(r.tilingData.num, 0u);
    EXPECT_EQ(r.tilingData.splitMode, 1u); // batch>1 in empty fast path
    EXPECT_EQ(r.tilingData.usedCoreNum, 1u);
    EXPECT_EQ(r.tilingData.batchPerCore, 0u); // batch split not computed on empty fast path
}

// minimal num == one alignFrame (fp32 normal alignFrames=2): single-core, numPerCore==2
TEST_F(NormalizeBBoxTiling, tiling_num_split_minimal_num)
{
    auto r = RunTiling({1, 2, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.num, 2u);
    EXPECT_EQ(r.tilingData.numPerCore, 2u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 1u);
}

// tail non-aligned num (num=3, alignFrames=2): 2 cores, last core count clamped in kernel
TEST_F(NormalizeBBoxTiling, tiling_num_split_tail_non_aligned)
{
    auto r = RunTiling({1, 3, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.num, 3u);
    EXPECT_EQ(r.tilingData.numPerCore, 2u);  // aligned segment length
    EXPECT_EQ(r.tilingData.usedCoreNum, 2u); // ceil(3/2)=2; core0=2 frames, core1 clamps 1
    // numPerCore does not divide num -> genuine tail-clamp scenario
    EXPECT_NE(r.tilingData.num % r.tilingData.numPerCore, 0u);
}

// tileLen: deterministic UB budget (kUbSize 245760) -> capped at MAX_TILE_LEN 8192, both dtypes
TEST_F(NormalizeBBoxTiling, tiling_tilelen_ub_budget_capped)
{
    auto f32 = RunTiling({1, 4096, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    auto f16 = RunTiling({1, 4096, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT16);
    ASSERT_EQ(f32.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(f16.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(f32.tilingData.tileLen, 8192u);
    EXPECT_EQ(f16.tilingData.tileLen, 8192u);
    // tileLen stays aligned to the per-dtype 256B vector repeat (fp32:64, fp16:128)
    EXPECT_EQ(f32.tilingData.tileLen % 64u, 0u);
    EXPECT_EQ(f16.tilingData.tileLen % 128u, 0u);
}

// ############################################################################################
// Iteration-3: full op_host coverage — plug the branches iter-2 left open.
//   Constants mirrored from normalize_bbox_regbase_tiling.cpp:
//     UB_RESERVE=32*1024=32768, UB_BUF_FACTOR=6, VREPEAT_SIZE=256, MAX_TILE_LEN=8192.
//     blockAlignElems = VREPEAT_SIZE/dsize -> fp32:64, fp16:128 (also the tileLen floor clamp).
// New coverage vs iter-2:
//   - GetPlatformInfo guard (coreNum==0 / ubSize==0 -> GRAPH_FAILED)
//   - tileLen small-UB clamp: floor-aligned tileLen < blockAlign, and ubSize <= UB_RESERVE branch
//   - empty fast-path via batch==0 (the other half of `batch==0 || num==0`)
//   - reversed batch==1 large non-aligned num multi-core: genuine tail-clamp tiling params
//     (the cross-row residual scenario A1-Main confirmed safe at kernel level)
//   - extreme core distribution: alignment forces usedCoreNum < coreNum even when num==coreNum
//   - split-by-batch boundary batch==coreNum -> batchPerCore=1, tailBatchNum=0, all cores full
//   - explicit last-core residual for normal fp16 (alignFrames=4) tail-clamp boundary
//   - dtype guard extended to fp64 (DT_DOUBLE) and int64 (DT_INT64), host layer
// ############################################################################################

// ---- GetPlatformInfo guard: coreNum==0 -> GRAPH_FAILED --------------------------------------
TEST_F(NormalizeBBoxTiling, tiling_fail_platform_zero_corenum)
{
    auto r = RunTiling({2, 8, 4}, {2, 3}, /*reversedBox=*/false, ge::DT_FLOAT, /*coreNum=*/0, kUbSize);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// ---- GetPlatformInfo guard: ubSize==0 -> GRAPH_FAILED ---------------------------------------
TEST_F(NormalizeBBoxTiling, tiling_fail_platform_zero_ubsize)
{
    auto r = RunTiling({2, 8, 4}, {2, 3}, /*reversedBox=*/false, ge::DT_FLOAT, kCoreNum, /*ubSize=*/0);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// ---- tileLen small-UB clamp: ubSize just above UB_RESERVE, floor-aligned tileLen < blockAlign
// ubSize=UB_RESERVE+256=33024 -> budget=(33024-32768)/(6*dsize) is tiny -> floor-align to 0
// -> clamped up to blockAlignElems (fp32:64, fp16:128). Exercises `tileLen < blockAlignElems`.
TEST_F(NormalizeBBoxTiling, tiling_tilelen_small_ub_clamp_to_blockalign)
{
    constexpr uint64_t kUbReserve = 32u * 1024u;
    auto f32 = RunTiling({1, 4096, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT, kCoreNum, kUbReserve + 256u);
    auto f16 = RunTiling({1, 4096, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT16, kCoreNum, kUbReserve + 256u);
    ASSERT_EQ(f32.status, ge::GRAPH_SUCCESS);
    ASSERT_EQ(f16.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(f32.tilingData.tileLen, 64u);  // clamped to blockAlignElems (fp32)
    EXPECT_EQ(f16.tilingData.tileLen, 128u); // clamped to blockAlignElems (fp16)
}

// ---- tileLen small-UB clamp: ubSize <= UB_RESERVE -> budget branch skipped (tileLen stays 0)
// then clamped to blockAlignElems. Exercises the `ubSize_ > UB_RESERVE` false branch.
TEST_F(NormalizeBBoxTiling, tiling_tilelen_ub_below_reserve)
{
    auto f32 = RunTiling({1, 4096, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT, kCoreNum, /*ubSize=*/16384u);
    ASSERT_EQ(f32.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(f32.tilingDataValid);
    EXPECT_EQ(f32.tilingData.tileLen, 64u); // budget path skipped, clamped to blockAlignElems
    // tiling still succeeds with a small UB (GetPlatformInfo only rejects ubSize==0)
    EXPECT_EQ(f32.tilingData.splitMode, 0u);
}

// ---- empty fast-path via batch==0 (the batch==0 half of `batch==0 || num==0`) ---------------
// batch==0 -> splitMode=(batch>1)?1:0=0, usedCoreNum=1, split params zeroed.
TEST_F(NormalizeBBoxTiling, tiling_empty_tensor_batch0)
{
    auto r = RunTiling({0, 8, 4}, {0, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.batch, 0u);
    EXPECT_EQ(r.tilingData.splitMode, 0u); // batch==0 is not > 1
    EXPECT_EQ(r.tilingData.usedCoreNum, 1u);
    EXPECT_EQ(r.tilingData.numPerCore, 0u);
    EXPECT_EQ(r.tilingData.batchPerCore, 0u);
}

// ---- reversed batch==1 large non-aligned num, multi-core genuine tail-clamp -----------------
// reversed {1,4,1000}, alignFrames=8: perCoreRaw=ceil(1000/64)=16 -> numPerCore=ceil(16/8)*8=16
// -> usedCoreNum=ceil(1000/16)=63. num%numPerCore=1000%16=8 != 0 -> last core residual = 8 frames.
// This is the tiling-side of the cross-row 32B residual scenario (A1-Main confirmed kernel-safe).
TEST_F(NormalizeBBoxTiling, tiling_num_split_reversed_large_tail_clamp)
{
    auto r = RunTiling({1, 4, 1000}, {1, 3}, /*reversedBox=*/true, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.num, 1000u);      // reversed -> num from dim2
    EXPECT_EQ(r.tilingData.numPerCore, 16u); // aligned to reversed fp32 blockAlign (8) x 2
    EXPECT_EQ(r.tilingData.numPerCore % 8u, 0u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 63u);                  // < coreNum (64): alignment leaves 1 idle
    EXPECT_NE(r.tilingData.num % r.tilingData.numPerCore, 0u); // genuine tail
    // last-core residual = num - (used-1)*numPerCore, must be in (0, numPerCore]
    uint64_t residual = r.tilingData.num - (r.tilingData.usedCoreNum - 1u) * r.tilingData.numPerCore;
    EXPECT_EQ(residual, 8u);
    EXPECT_GT(residual, 0u);
    EXPECT_LE(residual, r.tilingData.numPerCore);
    EXPECT_EQ(r.tilingData.numBigCore, r.tilingData.usedCoreNum);
}

// ---- extreme core distribution: alignment forces usedCoreNum < coreNum even at num==coreNum --
// normal fp32 {1,64,4}, alignFrames=2: perCoreRaw=ceil(64/64)=1 -> numPerCore=ceil(1/2)*2=2
// -> usedCoreNum=ceil(64/2)=32. Only half the cores engage because each core needs >=1 32B block.
TEST_F(NormalizeBBoxTiling, tiling_num_split_alignment_reduces_corenum)
{
    auto r = RunTiling({1, 64, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 0u);
    EXPECT_EQ(r.tilingData.num, 64u);
    EXPECT_EQ(r.tilingData.numPerCore, 2u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 32u); // < kCoreNum (64) due to 32B alignment floor
    EXPECT_EQ(r.tilingData.usedCoreNum * r.tilingData.numPerCore, r.tilingData.num); // exact fit
}

// ---- split-by-batch boundary: batch==coreNum -> batchPerCore=1, tailBatchNum=0, all cores full
TEST_F(NormalizeBBoxTiling, tiling_batch_split_full_no_tail)
{
    auto r = RunTiling({64, 8, 4}, {64, 3}, /*reversedBox=*/false, ge::DT_FLOAT);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.splitMode, 1u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 64u); // min(64, 64)
    EXPECT_EQ(r.tilingData.batchPerCore, 1u);
    EXPECT_EQ(r.tilingData.bigCoreNum, 64u);  // batch - used*(perCore-1) = 64 - 64*0
    EXPECT_EQ(r.tilingData.tailBatchNum, 0u); // no small cores when batch divides evenly
    // load conservation with zero tail cores
    EXPECT_EQ(r.tilingData.bigCoreNum * r.tilingData.batchPerCore, r.tilingData.batch);
}

// ---- explicit tail-clamp residual for normal fp16 (alignFrames=4) ---------------------------
// {1,300,4} fp16, alignFrames=4: perCoreRaw=ceil(300/64)=5 -> numPerCore=ceil(5/4)*4=8
// -> usedCoreNum=ceil(300/8)=38. residual = 300 - 37*8 = 4 frames on the last core.
TEST_F(NormalizeBBoxTiling, tiling_num_split_normal_fp16_tail_clamp)
{
    auto r = RunTiling({1, 300, 4}, {1, 3}, /*reversedBox=*/false, ge::DT_FLOAT16);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    ASSERT_TRUE(r.tilingDataValid);
    EXPECT_EQ(r.tilingData.numPerCore, 8u);
    EXPECT_EQ(r.tilingData.numPerCore % 4u, 0u);
    EXPECT_EQ(r.tilingData.usedCoreNum, 38u);
    uint64_t residual = r.tilingData.num - (r.tilingData.usedCoreNum - 1u) * r.tilingData.numPerCore;
    EXPECT_EQ(residual, 4u);
    EXPECT_GT(residual, 0u);
    EXPECT_LT(residual, r.tilingData.numPerCore); // strictly shorter -> genuine clamp
}

// ---- dtype guard extended (host layer): fp64 unsupported -> GRAPH_FAILED ---------------------
TEST_F(NormalizeBBoxTiling, tiling_fail_dtype_fp64)
{
    auto r = RunTiling({2, 8, 4}, {2, 3}, /*reversedBox=*/false, ge::DT_DOUBLE);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

// ---- dtype guard extended (host layer): int64 unsupported -> GRAPH_FAILED --------------------
TEST_F(NormalizeBBoxTiling, tiling_fail_dtype_int64)
{
    auto r = RunTiling({2, 8, 4}, {2, 3}, /*reversedBox=*/false, ge::DT_INT64);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}
