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
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/foreach_log_tiling_data.h"

using namespace ut_util;

class ForeachLogTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachLogTilingArch35 SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachLogTilingArch35 TearDown" << std::endl; }
};

static void DoCase(std::initializer_list<int64_t> xShape, ge::DataType dt, uint64_t expectKey)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc;
    std::map<std::string, std::string> ai;
    std::map<std::string, std::string> intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachLogCompileInfo {
    } ci;
    auto opType = std::string("ForeachLog");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xS = {xShape, xShape};
    gert::StorageShape yS = {xShape, xShape};
    auto param = gert::TilingData::CreateCap(8192);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(1, 1)
                  .IrInstanceNum({1})
                  .InputShapes({&xS})
                  .OutputShapes({&yS})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    EXPECT_EQ(tilingFn(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetTilingKey(), expectKey);
    auto* tilingData = ctx->GetTilingData<ForeachLogTilingData>();
    EXPECT_GT(tilingData->tileElems, 0U);
    EXPECT_EQ(tilingData->tensorCount, 1);
    int64_t shapeSize = 1;
    for (const int64_t dim : xShape) {
        shapeSize *= dim;
    }
    EXPECT_EQ(tilingData->tensorDataCountList[0], shapeSize);
    EXPECT_EQ(tilingData->tensorStartList[0], 0);
    EXPECT_EQ(tilingData->tensorStartOffsetList[0], 0);
    EXPECT_EQ(tilingData->tensorEndList[0], 0);
}

namespace {
constexpr int64_t MOCK_CORE_NUM = 64;
constexpr int64_t UT_MAX_CORE_COUNT = 80;
constexpr int64_t UT_TENSOR_BALANCED_AVG = 32768;
constexpr int64_t UT_MIN_CORE_WORK_BYTES = 4096;

int64_t CeilDivExpect(int64_t a, int64_t b) { return (a + b - 1) / b; }

struct CoreRangeExpect {
    uint16_t startTensor = 0;
    uint16_t endTensor = 0;
    int32_t startOffset = 0;
    int32_t endOffset = -1;
};

void LocateFlatOffsetExpect(int64_t flatOffset, const std::vector<int64_t>& numels, uint16_t& tensorIndex,
                            int64_t& tensorOffset)
{
    for (size_t index = 0; index < numels.size(); ++index) {
        if (flatOffset < numels[index]) {
            tensorIndex = static_cast<uint16_t>(index);
            tensorOffset = flatOffset;
            return;
        }
        flatOffset -= numels[index];
    }
    tensorIndex = static_cast<uint16_t>(numels.size() - 1);
    tensorOffset = numels[tensorIndex] - 1;
}

// Independently derive the dual-split contract from the tiling header constants, mirroring
// how the cosh UT couples its expectations to the mocked platform instead of the output.
std::vector<CoreRangeExpect> ExpectDualSplit(const std::vector<int64_t>& numels, uint32_t storageBytes)
{
    const int64_t tensorCount = static_cast<int64_t>(numels.size());
    int64_t totalDataCount = 0;
    int64_t maxDataCount = 0;
    for (int64_t numel : numels) {
        totalDataCount += numel;
        maxDataCount = std::max(maxDataCount, numel);
    }
    const int64_t limitedCoreNum = std::min<int64_t>(MOCK_CORE_NUM, UT_MAX_CORE_COUNT);
    std::vector<CoreRangeExpect> ranges;
    if (totalDataCount / tensorCount <= UT_TENSOR_BALANCED_AVG && maxDataCount <= UT_TENSOR_BALANCED_AVG) {
        const int64_t coreCount = std::min<int64_t>(limitedCoreNum, tensorCount);
        const int64_t baseTensors = tensorCount / coreCount;
        const int64_t remainderTensors = tensorCount % coreCount;
        int32_t tensorCursor = 0;
        for (int64_t coreIndex = 0; coreIndex < coreCount; ++coreIndex) {
            const int64_t share = baseTensors + static_cast<int64_t>(coreIndex < remainderTensors);
            const int32_t startTensor = tensorCursor;
            const int32_t endTensor = tensorCursor + static_cast<int32_t>(share) - 1;
            ranges.push_back({static_cast<uint16_t>(startTensor), static_cast<uint16_t>(endTensor), 0,
                              static_cast<int32_t>(numels[static_cast<size_t>(endTensor)] - 1)});
            tensorCursor = endTensor + 1;
        }
    } else {
        const int64_t elementsPerBlock = UT_MIN_CORE_WORK_BYTES / static_cast<int64_t>(storageBytes);
        const int64_t coreCount = std::max<int64_t>(
            1, std::min<int64_t>(limitedCoreNum, CeilDivExpect(totalDataCount, elementsPerBlock)));
        const int64_t totalBlocks = CeilDivExpect(totalDataCount, elementsPerBlock);
        const int64_t baseBlocks = totalBlocks / coreCount;
        const int64_t remainderBlocks = totalBlocks % coreCount;
        int64_t flatCursor = 0;
        for (int64_t coreIndex = 0; coreIndex < coreCount; ++coreIndex) {
            const int64_t blockCount = baseBlocks + static_cast<int64_t>(coreIndex < remainderBlocks);
            int64_t currentCount = totalDataCount - flatCursor;
            if (blockCount <= currentCount / elementsPerBlock) {
                currentCount = blockCount * elementsPerBlock;
            }
            uint16_t startTensor = 0;
            int64_t startOffset = 0;
            uint16_t endTensor = 0;
            int64_t endOffset = -1;
            LocateFlatOffsetExpect(flatCursor, numels, startTensor, startOffset);
            LocateFlatOffsetExpect(flatCursor + currentCount - 1, numels, endTensor, endOffset);
            ranges.push_back(
                {startTensor, endTensor, static_cast<int32_t>(startOffset), static_cast<int32_t>(endOffset)});
            flatCursor += currentCount;
        }
    }
    return ranges;
}
} // namespace

// list-shaped tiling: assert the full multi-core split fields (tensorStart/tensorEnd and
// both offsets) against the dual-split contract, not just the single-tensor core-0 slice
static void DoListCase(const std::vector<int64_t>& tensorNumels, ge::DataType dt)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc;
    std::map<std::string, std::string> ai;
    std::map<std::string, std::string> intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachLogCompileInfo {
    } ci;
    auto opType = std::string("ForeachLog");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    const uint32_t tensorNum = static_cast<uint32_t>(tensorNumels.size());
    std::vector<gert::StorageShape> inputShapes(tensorNum);
    std::vector<gert::StorageShape> outputShapes(tensorNum);
    for (uint32_t i = 0; i < tensorNum; i++) {
        inputShapes[i] = {{tensorNumels[i]}, {tensorNumels[i]}};
        outputShapes[i] = {{tensorNumels[i]}, {tensorNumels[i]}};
    }
    std::vector<gert::StorageShape*> inputShapeRefs;
    std::vector<gert::StorageShape*> outputShapeRefs;
    for (auto& shape : inputShapes) {
        inputShapeRefs.push_back(&shape);
    }
    for (auto& shape : outputShapes) {
        outputShapeRefs.push_back(&shape);
    }

    auto param = gert::TilingData::CreateCap(8192);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(1, 1)
                  .IrInstanceNum({tensorNum}, {tensorNum})
                  .InputShapes(inputShapeRefs)
                  .OutputShapes(outputShapeRefs)
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);

    EXPECT_EQ(tilingFn(ctx), ge::GRAPH_SUCCESS);
    auto* tilingData = ctx->GetTilingData<ForeachLogTilingData>();
    ASSERT_NE(tilingData, nullptr);
    EXPECT_GT(tilingData->tileElems, 0U);
    EXPECT_EQ(tilingData->tensorCount, static_cast<int32_t>(tensorNum));
    for (uint32_t i = 0; i < tensorNum; i++) {
        EXPECT_EQ(tilingData->tensorDataCountList[i], tensorNumels[i]);
    }

    const uint32_t storageBytes = (dt == ge::DT_FLOAT) ? 4U : 2U;
    const auto expectRanges = ExpectDualSplit(tensorNumels, storageBytes);
    for (size_t coreIndex = 0; coreIndex < expectRanges.size(); coreIndex++) {
        EXPECT_EQ(tilingData->tensorStartList[coreIndex], expectRanges[coreIndex].startTensor);
        EXPECT_EQ(tilingData->tensorEndList[coreIndex], expectRanges[coreIndex].endTensor);
        EXPECT_EQ(tilingData->tensorStartOffsetList[coreIndex], expectRanges[coreIndex].startOffset);
        EXPECT_EQ(tilingData->tensorEndOffsetList[coreIndex], expectRanges[coreIndex].endOffset);
    }
}

// Two small tensors: the tensor-balanced branch gives each core whole tensors in order.
TEST_F(ForeachLogTilingArch35, foreach_log_list_two_tensors_two_cores) { DoListCase({5, 3}, ge::DT_FLOAT); }

// A 0-element tensor inside the list owns a core with an empty range (endOffset == -1);
// the kernel skips it without resolving a data pointer.
TEST_F(ForeachLogTilingArch35, foreach_log_list_with_empty_tensor) { DoListCase({4, 0, 6}, ge::DT_FLOAT); }

// Tensors exactly at the TENSOR_BALANCED_AVG boundary keep the tensor-balanced split.
TEST_F(ForeachLogTilingArch35, foreach_log_list_at_avg_boundary) { DoListCase({32768, 32768}, ge::DT_FLOAT); }

// Skewed list: the average stays below the threshold but one tensor far exceeds it, so
// the split must fall back to element-balanced instead of pinning the huge tensor to a
// single core while the other 63 cores idle.
TEST_F(ForeachLogTilingArch35, foreach_log_skewed_list_falls_back_to_element_balanced)
{
    std::vector<int64_t> numels(255, 100);
    numels.push_back(8300000);
    DoListCase(numels, ge::DT_FLOAT);
}

// Small single tensor: average below TENSOR_BALANCED_AVG, tensor-balanced
// split (a single core covers the whole list).
TEST_F(ForeachLogTilingArch35, foreach_log_float32) { DoCase({4, 4}, ge::DT_FLOAT, 0); }

TEST_F(ForeachLogTilingArch35, foreach_log_float16) { DoCase({4, 4}, ge::DT_FLOAT16, 0); }

TEST_F(ForeachLogTilingArch35, foreach_log_bfloat16) { DoCase({4, 4}, ge::DT_BF16, 0); }

// One tensor larger than the tensor-balanced average: element-balanced split;
// core 0 covers only the first block range.
TEST_F(ForeachLogTilingArch35, foreach_log_float32_large_flow)
{
    DoCase({524288}, ge::DT_FLOAT, 0);
    // 64 cores, elementsPerBlock=1024: 512 blocks / 64 cores = 8 blocks/core.
}

// NOTE: the 0-instance empty-list no-op path (ForeachLogEmptyListNoOp) cannot be
// driven through this faker: TilingContextFaker hangs during Build() for a list with
// zero instances, so that path stays covered by code review only.

// A single tensor whose element count exceeds the int32 offset limit must be
// rejected before any narrowing write into the tiling fields.
TEST_F(ForeachLogTilingArch35, foreach_log_float32_int32_overflow_rejected)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc;
    std::map<std::string, std::string> ai;
    std::map<std::string, std::string> intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachLogCompileInfo {
    } ci;
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachLog")->tiling;

    gert::StorageShape xS = {{2147483648LL}, {2147483648LL}};
    gert::StorageShape yS = {{2147483648LL}, {2147483648LL}};
    auto param = gert::TilingData::CreateCap(8192);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(1, 1)
                  .IrInstanceNum({1})
                  .InputShapes({&xS})
                  .OutputShapes({&yS})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    EXPECT_NE(tilingFn(ctx), ge::GRAPH_SUCCESS);
}
