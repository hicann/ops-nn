/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <gtest/gtest.h>
#include "norm/batch_norm_grad_v3/op_host/batch_norm_grad_v3_tiling.h"
#include "kernel_run_context_facker.h"
#include "op_host/tiling_util.h"
#include "test_cube_util.h"
#include "tests/ut/common/ut_op_util.h"

namespace {
constexpr uint64_t rarFullLoad = 10000000;
constexpr uint64_t rarSplitCoreR1 = 1000;
constexpr uint64_t rarRecomputeSplitR1 = 31000000;
constexpr uint64_t rarRecomputeSplitR0 = 32000000;
constexpr uint64_t defaultUbSize = 253952;

class BatchNormGradV3RarCapacity : public testing::TestWithParam<std::string> {
protected:
    template <typename Check>
    void WithTilingContext(const std::vector<int64_t>& dimensions, uint64_t ubSize, Check check)
    {
        SCOPED_TRACE(GetParam());
        gert::Shape shape;
        for (auto dim : dimensions) {
            shape.AppendDim(dim);
        }
        const int64_t channels = dimensions[1];
        constexpr auto format = ge::FORMAT_NCHW;
        gert::StorageShape tensorShape;
        tensorShape.MutableOriginShape() = shape;
        tensorShape.MutableStorageShape() = shape;
        gert::StorageShape channelShape = {{channels}, {channels}};
        std::string compileInfoJson = R"({"hardware_info":{"UB_SIZE":)" + std::to_string(ubSize) +
                                      R"(,"L2_SIZE":33554432,
            "L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072,"CORE_NUM":64,
            "vector_core_cnt":64,"socVersion":")" +
                                      GetParam() + R"("}})";
        std::map<std::string, std::string> socInfo, coreSpec, intrinsics, version;
        GetPlatFormInfos(compileInfoJson.c_str(), socInfo, coreSpec, intrinsics, version);
        fe::PlatFormInfos platform;
        ASSERT_TRUE(platform.Init());
        platform.SetPlatformRes("version", version);
        platform.SetPlatformRes("SoCInfo", socInfo);
        platform.SetPlatformRes("AICoreSpec", coreSpec);
        platform.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
        platform.SetCoreNumByCoreType("AICore");
        optiling::BatchNormGradV3CompileInfo compileInfo = {64, static_cast<int64_t>(ubSize), 32, 64};
        auto data = gert::TilingData::CreateCap(4096);
        auto workspace = gert::ContinuousVector::Create<size_t>(1);
        auto holder = gert::TilingContextFaker()
                          .SetOpType("BatchNormGradV3")
                          .NodeIoNum(7, 3)
                          .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                          .InputShapes({&tensorShape, &tensorShape, &channelShape, &channelShape, &channelShape,
                                        &channelShape, &channelShape})
                          .OutputShapes({&tensorShape, &channelShape, &channelShape})
                          .CompileInfo(&compileInfo)
                          .PlatformInfo(reinterpret_cast<char*>(&platform))
                          .NodeInputTd(0, ge::DT_FLOAT, format, format)
                          .NodeInputTd(1, ge::DT_FLOAT, format, format)
                          .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT, format, format)
                          .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({{"is_training", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                      {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-5)}})
                          .TilingData(data.get())
                          .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                          .Build();
        auto context = holder.GetContext<gert::TilingContext>();
        ASSERT_TRUE(Ops::NN::OpTiling::IsRegbaseSocVersion(context));
        check(context);
    }

    void ExpectTilingKey(const std::vector<int64_t>& dimensions, uint64_t expectedKey, uint64_t ubSize = defaultUbSize)
    {
        WithTilingContext(dimensions, ubSize, [expectedKey](gert::TilingContext* context) {
            auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchNormGradV3");
            ASSERT_NE(impl, nullptr);
            ASSERT_EQ(impl->tiling(context), ge::GRAPH_SUCCESS);
            EXPECT_EQ(context->GetTilingKey(), expectedKey);
        });
    }

    void ExpectRecomputeSplitR1Factor(const std::vector<int64_t>& dimensions, int64_t expectedFactor)
    {
        WithTilingContext(dimensions, defaultUbSize, [expectedFactor](gert::TilingContext* context) {
            // Run the recompute template directly, since the full-load template can win at this boundary.
            optiling::BatchNormGradV3RARRecomputeTilingBase recompute(context);
            ASSERT_EQ(recompute.DoTiling(), ge::GRAPH_SUCCESS);
            ASSERT_EQ(context->GetTilingKey(), rarRecomputeSplitR1);

            const optiling::BatchNormGradV3BaseTilingData baseData;
            const optiling::BatchNormGradV3BinaryAddTilingData binaryAddData;
            const size_t factorOffset = baseData.GetDataSize() + 2 * binaryAddData.GetDataSize();
            const auto* raw = context->GetRawTilingData();
            int64_t factor = 0;
            ASSERT_GE(raw->GetDataSize(), factorOffset + sizeof(factor));
            std::memcpy(&factor, static_cast<const uint8_t*>(raw->GetData()) + factorOffset, sizeof(factor));
            EXPECT_EQ(factor, expectedFactor);
        });
    }
};

INSTANTIATE_TEST_SUITE_P(TargetPlatforms, BatchNormGradV3RarCapacity,
                         testing::Values("Ascend950", "Ascend960DT", "Ascend960PR"));

TEST_P(BatchNormGradV3RarCapacity, SelectsTemplateByReductionCapacity)
{
    ExpectTilingKey({2, 8, 2, 2}, rarFullLoad);
    ExpectTilingKey({64, 8, 1024, 1}, rarSplitCoreR1);
    ExpectTilingKey({32, 32, 1024, 1}, rarRecomputeSplitR1);
    ExpectTilingKey({2, 32, 10000, 1}, rarRecomputeSplitR0);
}

TEST_P(BatchNormGradV3RarCapacity, CrossCoreCapacityUsesStrictBoundary)
{
    // R1 * R0 * 5 is less than UB at R0=799, and equals UB at R0=800.
    constexpr uint64_t ubSize = 256000;
    ExpectTilingKey({64, 8, 799, 1}, rarRecomputeSplitR1, ubSize);
    ExpectTilingKey({64, 8, 800, 1}, rarSplitCoreR1, ubSize);
    ExpectTilingKey({64, 8, 801, 1}, rarSplitCoreR1, ubSize);
}

TEST_P(BatchNormGradV3RarCapacity, R0CapacityIncludesExactFit)
{
    // The aligned scratch budget is 61,440 bytes (7,680 FP32 elements with two buffers).
    ExpectTilingKey({8, 32, 7680, 1}, rarRecomputeSplitR1);
    ExpectTilingKey({8, 32, 7681, 1}, rarRecomputeSplitR0);
}

TEST_P(BatchNormGradV3RarCapacity, SplitR1ReductionCapacityIncludesExactFit)
{
    // 3 * 5,120 FP32 values fill the scratch budget. One more element per R0 row requires splitting R1.
    ExpectRecomputeSplitR1Factor({3, 32, 5120, 1}, 15360);
    ExpectRecomputeSplitR1Factor({3, 32, 5121, 1}, 10242);
}
} // namespace
