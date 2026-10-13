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
#include <limits>
#include <utility>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "tiling_case_executor.h"
#include "../../../op_kernel/max_pool2d_with_mask_tiling_data.h"

namespace {
#ifndef MAX_POOL2D_WITH_MASK_TEST_SOC
#define MAX_POOL2D_WITH_MASK_TEST_SOC ""
#endif

constexpr bool IS_ASCEND_310P_TEST = std::string_view(MAX_POOL2D_WITH_MASK_TEST_SOC) == "ascend310p";

struct MaxPool2dWithMaskCompileInfo {};
MaxPool2dWithMaskCompileInfo g_compileInfo;

gert::TilingContextPara MakeTilingContext(ge::DataType dtype, uint64_t ubBytes = 196608)
{
    return gert::TilingContextPara("MaxPool2dWithMask", {{{{2, 3, 8, 9}, {2, 3, 8, 9}}, dtype, ge::FORMAT_ND}},
                                   {{{{2, 3, 4, 5}, {2, 3, 4, 5}}, dtype, ge::FORMAT_ND},
                                    {{{2, 3, 9, 96}, {2, 3, 9, 96}}, ge::DT_INT8, ge::FORMAT_ND}},
                                   {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{3, 3})},
                                    {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                                    {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                                   &g_compileInfo, 40, ubBytes, 4096);
}

gert::TilingContextPara MakeCacheLinePartitionContext(ge::DataType dtype = ge::DT_FLOAT16, uint64_t ubBytes = 196608,
                                                      int64_t channels = 3)
{
    return gert::TilingContextPara("MaxPool2dWithMask",
                                   {{{{channels, 32, 32}, {channels, 32, 32}}, dtype, ge::FORMAT_ND}},
                                   {{{{channels, 16, 16}, {channels, 16, 16}}, dtype, ge::FORMAT_ND},
                                    {{{channels, 9, 544}, {channels, 9, 544}}, ge::DT_INT8, ge::FORMAT_ND}},
                                   {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{3, 3})},
                                    {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                                    {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                                   &g_compileInfo, 40, ubBytes, 4096);
}

TEST(MaxPool2dWithMaskTiling, PackedIntervalsRequireExactWindowsAndBudget)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Packed interval pooling is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        for (const int64_t channels : {64, 512, 1024, 16384}) {
            auto context = gert::TilingContextPara(
                "MaxPool2dWithMask", {{{{1, channels, 7, 7}, {1, channels, 7, 7}}, dtype, ge::FORMAT_ND}},
                {{{{1, channels, 4, 4}, {1, channels, 4, 4}}, dtype, ge::FORMAT_ND},
                 {{{1, channels, 49, 64}, {1, channels, 49, 64}}, ge::DT_INT8, ge::FORMAT_ND}},
                {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{7, 7})},
                 {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                 {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{3, 3})},
                 {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                 {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                &g_compileInfo, 8, 262144, 4096);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(context, info));
            const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
            if (channels == 64) {
                EXPECT_NE(info.tilingKey, 46U);
                EXPECT_NE(info.tilingKey, 47U);
            } else {
                EXPECT_EQ(info.tilingKey, data->m200NcFactor == 64 ? 46U : 47U);
                const uint64_t nc = data->m200NcFactor;
                const uint64_t bytes = 181 * nc * (dtype == ge::DT_FLOAT ? 4 : 2) + 172 * nc * 4 + nc + 8192;
                EXPECT_LE(bytes, 248 * 1024);
            }
        }
    }
}

TEST(MaxPool2dWithMaskTiling, MapTilesUseIdleCoresWithoutChangingOtherPaths)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Spatial map tasks are specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        for (const int64_t channels : {1, 3, 7, 8, 9}) {
            for (const int64_t kernel : {3, 7}) {
                const int64_t stride = kernel == 3 ? 1 : 2;
                const int64_t width = 224 / stride;
                const int64_t pitch = ((width * width + 15) / 16 + 1) * 32;
                auto context = gert::TilingContextPara(
                    "MaxPool2dWithMask", {{{{1, channels, 224, 224}, {1, channels, 224, 224}}, dtype, ge::FORMAT_ND}},
                    {{{{1, channels, width, width}, {1, channels, width, width}}, dtype, ge::FORMAT_ND},
                     {{{1, channels, kernel * kernel, pitch}, {1, channels, kernel * kernel, pitch}},
                      ge::DT_INT8,
                      ge::FORMAT_ND}},
                    {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{kernel, kernel})},
                     {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{stride, stride})},
                     {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{kernel / 2, kernel / 2})},
                     {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                     {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                    &g_compileInfo, 8, 262144, 4096);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(context, info));
                const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
                ASSERT_GT(data->m200MapRows, 0);
                EXPECT_EQ(info.tilingKey, channels < 8 ? (dtype == ge::DT_FLOAT16 ? 49U : 45U) : 11U);
                EXPECT_EQ(data->m200MapRows * width * (dtype == ge::DT_FLOAT ? 4 : 2) % 64, 0);
            }
        }
    }
}

TEST(MaxPool2dWithMaskTiling, SpatialSeparableUsesAlignedPlanesAndFitsSdkScratchBoundary)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Spatial separable pooling is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        for (const int64_t width : {80, 112, 224}) {
            const int64_t outWidth = width / 2;
            const int64_t maskPitch = ((outWidth * outWidth + 15) / 16 + 1) * 32;
            auto context = gert::TilingContextPara(
                "MaxPool2dWithMask", {{{{64, width, width}, {64, width, width}}, dtype, ge::FORMAT_ND}},
                {{{{64, outWidth, outWidth}, {64, outWidth, outWidth}}, dtype, ge::FORMAT_ND},
                 {{{64, 9, maskPitch}, {64, 9, maskPitch}}, ge::DT_INT8, ge::FORMAT_ND}},
                {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{3, 3})},
                 {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                 {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                 {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                 {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                &g_compileInfo, 8, 262144, 4096);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(context, info));
            if (dtype != ge::DT_FLOAT || width == 224) {
                // This candidate requires FP32 and a complete plane in UB.
                EXPECT_NE(info.tilingKey, 40U);
                continue;
            }
            const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
            EXPECT_EQ(info.tilingKey, width == 112 ? 40U : 37U);
            EXPECT_EQ(data->m200SpatialBatch, 1);
            EXPECT_EQ(data->m200NcRows, 0);
            EXPECT_EQ(data->m200SpatialRows, outWidth);
            const int64_t alignment = width == 112 ? 64 : 256;
            const int64_t h = (((2 * data->m200SpatialRows + 1) * outWidth + alignment - 1) / alignment) * alignment;
            const int64_t o = ((data->m200SpatialRows * outWidth + alignment - 1) / alignment) * alignment;
            const int64_t bytesPerValue = dtype == ge::DT_FLOAT ? 4 : 2;
            const int64_t bytes = h * (6 * bytesPerValue + 8) + o * 4 + 2 * ((h / 8 + 31) / 32 * 32) +
                                  (h * 4 > 8192 ? h * 4 : 8192);
            if (width == 112) {
                EXPECT_EQ(h, 6336);
                EXPECT_EQ(o, 3136);
                EXPECT_EQ(bytes, 242240);
            }
            EXPECT_LE(bytes, 248 * 1024);
            EXPECT_LE(o * (bytesPerValue + 8), width * width * bytesPerValue);
            EXPECT_LE(2 * h * bytesPerValue / 256, 255);
        }
    }
}

gert::TilingContextPara MakeHorizontalCacheContext(ge::DataType dtype, int64_t height, int64_t width,
                                                   bool ceilMode = false, int64_t channels = 64,
                                                   uint64_t ubBytes = 262144)
{
    const int64_t outH = (height + 1 + (ceilMode ? 1 : 0)) / 2;
    const int64_t outW = (width + 1 + (ceilMode ? 1 : 0)) / 2;
    const int64_t maskPitch = ((outH * outW + 15) / 16 + 1) * 32;
    return gert::TilingContextPara(
        "MaxPool2dWithMask", {{{{1, channels, height, width}, {1, channels, height, width}}, dtype, ge::FORMAT_ND}},
        {{{{1, channels, outH, outW}, {1, channels, outH, outW}}, dtype, ge::FORMAT_ND},
         {{{1, channels, 9, maskPitch}, {1, channels, 9, maskPitch}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{3, 3})},
         {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
         {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
         {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
         {"ceilMode", Ops::NN::AnyValue::CreateFrom(ceilMode)}},
        &g_compileInfo, 8, ubBytes, 4096);
}

TEST(MaxPool2dWithMaskTiling, SpatialPlaneDispatchPreservesHorizontalCacheFallbacks)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Horizontal row caching is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeHorizontalCacheContext(dtype, 112, 112);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(info.tilingKey, dtype == ge::DT_FLOAT ? 40U : 30U);
        EXPECT_EQ(data->m200NcRows, dtype == ge::DT_FLOAT ? 0 : 6);
        EXPECT_EQ(data->m200NcFactor, 16);
    }
    auto context = MakeHorizontalCacheContext(ge::DT_FLOAT, 224, 224);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 30U);
    EXPECT_EQ(data->m200NcRows, 2);
}

TEST(MaxPool2dWithMaskTiling, HorizontalCacheRequiresSameCoreAdjacentTileReuse)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Horizontal row caching is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeHorizontalCacheContext(dtype, 64, 64, false, 16);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(info.tilingKey, 24U);
        EXPECT_EQ(data->m200NcRows, 4);
        EXPECT_EQ(data->m200NcFactor, 16);
    }
}

TEST(MaxPool2dWithMaskTiling, HorizontalCacheFallsBackWithoutShrinkingRows)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Horizontal row caching is specific to Ascend 310P";
    }
    // The existing 16-row tile fits, but adding its cached value/index row does not.
    auto context = MakeHorizontalCacheContext(ge::DT_FLOAT16, 56, 56);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 24U);
    EXPECT_EQ(data->m200NcRows, 16);
}

TEST(MaxPool2dWithMaskTiling, HorizontalCacheAcceptsCeilTailWithNoFreshInputRow)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Horizontal row caching is specific to Ascend 310P";
    }
    auto context = MakeHorizontalCacheContext(ge::DT_FLOAT, 112, 111, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 30U);
    EXPECT_EQ(data->outputHeight, 57);
    EXPECT_EQ(data->outputWidth, 56);
    EXPECT_EQ(data->m200NcRows, 4);
    EXPECT_EQ((data->outputHeight - 1) * data->strideH - data->padH + 1, data->inputHeight);
}

TEST(MaxPool2dWithMaskTiling, HalfWindowOffsetsRequireExactIntegerRange)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Half window offsets are specific to Ascend 310P";
    }
    // A synthetic large budget isolates the numerical guard from UB tiling.
    // This is not a claim that a physical 310P has 1MiB of UB.
    for (const int64_t width : {1023, 1024}) {
        auto context = MakeHorizontalCacheContext(ge::DT_FLOAT16, 128, width, false, 64, 1024 * 1024);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey, width == 1023 ? 30U : 28U);
    }
}

gert::TilingContextPara MakeSpatialContext(ge::DataType dtype, int64_t height, int64_t width, int64_t channels,
                                           bool ceilMode = false)
{
    const int64_t outH = (height + (ceilMode ? 1 : 0)) / 2;
    const int64_t outW = (width + (ceilMode ? 1 : 0)) / 2;
    const int64_t maskPitch = ((outH * outW + 15) / 16 + 1) * 32;
    return gert::TilingContextPara("MaxPool2dWithMask",
                                   {{{{channels, height, width}, {channels, height, width}}, dtype, ge::FORMAT_ND}},
                                   {{{{channels, outH, outW}, {channels, outH, outW}}, dtype, ge::FORMAT_ND},
                                    {{{channels, 4, maskPitch}, {channels, 4, maskPitch}}, ge::DT_INT8, ge::FORMAT_ND}},
                                   {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                                    {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                                    {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0})},
                                    {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"ceilMode", Ops::NN::AnyValue::CreateFrom(ceilMode)}},
                                   &g_compileInfo, 8, 262144, 4096);
}

gert::TilingContextPara MakeSmallNcTransposeContext(ge::DataType dtype, int64_t height, int64_t width,
                                                    uint64_t ubBytes = 262144)
{
    const int64_t outH = (height + 1) / 2;
    const int64_t outW = (width + 1) / 2;
    const int64_t maskPitch = ((outH * outW + 15) / 16 + 1) * 32;
    return gert::TilingContextPara("MaxPool2dWithMask",
                                   {{{{4, 4, height, width}, {4, 4, height, width}}, dtype, ge::FORMAT_ND}},
                                   {{{{4, 4, outH, outW}, {4, 4, outH, outW}}, dtype, ge::FORMAT_ND},
                                    {{{4, 4, 49, maskPitch}, {4, 4, 49, maskPitch}}, ge::DT_INT8, ge::FORMAT_ND}},
                                   {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{7, 7})},
                                    {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                                    {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{3, 3})},
                                    {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                                   &g_compileInfo, 8, ubBytes, 4096);
}

gert::TilingContextPara MakeFlatContext(ge::DataType dtype, int64_t side)
{
    const int64_t maskPitch = ((side * side + 15) / 16 + 1) * 32;
    return gert::TilingContextPara("MaxPool2dWithMask",
                                   {{{{1, 64, side, side}, {1, 64, side, side}}, dtype, ge::FORMAT_ND}},
                                   {{{{1, 64, side, side}, {1, 64, side, side}}, dtype, ge::FORMAT_ND},
                                    {{{1, 64, 9, maskPitch}, {1, 64, 9, maskPitch}}, ge::DT_INT8, ge::FORMAT_ND}},
                                   {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{3, 3})},
                                    {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                                    {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                                   &g_compileInfo, 8, 262144, 4096);
}

TEST(MaxPool2dWithMaskTiling, FlatReuseAllowsMoreRowsWithoutCandidateStorage)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Flat buffer reuse is specific to Ascend 310P";
    }
    for (const int64_t width : {56, 112, 224}) {
        auto context = MakeFlatContext(ge::DT_FLOAT, width);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(info.tilingKey, 48U);
        EXPECT_EQ(data->m200NcRows, width == 56 ? 8 : (width == 112 ? 4 : 2));
        EXPECT_EQ(data->m200NcFactor, 16);
    }
}

TEST(MaxPool2dWithMaskTiling, FlatReuseAvoidsUnusedRowsWithoutAddingTiles)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Flat buffer reuse is specific to Ascend 310P";
    }
    auto context = MakeFlatContext(ge::DT_FLOAT, 56);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->m200NcRows, 8);
    EXPECT_EQ((data->outputHeight + data->m200NcRows - 1) / data->m200NcRows, (56 + 9 - 1) / 9);
    EXPECT_EQ(data->outputHeight % data->m200NcRows, 0);
    EXPECT_EQ(data->m200NcRows * data->outputWidth * 4 % 32, 0);
}

TEST(MaxPool2dWithMaskTiling, FlatHalfRetainsCompactionWhenTransposeIsTooSmall)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Flat buffer reuse is specific to Ascend 310P";
    }
    auto context = MakeFlatContext(ge::DT_FLOAT16, 112);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 27U);
    EXPECT_EQ(data->m200NcRows, 5);
    const int64_t inputSpace = (((data->m200NcRows + 2) * 112 + 32 + 15) / 16) * 16;
    EXPECT_LT(inputSpace * 2, data->m200NcRows * 112 * 4);
}

TEST(MaxPool2dWithMaskTiling, FlatReuseDoesNotChangePackedAllocation)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Flat buffer reuse is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeFlatContext(dtype, 7);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_NE(info.tilingKey, 27U);
    }
}

TEST(MaxPool2dWithMaskTiling, LargerUbDoesNotDisableSmallNcTranspose)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "The channel transpose is specific to Ascend 310P";
    }
    // F115: aliasing allows seven rows, but that produces only five tasks.
    // Four rows still fit and supply eight tasks without changing the kernel.
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        for (const uint64_t ubBytes : {196608U, 262144U}) {
            auto context = MakeSmallNcTransposeContext(dtype, 64, 64, ubBytes);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(context, info));
            const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
            EXPECT_EQ(info.tilingKey, 24U);
            EXPECT_EQ(data->m200NcRows, 4);
            EXPECT_EQ(data->m200NcFactor, 16);
            EXPECT_EQ(data->m200MapRows, 0);
        }
    }
}

TEST(MaxPool2dWithMaskTiling, SmallNcTransposeKeepsRowAlignment)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "The channel transpose is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeSmallNcTransposeContext(dtype, 64, 56);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(info.tilingKey, 24U);
        EXPECT_EQ(data->m200NcRows, 4);
        const int64_t bytes = dtype == ge::DT_FLOAT ? 4 : 2;
        EXPECT_EQ(data->m200NcRows * data->outputWidth * bytes % 32, 0);
    }
}

TEST(MaxPool2dWithMaskTiling, DoesNotSplitPackedPlaneToIncreaseTaskCount)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "The channel transpose is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeSmallNcTransposeContext(dtype, 7, 7);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(data->m200NcRows, 0);
        EXPECT_NE(info.tilingKey, 12U);
        EXPECT_NE(info.tilingKey, 16U);
    }
}

TEST(MaxPool2dWithMaskTiling, SeparableReuseFallsBackWhenSavedMasksDoNotFit)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "The channel transpose is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeSmallNcTransposeContext(dtype, 16, 64);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        ASSERT_EQ(data->m200NcRows, 1);
        EXPECT_EQ(info.tilingKey, 12U);
        // Seven horizontal rows need 6 * (7 * 32 * 16 / 8) mask bytes,
        // exceeding the one-row output index buffer (32 * 16 * 4 bytes).
        EXPECT_GT(6 * (7 * 32 * 16 / 8), 32 * 16 * 4);
    }
}

TEST(MaxPool2dWithMaskTiling, SpatialBatchAllowsPartialFinalNcGroup)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Spatial GatherMask is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeSpatialContext(dtype, 64, 64, 72);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(info.tilingKey, dtype == ge::DT_FLOAT ? 23U : 15U);
        EXPECT_EQ(data->m200SpatialRows, 32);
        EXPECT_EQ(data->slicesPerCore, 9);
        EXPECT_GT(data->m200SpatialBatch, 1);
        EXPECT_NE(data->slicesPerCore % data->m200SpatialBatch, 0);
    }
}

TEST(MaxPool2dWithMaskTiling, SpatialBatchDoesNotCrossUnusedInputRows)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Spatial GatherMask is specific to Ascend 310P";
    }
    auto context = MakeSpatialContext(ge::DT_FLOAT, 65, 64, 72);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 25U);
    EXPECT_EQ(data->m200SpatialRows, 32);
    EXPECT_EQ(data->m200SpatialBatch, 1);
}

TEST(MaxPool2dWithMaskTiling, SpatialInputPipelineKeepsTileAndChecksCapacity)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Spatial input pipeline is specific to Ascend 310P";
    }
    for (const int64_t width : {112, 224}) {
        auto context = MakeSpatialContext(ge::DT_FLOAT, 112, width, 256);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(data->m200SpatialRows, width == 112 ? 56 : 38);
        EXPECT_EQ(data->m200SpatialBatch, 1);
        EXPECT_EQ(info.tilingKey, width == 112 ? 25U : 38U);
    }
    auto context = MakeSpatialContext(ge::DT_FLOAT, 112, 112, 8);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, 15U); // One slice per core has no next input.
}

TEST(MaxPool2dWithMaskTiling, SpatialIoRetilePreservesRoundedVectorWork)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Spatial input-output pipeline is specific to Ascend 310P";
    }
    auto context = MakeSpatialContext(ge::DT_FLOAT, 224, 224, 128);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 44U);
    EXPECT_EQ(data->m200SpatialRows, 56);
    EXPECT_EQ(data->m200SpatialBatch, 1);
    const uint64_t capacity = ((data->m200SpatialRows * data->outputWidth + 63) / 64) * 64;
    const uint64_t bytes = capacity * 40 + ((capacity / 8 + 31) / 32) * 32;
    EXPECT_EQ(bytes, 251680U);
    EXPECT_LT(bytes, 248U * 1024);
    EXPECT_EQ(2 * capacity, 12544U);
}

TEST(MaxPool2dWithMaskTiling, SpatialIoRetileKeepsSingleSliceFallback)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Spatial input-output pipeline is specific to Ascend 310P";
    }
    auto context = MakeSpatialContext(ge::DT_FLOAT, 224, 224, 8);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 15U);
    EXPECT_EQ(data->m200SpatialRows, 52);
}

TEST(MaxPool2dWithMaskTiling, SpatialHalfInputAllowsPartialFinalCore)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Independent row halves are specific to Ascend 310P";
    }
    for (int64_t channels : {65, 1024}) {
        auto context = MakeSpatialContext(ge::DT_FLOAT, 224, 224, channels);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(info.tilingKey, 44U);
        EXPECT_EQ(data->m200SpatialRows, 56);
        EXPECT_EQ(data->m200SpatialBatch, 1);
    }
}

TEST(MaxPool2dWithMaskTiling, SpatialHalfInputKeepsHalfPrecisionRowLoad)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Independent row halves are specific to Ascend 310P";
    }
    auto context = MakeSpatialContext(ge::DT_FLOAT16, 224, 224, 128);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, 38U);
}

TEST(MaxPool2dWithMaskTiling, SpatialPathRejectsCeilOverrunAndUnalignedRows)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Spatial GatherMask is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeSpatialContext(dtype, 33, 64, 72, true);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(data->m200SpatialRows, 0);
        EXPECT_NE(info.tilingKey, 15U);
    }
    auto context = MakeSpatialContext(ge::DT_FLOAT16, 56, 56, 72);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->m200SpatialRows, 0);
    EXPECT_NE(info.tilingKey, 15U);
}

TEST(MaxPool2dWithMaskTiling, LargeWindowUsesBoundedMemoryOnA2)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "A2 window reduction fallback";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        for (const uint64_t ubBytes : {196608U, 262144U}) {
            for (const int64_t channels : {1, 34, 270, 641}) {
                auto context = gert::TilingContextPara(
                    "MaxPool2dWithMask", {{{{1, channels, 64, 64}, {1, channels, 64, 64}}, dtype, ge::FORMAT_ND}},
                    {{{{1, channels, 1, 1}, {1, channels, 1, 1}}, dtype, ge::FORMAT_ND},
                     {{{1, channels, 4096, 64}, {1, channels, 4096, 64}}, ge::DT_INT8, ge::FORMAT_ND}},
                    {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{64, 64})},
                     {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{64, 64})},
                     {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0})},
                     {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                     {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                    &g_compileInfo, 40, ubBytes, 4096);
                TilingInfo info;
                ASSERT_TRUE(ExecuteTiling(context, info));
                EXPECT_EQ(info.tilingKey, dtype == ge::DT_BF16 && channels <= 40 ? 4U : 2U);
                const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
                EXPECT_EQ(data->inputBufferSize, 0U);
                EXPECT_EQ(data->validIndexBytes, channels * 4);
                EXPECT_GE(info.blockNum, std::min<uint64_t>(40, channels));
                const uint64_t tile = data->windowOutputTile;
                EXPECT_GE(tile, 1U);
                EXPECT_LE(tile, 8U);
                EXPECT_EQ(tile & (tile - 1), 0U);
                EXPECT_EQ(data->windowTailBoundaryCore, (channels - 1) / tile % info.blockNum);
                EXPECT_EQ(data->zeroBytesPerCore, 0);
                EXPECT_EQ(data->zeroBufferBytes, 0);
                const uint64_t vectorBytes = dtype == ge::DT_FLOAT ? 0 : MAX_POOL2D_WINDOW_VECTOR_BYTES;
                EXPECT_LT(MAX_POOL2D_WINDOW_INPUT_BYTES + 2 * MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS * sizeof(uint32_t) +
                              data->zeroBufferBytes + vectorBytes + 4096,
                          ubBytes);
            }
        }
    }
}

TEST(MaxPool2dWithMaskTiling, CachedPlanesPreserveSmallWindowBatchingAndBufferLimits)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "BF16 cached planes are supported on A2 only";
    }
    struct Geometry {
        int64_t nc;
        int64_t h;
        int64_t w;
        int64_t kh;
        int64_t kw;
        int64_t sh;
        int64_t sw;
        int64_t ph;
        int64_t pw;
        int64_t oh;
        int64_t ow;
        bool cached;
    };
    const std::vector<Geometry> geometries = {
        {65, 17, 13, 9, 5, 4, 2, 4, 2, 5, 7, false},          {8192, 7, 7, 7, 7, 2, 2, 3, 3, 4, 4, false},
        {41, 223, 4, 31, 4, 6, 4, 0, 0, 33, 1, true},         {41, 409, 4, 31, 4, 6, 4, 0, 0, 64, 1, true},
        {41, 415, 4, 31, 4, 6, 4, 0, 0, 65, 1, false},        {41, 256, 16, 256, 16, 256, 16, 128, 8, 2, 2, true},
        {41, 257, 16, 256, 16, 256, 16, 128, 8, 2, 2, false},
    };
    for (const auto& g : geometries) {
        const int64_t maskWidth = ((g.oh * g.ow + 15) / 16 + 1) * 32;
        auto context = gert::TilingContextPara(
            "MaxPool2dWithMask", {{{{1, g.nc, g.h, g.w}, {1, g.nc, g.h, g.w}}, ge::DT_BF16, ge::FORMAT_ND}},
            {{{{1, g.nc, g.oh, g.ow}, {1, g.nc, g.oh, g.ow}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{1, g.nc, g.kh * g.kw, maskWidth}, {1, g.nc, g.kh * g.kw, maskWidth}}, ge::DT_INT8, ge::FORMAT_ND}},
            {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.kh, g.kw})},
             {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.sh, g.sw})},
             {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.ph, g.pw})},
             {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
             {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
            &g_compileInfo, 40, 196608, 4096);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey == 5, g.cached);
        if (g.cached) {
            const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
            EXPECT_EQ(data->windowOutputTile, g.oh * g.ow);
            EXPECT_EQ(data->windowTailBoundaryCore, (g.nc - 1) % info.blockNum);
            EXPECT_EQ(data->windowPrefetch, 0U);
            EXPECT_LE(3 * MAX_POOL2D_WINDOW_INPUT_BYTES + MAX_POOL2D_WINDOW_VECTOR_BYTES +
                          2 * MAX_POOL2D_WINDOW_OUTPUT_ELEMENTS * sizeof(uint32_t) + data->zeroBufferBytes + 4096,
                      196608U);
        }
    }
}

TEST(MaxPool2dWithMaskTiling, WindowDispatchRespectsCoreAndInputCapacityBoundaries)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "BF16 cached planes are supported on A2 only";
    }
    struct Geometry {
        int64_t nc;
        int64_t h;
        int64_t w;
        int64_t kh;
        int64_t kw;
        int64_t sh;
        int64_t sw;
        int64_t ph;
        int64_t pw;
        int64_t oh;
        int64_t ow;
        uint64_t expectedKey;
    };
    const std::vector<Geometry> geometries = {
        {40, 64, 64, 64, 64, 64, 64, 0, 0, 1, 1, 4},    {41, 64, 64, 64, 64, 64, 64, 0, 0, 1, 1, 2},
        {39, 64, 65, 64, 65, 64, 65, 0, 0, 1, 1, 3},    {1, 64, 128, 64, 128, 64, 128, 0, 0, 1, 1, 3},
        {48, 64, 127, 64, 127, 64, 127, 0, 0, 1, 1, 3}, {128, 64, 97, 63, 97, 31, 32, 15, 16, 2, 2, 3},
        {1, 65, 128, 65, 128, 65, 128, 0, 0, 1, 1, 7},  {7, 96, 128, 96, 128, 96, 128, 0, 0, 1, 1, 7},
        {7, 97, 128, 97, 128, 97, 128, 0, 0, 1, 1, 2},  {3, 31, 32, 7, 7, 1, 1, 3, 3, 31, 32, 8},
        {3, 31, 32, 5, 5, 1, 1, 2, 2, 31, 32, 8},       {3, 31, 32, 1, 17, 1, 1, 0, 8, 31, 32, 8},
        {3, 31, 32, 5, 4, 1, 1, 2, 2, 31, 33, 8},       {3, 31, 32, 3, 3, 1, 1, 1, 1, 31, 32, 6},
        {7, 32, 128, 32, 128, 32, 128, 0, 0, 1, 1, 4},  {7, 64, 128, 32, 128, 32, 128, 0, 0, 2, 1, 4},
        {19, 64, 128, 32, 128, 32, 128, 0, 0, 2, 1, 4}, {21, 64, 128, 32, 128, 32, 128, 0, 0, 2, 1, 2},
        {7, 66, 128, 33, 128, 33, 128, 0, 0, 2, 1, 3},  {1, 36, 48, 31, 33, 10, 11, 7, 8, 2, 3, 4},
        {5, 36, 48, 31, 33, 10, 11, 7, 8, 2, 3, 4},     {7, 36, 48, 31, 33, 10, 11, 7, 8, 2, 3, 2},
        {19, 35, 81, 2, 2, 2, 2, 0, 0, 17, 40, 10},     {17, 33, 97, 2, 2, 2, 2, 0, 0, 16, 48, 10},
        {37, 33, 129, 2, 2, 2, 2, 0, 0, 16, 64, 6},     {5, 129, 33, 2, 2, 2, 2, 0, 0, 64, 16, 6},
        {37, 33, 65, 2, 2, 2, 2, 0, 0, 16, 32, 6},      {40, 3, 3, 8, 8, 8, 8, 4, 4, 1, 1, 4},
    };
    for (const auto& g : geometries) {
        const int64_t maskWidth = ((g.oh * g.ow + 15) / 16 + 1) * 32;
        auto context = gert::TilingContextPara(
            "MaxPool2dWithMask", {{{{1, g.nc, g.h, g.w}, {1, g.nc, g.h, g.w}}, ge::DT_BF16, ge::FORMAT_ND}},
            {{{{1, g.nc, g.oh, g.ow}, {1, g.nc, g.oh, g.ow}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{1, g.nc, g.kh * g.kw, maskWidth}, {1, g.nc, g.kh * g.kw, maskWidth}}, ge::DT_INT8, ge::FORMAT_ND}},
            {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.kh, g.kw})},
             {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.sh, g.sw})},
             {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.ph, g.pw})},
             {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
             {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
            &g_compileInfo, 40, 196608, 4096);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey, g.expectedKey);
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_EQ(data->validIndexBytes, g.nc * g.oh * g.ow * sizeof(int32_t));
        EXPECT_LE(info.blockNum, 40U);
        const auto* packet = reinterpret_cast<const MaxPool2dWithMaskTilingPacket*>(info.tilingData.get());
        const auto& window = packet->window;
        EXPECT_EQ(window.heavyCoreCount, 0U);
        EXPECT_EQ(window.heavyCoreZeroBytes, 0U);
        EXPECT_EQ(window.zeroBytesPerCore, 0);
        EXPECT_EQ(window.zeroBufferBytes, 0);
        if (g.expectedKey == 4) {
            EXPECT_EQ(data->windowOutputTile, 1);
            EXPECT_EQ(data->windowPrefetch, 0U);
        } else if (g.expectedKey == 3) {
            EXPECT_EQ(data->windowPrefetch, 1U);
            EXPECT_LE(data->zeroBufferBytes, MAX_POOL2D_WINDOW_LARGE_ZERO_BYTES);
        } else if (g.expectedKey == 7) {
            EXPECT_EQ(window.windowPrefetch, 1U);
            EXPECT_LE(window.zeroBufferBytes, MAX_POOL2D_WINDOW_WIDE_ZERO_BYTES);
        } else if (g.expectedKey == 8) {
            ASSERT_GT(window.windowOutputTile, 0U);
            EXPECT_LE(window.windowOutputTile, static_cast<uint64_t>(g.oh));
            const uint64_t tiles = (g.oh + window.windowOutputTile - 1) / window.windowOutputTile;
            EXPECT_EQ(window.windowTailBoundaryCore, (g.nc * tiles - 1) % info.blockNum);
        }
    }
}

TEST(MaxPool2dWithMaskTiling, SpatialSmallWindowRequiresCapacityAndBoundedWork)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "BF16 spatial Gather is supported on A2 only";
    }
    struct Geometry {
        int64_t nc;
        int64_t h;
        int64_t w;
        int64_t k;
        int64_t p;
        int64_t oh;
        int64_t ow;
        uint64_t ub;
        bool selected;
    };
    const std::vector<Geometry> geometries = {
        {20, 64, 128, 2, 0, 32, 64, 196608, true},  {19, 64, 128, 2, 0, 32, 64, 196608, true},
        {20, 65, 128, 2, 0, 33, 64, 196608, false}, {37, 33, 65, 2, 0, 17, 33, 196608, true},
        {37, 33, 65, 2, 0, 17, 33, 190719, false},  {64, 35, 51, 3, 1, 18, 26, 196608, false},
        {3, 31, 32, 3, 1, 16, 16, 196608, true},    {1, 33, 63, 3, 1, 17, 32, 196608, true},
        {40, 56, 56, 2, 0, 28, 28, 196608, true},   {41, 56, 56, 2, 0, 28, 28, 196608, false},
        {512, 56, 56, 2, 0, 28, 28, 196608, false}, {1024, 56, 56, 2, 0, 28, 28, 196608, false},
    };
    for (const auto& g : geometries) {
        const int64_t maskWidth = ((g.oh * g.ow + 15) / 16 + 1) * 32;
        auto context = gert::TilingContextPara(
            "MaxPool2dWithMask", {{{{1, g.nc, g.h, g.w}, {1, g.nc, g.h, g.w}}, ge::DT_BF16, ge::FORMAT_ND}},
            {{{{1, g.nc, g.oh, g.ow}, {1, g.nc, g.oh, g.ow}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{1, g.nc, g.k * g.k, maskWidth}, {1, g.nc, g.k * g.k, maskWidth}}, ge::DT_INT8, ge::FORMAT_ND}},
            {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.k})},
             {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2})},
             {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.p})},
             {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1})},
             {"ceilMode", Ops::NN::AnyValue::CreateFrom(g.k == 2)}},
            &g_compileInfo, 40, g.ub, 4096);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey == 6 || info.tilingKey == 10, g.selected);
        if (g.selected) {
            const uint64_t span = (g.ow - 1) * 2 + g.k;
            uint64_t powerPitch = 16;
            while (powerPitch < span) {
                powerPitch *= 2;
            }
            const bool compact = g.k == 2 && g.p == 0 && (span + 15) / 16 * 16 < powerPitch;
            EXPECT_EQ(info.tilingKey, compact ? 10U : 6U);
            const auto* packet = reinterpret_cast<const MaxPool2dWithMaskTilingPacket*>(info.tilingData.get());
            EXPECT_EQ(packet->window.zeroBytesPerCore % 32, 0);
            const auto rows = packet->window.windowOutputTile;
            ASSERT_GT(rows, 0U);
            ASSERT_LE(rows, static_cast<uint64_t>(g.oh));
            if (rows < static_cast<uint64_t>(g.oh)) {
                EXPECT_EQ(rows * g.ow * sizeof(uint16_t) % 32, 0U);
            }
            const auto tiles = (g.oh + rows - 1) / rows;
            EXPECT_EQ(packet->window.windowTailBoundaryCore, (g.nc * tiles - 1) % info.blockNum);
        }
    }
}

TEST(MaxPool2dWithMaskTiling, SpatialLowChannelsAllowRectangularWindowsAndGappedStride)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "BF16 spatial Gather is supported on A2 only";
    }
    struct Geometry {
        int64_t nc;
        int64_t h;
        int64_t w;
        int64_t kh;
        int64_t kw;
        int64_t sh;
        int64_t sw;
        int64_t ph;
        int64_t pw;
        int64_t oh;
        int64_t ow;
        bool selected;
    };
    const std::vector<Geometry> geometries = {
        {3, 42, 66, 7, 2, 3, 2, 3, 1, 14, 34, true},
        {16, 28, 51, 2, 2, 3, 2, 1, 1, 10, 26, true},
        {9, 35, 51, 3, 3, 2, 2, 1, 1, 18, 26, false},
    };
    for (const auto& g : geometries) {
        const int64_t maskWidth = ((g.oh * g.ow + 15) / 16 + 1) * 32;
        auto context = gert::TilingContextPara(
            "MaxPool2dWithMask", {{{{1, g.nc, g.h, g.w}, {1, g.nc, g.h, g.w}}, ge::DT_BF16, ge::FORMAT_ND}},
            {{{{1, g.nc, g.oh, g.ow}, {1, g.nc, g.oh, g.ow}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{1, g.nc, g.kh * g.kw, maskWidth}, {1, g.nc, g.kh * g.kw, maskWidth}}, ge::DT_INT8, ge::FORMAT_ND}},
            {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.kh, g.kw})},
             {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.sh, g.sw})},
             {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.ph, g.pw})},
             {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1})},
             {"ceilMode", Ops::NN::AnyValue::CreateFrom(g.kh == 2)}},
            &g_compileInfo, 40, 196608, 4096);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey == 6, g.selected);
    }
}

TEST(MaxPool2dWithMaskTiling, SpatialScanComparesParallelWork)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "BF16 spatial Gather is supported on A2 only";
    }
    struct Geometry {
        int64_t nc;
        int64_t h;
        int64_t w;
        int64_t kh;
        int64_t kw;
        int64_t sh;
        int64_t sw;
        int64_t ph;
        int64_t pw;
        int64_t oh;
        int64_t ow;
        bool selected;
    };
    const std::vector<Geometry> geometries = {
        // Few NC lanes favor scan; many channels with a small window do not.
        {3, 31, 32, 7, 7, 1, 1, 3, 3, 31, 32, true},
        {127, 15, 29, 5, 5, 1, 1, 2, 2, 15, 29, false},
        // Many channels alone must not exclude scan when window reuse is high.
        {67, 41, 19, 15, 9, 1, 2, 7, 4, 41, 10, true},
        {33, 45, 63, 4, 6, 2, 3, 2, 3, 23, 22, false},
        // The scan decision comes from work/reuse, not the old 128-output gate.
        {31, 17, 25, 23, 9, 2, 2, 11, 4, 9, 13, true},
        {1, 1, 17, 5, 5, 1, 1, 2, 2, 1, 17, false},
        // Neutral right padding must not double the scan state or force NC.
        {97, 19, 61, 7, 7, 1, 2, 0, 3, 13, 31, true},
        // The compact pitch also admits a plane excluded by the old UB bound.
        {3, 15, 113, 31, 31, 1, 1, 15, 15, 15, 113, true},
        // Width/height above 31 remain legal when their padded states fit.
        {39, 23, 35, 15, 47, 1, 1, 7, 23, 23, 35, true},
        {1, 23, 13, 47, 15, 1, 3, 23, 7, 23, 5, true},
        {3, 15, 33, 15, 63, 1, 1, 7, 31, 15, 33, true},
        {3, 17, 31, 63, 7, 1, 1, 31, 3, 17, 31, true},
        // Physical state capacity is still mandatory after removing the cap.
        {3, 15, 113, 95, 63, 1, 1, 47, 31, 15, 113, false},
    };
    for (const auto& g : geometries) {
        const int64_t maskWidth = ((g.oh * g.ow + 15) / 16 + 1) * 32;
        auto context = gert::TilingContextPara(
            "MaxPool2dWithMask", {{{{1, g.nc, g.h, g.w}, {1, g.nc, g.h, g.w}}, ge::DT_BF16, ge::FORMAT_ND}},
            {{{{1, g.nc, g.oh, g.ow}, {1, g.nc, g.oh, g.ow}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{1, g.nc, g.kh * g.kw, maskWidth}, {1, g.nc, g.kh * g.kw, maskWidth}}, ge::DT_INT8, ge::FORMAT_ND}},
            {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.kh, g.kw})},
             {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.sh, g.sw})},
             {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.ph, g.pw})},
             {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1})},
             {"ceilMode", Ops::NN::AnyValue::CreateFrom(g.kh == 2)}},
            &g_compileInfo, 40, 196608, 4096);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey == 8, g.selected);
    }
}

TEST(MaxPool2dWithMaskTiling, EquivalentRectanglesKeepIndependentWindowFallbacks)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "BF16 equivalent rectangles are supported on A2 only";
    }
    struct Geometry {
        int64_t nc;
        int64_t h;
        int64_t w;
        int64_t kh;
        int64_t kw;
        int64_t sh;
        int64_t sw;
        int64_t ph;
        int64_t pw;
        int64_t oh;
        int64_t ow;
        uint64_t key;
    };
    const std::vector<Geometry> geometries = {
        {17, 13, 17, 31, 31, 3, 1, 15, 15, 5, 17, 9},
        {17, 33, 13, 31, 31, 3, 3, 15, 15, 11, 5, 9},
        {17, 13, 33, 31, 31, 3, 3, 15, 15, 5, 11, 9},
        {17, 3, 63, 7, 127, 1, 1, 3, 63, 3, 63, 9},
        // Existing scan and single-window algorithms retain their precedence.
        {3, 13, 17, 31, 31, 3, 1, 15, 15, 5, 17, 8},
        {1, 17, 17, 17, 17, 1, 1, 0, 0, 1, 1, 4},
    };
    for (const auto& g : geometries) {
        const int64_t maskWidth = ((g.oh * g.ow + 15) / 16 + 1) * 32;
        auto context = gert::TilingContextPara(
            "MaxPool2dWithMask", {{{{1, g.nc, g.h, g.w}, {1, g.nc, g.h, g.w}}, ge::DT_BF16, ge::FORMAT_ND}},
            {{{{1, g.nc, g.oh, g.ow}, {1, g.nc, g.oh, g.ow}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{1, g.nc, g.kh * g.kw, maskWidth}, {1, g.nc, g.kh * g.kw, maskWidth}}, ge::DT_INT8, ge::FORMAT_ND}},
            {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.kh, g.kw})},
             {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.sh, g.sw})},
             {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{g.ph, g.pw})},
             {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1})},
             {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
            &g_compileInfo, 40, 196608, 4096);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey, g.key);
    }
}

TEST(MaxPool2dWithMaskTiling, SmallBfloat16UsesIndependentChannelTasks)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "BF16 is supported on A2 only";
    }
    auto context = MakeTilingContext(ge::DT_BF16);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->coreNums, 1U);
    EXPECT_EQ(info.tilingKey, 6U);
    EXPECT_EQ(info.blockNum, data->totalSlices);
    EXPECT_GE(data->ncFactor, data->totalSlices);
}

TEST(MaxPool2dWithMaskTiling, WideInputUsesAlgorithmSpecificBuffers)
{
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        for (const uint64_t ubBytes : {196608U, 262144U}) {
            auto context = gert::TilingContextPara(
                "MaxPool2dWithMask", {{{{1, 1, 2, 32768}, {1, 1, 2, 32768}}, dtype, ge::FORMAT_ND}},
                {{{{1, 1, 1, 16384}, {1, 1, 1, 16384}}, dtype, ge::FORMAT_ND},
                 {{{1, 1, 4, 32800}, {1, 1, 4, 32800}}, ge::DT_INT8, ge::FORMAT_ND}},
                {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                 {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                 {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0})},
                 {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                 {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                &g_compileInfo, 8, ubBytes, 4096);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(context, info));
            const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
            EXPECT_EQ(info.tilingKey, IS_ASCEND_310P_TEST ? 50U : 0U);
            if constexpr (IS_ASCEND_310P_TEST) {
                EXPECT_EQ(data->outputRowsPerTile, 0);
                EXPECT_LT(data->zeroBufferBytes, ubBytes);
            } else {
                EXPECT_GT(data->wyFactor, 0U);
                EXPECT_LT(data->wyFactor, 16384U);
                const auto bytes = 2 * (data->inputBufferSize + data->outputMaxBufferSize +
                                        data->outputIndiceBufferSize + data->maskBufferSize) +
                                   data->indiceTempBufferSize;
                EXPECT_LT(bytes, ubBytes);
            }
        }
    }
}

TEST(MaxPool2dWithMaskTiling, SmallSpatialOutputsUseIndependentChannels)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "A2 BF16 spatial path";
    }
    for (const int64_t channels : {8, 16}) {
        auto context = gert::TilingContextPara(
            "MaxPool2dWithMask", {{{{1, channels, 9, 13}, {1, channels, 9, 13}}, ge::DT_BF16, ge::FORMAT_ND}},
            {{{{1, channels, 4, 6}, {1, channels, 4, 6}}, ge::DT_BF16, ge::FORMAT_ND},
             {{{1, channels, 4, 96}, {1, channels, 4, 96}}, ge::DT_INT8, ge::FORMAT_ND}},
            {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
             {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
             {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0})},
             {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
             {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
            &g_compileInfo, 40, 196608, 4096);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        EXPECT_EQ(info.tilingKey, channels == 8 ? 6U : 0U);
        EXPECT_EQ(info.blockNum, channels == 8 ? 8U : 1U);
    }
}

TEST(MaxPool2dWithMaskTiling, SmallIndependentOutputsUseWindowReduction)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "A2 output-parallel window reduction";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        for (const int64_t channels : {1, 3}) {
            auto context = gert::TilingContextPara(
                "MaxPool2dWithMask", {{{{1, channels, 4, 4}, {1, channels, 4, 4}}, dtype, ge::FORMAT_ND}},
                {{{{1, channels, 2, 2}, {1, channels, 2, 2}}, dtype, ge::FORMAT_ND},
                 {{{1, channels, 4, 64}, {1, channels, 4, 64}}, ge::DT_INT8, ge::FORMAT_ND}},
                {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                 {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{2, 2})},
                 {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0})},
                 {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
                 {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                &g_compileInfo, 8, 196608, 4096);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(context, info));
            if (channels == 1) {
                EXPECT_EQ(info.tilingKey, 2U);
                EXPECT_EQ(info.blockNum, 4U);
            } else {
                EXPECT_EQ(info.tilingKey, 0U);
            }
        }
    }
}

TEST(MaxPool2dWithMaskTiling, TaskCaseFloat32)
{
    auto context = MakeTilingContext(ge::DT_FLOAT);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, IS_ASCEND_310P_TEST ? 10U : 11U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], 0U);
    ASSERT_GE(info.tilingDataSize, sizeof(MaxPool2dWithMaskTilingData));

    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->batchSize, 2);
    EXPECT_EQ(data->channels, 3);
    EXPECT_EQ(data->inputHeight, 8);
    EXPECT_EQ(data->inputWidth, 9);
    EXPECT_EQ(data->outputHeight, 4);
    EXPECT_EQ(data->outputWidth, 5);
    EXPECT_EQ(data->totalSlices, 6);
    EXPECT_EQ(data->outputRowsPerTile, IS_ASCEND_310P_TEST ? 4 : 0);
    EXPECT_EQ(data->inputWidthAligned, IS_ASCEND_310P_TEST ? 16 : 0);
    EXPECT_EQ(data->outputElementsPerTileAligned, IS_ASCEND_310P_TEST ? 24 : 0);
    EXPECT_EQ(data->validIndexBytes, 480);
    EXPECT_EQ(data->containerBytes, 5184);
    EXPECT_EQ(data->nc, 6U);
    EXPECT_EQ(data->m200MapRows, 0); // The unaligned output plane needs the general path.
    EXPECT_EQ(data->columnBufferElements, IS_ASCEND_310P_TEST ? 16 : 0);
    if constexpr (IS_ASCEND_310P_TEST) {
        EXPECT_EQ(data->slicesPerCore, 6);
        EXPECT_EQ(info.blockNum, 1U);
        EXPECT_EQ(data->zeroBytesPerCore, 4704);
        EXPECT_EQ(data->ncFactor, 0U);
        EXPECT_EQ(data->ncOuter, 0U);
        EXPECT_EQ(data->hyFactor, 0U);
        EXPECT_EQ(data->wyFactor, 0U);
        EXPECT_EQ(data->coreNums, 0U);
        EXPECT_EQ(data->inputBufferSize, 0U);
        EXPECT_EQ(data->outputMaxBufferSize, 0U);
        EXPECT_EQ(data->outputIndiceBufferSize, 0U);
        EXPECT_EQ(data->indiceTempBufferSize, 0U);
        EXPECT_EQ(data->maskBufferSize, 0U);
    } else {
        EXPECT_EQ(data->slicesPerCore, 1);
        EXPECT_EQ(info.blockNum, 6U);
        EXPECT_EQ(data->zeroBytesPerCore, 784);
        EXPECT_EQ(data->ncFactor, 6U);
        EXPECT_EQ(data->ncOuter, 1U);
        EXPECT_EQ(data->hyFactor, 4U);
        EXPECT_EQ(data->wyFactor, 5U);
        EXPECT_EQ(data->coreNums, 1U);
        EXPECT_EQ(data->inputBufferSize, 9216U);
        EXPECT_EQ(data->outputMaxBufferSize, 4096U);
        EXPECT_EQ(data->outputIndiceBufferSize, 16384U);
        EXPECT_EQ(data->indiceTempBufferSize, 4096U);
        EXPECT_EQ(data->maskBufferSize, 512U);
    }
}

TEST(MaxPool2dWithMaskTiling, FallsBackWhenColumnScratchDoesNotFit)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Separable reduction is specific to Ascend 310P";
    }
    // Fixed buffers use 12288 bytes and one tile row uses 512 bytes. The remaining
    // 64 bytes cannot hold the 128-byte column scratch, but the direct kernel fits.
    auto context = MakeTilingContext(ge::DT_FLOAT, 12864);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->columnBufferElements, 0);
    EXPECT_EQ(data->outputRowsPerTile, 1);
}

TEST(MaxPool2dWithMaskTiling, UsesVectorGatherForWideFloat32Interior)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "Gather reduction is specific to Ascend 310P";
    }
    auto context = MakeCacheLinePartitionContext(ge::DT_FLOAT, 20 * 1024);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(info.tilingKey, 10U);
    EXPECT_EQ(data->vectorBufferElements, 64);
    EXPECT_EQ(data->columnBufferElements, 0);
}

TEST(MaxPool2dWithMaskTiling, RejectsUnsupportedDtype)
{
    auto context = MakeTilingContext(ge::DT_INT32);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(MaxPool2dWithMaskTiling, UsesAlignedVectorTilesForBothFloatingTypes)
{
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeCacheLinePartitionContext(dtype);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        if constexpr (IS_ASCEND_310P_TEST) {
            ASSERT_GT(data->m200MapRows, 0);
            EXPECT_EQ(info.tilingKey, 11U);
            EXPECT_LE(data->m200MapRows, data->outputHeight);
            const int64_t bytes = dtype == ge::DT_FLOAT ? 4 : 2;
            EXPECT_EQ(data->m200MapRows * data->outputWidth * bytes % 64, 0);
        } else {
            EXPECT_EQ(data->m200MapRows, 0);
        }
    }
}

TEST(MaxPool2dWithMaskTiling, FallsBackWhenVectorMapsDoNotFit)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "The window maps are specific to Ascend 310P";
    }
    auto context = MakeCacheLinePartitionContext(ge::DT_FLOAT16, 20 * 1024);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->m200MapRows, 0);
    EXPECT_GT(data->outputRowsPerTile, 0);
    EXPECT_EQ(info.tilingKey, 10U);
}

TEST(MaxPool2dWithMaskTiling, UsesChannelTransposeForAlignedGroups)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "The channel transpose is specific to Ascend 310P";
    }
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16}) {
        auto context = MakeCacheLinePartitionContext(dtype, 196608, 64);
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(context, info));
        const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
        EXPECT_GT(data->m200NcRows, 0);
        EXPECT_LE(data->m200NcRows, data->outputHeight);
        EXPECT_EQ(data->m200MapRows, 0);
        // This context launches 40 cores for only a few spatial tiles. A core
        // does not own adjacent tiles of the same channel group, so do not cache.
        EXPECT_EQ(info.tilingKey, 24U);
    }
}

TEST(MaxPool2dWithMaskTiling, AlignsScalarCacheLinePartitionsOn310P)
{
    if constexpr (!IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "M200 scalar cache partitioning is specific to Ascend 310P";
    }
    auto context = MakeCacheLinePartitionContext();
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
    EXPECT_EQ(data->totalSlices, 3);
    EXPECT_EQ(data->slicesPerCore, 1);
    EXPECT_EQ(data->validIndexBytes, 3072);
    EXPECT_EQ(data->containerBytes, 14688);
    EXPECT_EQ(data->zeroBytesPerCore, 3904);
}
TEST(MaxPool2dWithMaskTiling, OneByOneContainerIncludesAllIndices)
{
    for (const auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        if (IS_ASCEND_310P_TEST && dtype == ge::DT_BF16) {
            continue;
        }
        for (const auto& boundary :
             {std::pair<int64_t, int64_t>{16, 64}, {17, 96}, {24, 96}, {25, 128}, {32, 128}, {33, 160}}) {
            const int64_t elements = boundary.first;
            const int64_t width = boundary.second;
            auto context = gert::TilingContextPara(
                "MaxPool2dWithMask", {{{{2, 3, 1, elements}, {2, 3, 1, elements}}, dtype, ge::FORMAT_ND}},
                {{{{2, 3, 1, elements}, {2, 3, 1, elements}}, dtype, ge::FORMAT_ND},
                 {{{2, 3, 1, width}, {2, 3, 1, width}}, ge::DT_INT8, ge::FORMAT_ND}},
                {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1})},
                 {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1})},
                 {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0})},
                 {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1})},
                 {"ceilMode", Ops::NN::AnyValue::CreateFrom(false)}},
                &g_compileInfo, 40, 196608, 4096);
            TilingInfo info;
            ASSERT_TRUE(ExecuteTiling(context, info));
            const auto* data = reinterpret_cast<const MaxPool2dWithMaskTilingData*>(info.tilingData.get());
            EXPECT_EQ(data->validIndexBytes, 6 * elements * sizeof(int32_t));
            EXPECT_EQ(data->containerBytes, 6 * width);
            EXPECT_GE(data->containerBytes, data->validIndexBytes);
        }
    }
}

TEST(MaxPool2dWithMaskTiling, MaximumStrideDoesNotCreateEquivalentWindows)
{
    if constexpr (IS_ASCEND_310P_TEST) {
        GTEST_SKIP() << "BF16 window grouping is specific to A2";
    }
    const int64_t stride = std::numeric_limits<int64_t>::max();
    auto context = gert::TilingContextPara(
        "MaxPool2dWithMask", {{{{1, 1, 5, 5}, {1, 1, 5, 5}}, ge::DT_BF16, ge::FORMAT_ND}},
        {{{{1, 1, 1, 1}, {1, 1, 1, 1}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1, 1, 9, 64}, {1, 1, 9, 64}}, ge::DT_INT8, ge::FORMAT_ND}},
        {{"kernelSize", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{3, 3})},
         {"stride", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{stride, stride})},
         {"padding", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{0, 0})},
         {"dilation", Ops::NN::AnyValue::CreateFrom(std::vector<int64_t>{1, 1})},
         {"ceilMode", Ops::NN::AnyValue::CreateFrom(true)}},
        &g_compileInfo, 40, 196608, 4096);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, 4U);
    const auto* packet = reinterpret_cast<const MaxPool2dWithMaskTilingPacket*>(info.tilingData.get());
    EXPECT_EQ(packet->window.sh, stride);
    EXPECT_EQ(packet->window.sw, stride);
    EXPECT_EQ(packet->window.validIndexBytes, sizeof(int32_t));
}

} // namespace
