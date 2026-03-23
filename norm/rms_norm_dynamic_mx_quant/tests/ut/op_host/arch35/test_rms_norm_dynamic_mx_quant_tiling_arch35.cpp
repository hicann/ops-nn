/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "register/op_impl_registry.h"
#include "tests/utils/op_host_test_util.h"
#include "op_host/arch35/rms_norm_dynamic_mx_quant_tiling_arch35.h"

using namespace optiling;
#define TILING_KEY_FULL_LOAD_GENERAL 1000
class RmsNormDynamicMxQuantTilingTest : public testing::Test {};

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_small_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_medium_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{4096}, {4096}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_large_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_2d_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_3d_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_single_element_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_aligned_256_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{256}, {256}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_4d_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_3d_medium_fp16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_small_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{1024}, {1024}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_medium_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_large_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_2d_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{64, 128}, {64, 128}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{64, 128}, {64, 128}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_3d_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{16, 32, 64}, {16, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{16, 32, 64}, {16, 32, 64}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_single_element_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND}, {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_aligned_256_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_4d_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}

TEST_F(RmsNormDynamicMxQuantTilingTest, tiling_3d_medium_bf16)
{
    gert::TilingContextPara tilingContextPara(
        "RmsNormDynamicMxQuant", {{{8, 16, 32}, {8, 16, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{8, 16, 32}, {8, 16, 32}}, ge::DT_BF16, ge::FORMAT_ND});
    auto context = gert::TilingContext(tilingContextPara);
    uint64_t tilingKey = context.GetTilingKey();
    EXPECT_EQ(tilingKey, TILING_KEY_FULL_LOAD_GENERAL);
}
