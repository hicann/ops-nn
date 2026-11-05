/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv_basic_block_extra_tiling.cpp
 * \brief conv_basic_block.cpp 基础块决策函数补测：GetInitBasicBlockMN(常规L0C)与
 *        Get512InitBasicBlockMN(512K L0C, 960系)的全部档位分支均为自由函数，
 *        直接以参数化shape/attr组合调用覆盖各档位，与全流程tiling用例互补。
 */

#include <gtest/gtest.h>

#include <vector>

#include "../../../op_host/op_tiling/arch35/conv_basic_block.h"
#include "../../../op_host/op_tiling/arch35/conv_base.h"

using namespace conv_tiling;

namespace {
constexpr uint64_t BB_M_64 = 64;
constexpr uint64_t BB_M_128 = 128;
constexpr uint64_t BB_M_256 = 256;
constexpr uint64_t BB_M_512 = 512;
constexpr uint64_t BB_M_1024 = 1024;
constexpr uint64_t BB_N_64 = 64;
constexpr uint64_t BB_N_128 = 128;
constexpr uint64_t BB_N_256 = 256;
constexpr uint64_t BB_N_512 = 512;
constexpr uint64_t BB_N_1024 = 1024;

ConvAscendcShapesInfo MakeShapes(uint64_t ho, uint64_t wo, uint64_t kh, uint64_t kw)
{
    ConvAscendcShapesInfo shapeInfo;
    shapeInfo.ho = ho;
    shapeInfo.wo = wo;
    shapeInfo.kh = kh;
    shapeInfo.kw = kw;
    return shapeInfo;
}

ConvAscendcAttrInfo MakeAttrs(uint64_t strideH, uint64_t strideW)
{
    ConvAscendcAttrInfo attrInfo;
    attrInfo.strideH = strideH;
    attrInfo.strideW = strideW;
    return attrInfo;
}
} // namespace

class ConvBasicBlockExtraTest : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

// Get512InitBasicBlockMN: co<=128档
TEST_F(ConvBasicBlockExtraTest, Get512_co_le_128)
{
    auto shapeInfo = MakeShapes(32, 32, 3, 3);
    auto attrInfo = MakeAttrs(1, 1);
    shapeInfo.co = 100;
    uint64_t m = 0;
    uint64_t n = 0;
    Get512InitBasicBlockMN(shapeInfo, attrInfo, m, n);
    EXPECT_EQ(m, BB_M_1024);
    EXPECT_EQ(n, BB_N_128);
}

// Get512InitBasicBlockMN: 128<co<=256档
TEST_F(ConvBasicBlockExtraTest, Get512_co_in_128_256)
{
    auto shapeInfo = MakeShapes(32, 32, 3, 3);
    auto attrInfo = MakeAttrs(1, 1);
    shapeInfo.co = 200;
    uint64_t m = 0;
    uint64_t n = 0;
    Get512InitBasicBlockMN(shapeInfo, attrInfo, m, n);
    EXPECT_EQ(m, BB_M_512);
    EXPECT_EQ(n, BB_N_256);
}

// Get512InitBasicBlockMN: co>256且howo<=128档
TEST_F(ConvBasicBlockExtraTest, Get512_howo_le_128)
{
    auto shapeInfo = MakeShapes(8, 8, 3, 3);
    auto attrInfo = MakeAttrs(1, 1);
    shapeInfo.co = 512;
    uint64_t m = 0;
    uint64_t n = 0;
    Get512InitBasicBlockMN(shapeInfo, attrInfo, m, n);
    EXPECT_EQ(m, BB_M_128);
    EXPECT_EQ(n, BB_N_1024);
}

// Get512InitBasicBlockMN: 128<howo<=256档
TEST_F(ConvBasicBlockExtraTest, Get512_howo_in_128_256)
{
    auto shapeInfo = MakeShapes(12, 12, 3, 3); // howo=144 落入(128,256]
    auto attrInfo = MakeAttrs(1, 1);
    shapeInfo.co = 512;
    uint64_t m = 0;
    uint64_t n = 0;
    Get512InitBasicBlockMN(shapeInfo, attrInfo, m, n);
    EXPECT_EQ(m, BB_M_256);
    EXPECT_EQ(n, BB_N_512);
}

// Get512InitBasicBlockMN: khkw>shsw档(如3x3核+1x1步长)
TEST_F(ConvBasicBlockExtraTest, Get512_khkw_gt_shsw)
{
    auto shapeInfo = MakeShapes(32, 64, 3, 3);
    auto attrInfo = MakeAttrs(1, 1);
    shapeInfo.co = 512;
    uint64_t m = 0;
    uint64_t n = 0;
    Get512InitBasicBlockMN(shapeInfo, attrInfo, m, n);
    EXPECT_EQ(m, BB_M_512);
    EXPECT_EQ(n, BB_N_256);
}

// Get512InitBasicBlockMN: khkw<shsw档(如1x1核+2x2步长)
TEST_F(ConvBasicBlockExtraTest, Get512_khkw_lt_shsw)
{
    auto shapeInfo = MakeShapes(32, 64, 1, 1);
    auto attrInfo = MakeAttrs(2, 2);
    shapeInfo.co = 512;
    uint64_t m = 0;
    uint64_t n = 0;
    Get512InitBasicBlockMN(shapeInfo, attrInfo, m, n);
    EXPECT_EQ(m, BB_M_256);
    EXPECT_EQ(n, BB_N_512);
}

// Get512InitBasicBlockMN: khkw==shsw档
TEST_F(ConvBasicBlockExtraTest, Get512_khkw_eq_shsw)
{
    auto shapeInfo = MakeShapes(32, 64, 2, 2);
    auto attrInfo = MakeAttrs(2, 2);
    shapeInfo.co = 512;
    uint64_t m = 0;
    uint64_t n = 0;
    Get512InitBasicBlockMN(shapeInfo, attrInfo, m, n);
    EXPECT_EQ(m, BB_M_256);
    EXPECT_EQ(n, BB_N_256);
}

// GetInitBasicBlockMN: co<=64且fp16x+int8w档
TEST_F(ConvBasicBlockExtraTest, GetInit_co_le_64_fp16x_int8w)
{
    auto shapeInfo = MakeShapes(32, 32, 3, 3);
    ConvAscendcDescInfo descInfo;
    descInfo.fMapDtype = ge::DT_FLOAT16;
    descInfo.weightDtype = ge::DT_INT8;
    uint64_t m = 0;
    uint64_t n = 0;
    GetInitBasicBlockMN(shapeInfo, descInfo, 64, m, n);
    EXPECT_EQ(m, BB_M_512);
    EXPECT_EQ(n, BB_N_64);
}

// GetInitBasicBlockMN: co<=64其他dtype档
TEST_F(ConvBasicBlockExtraTest, GetInit_co_le_64_other_dtype)
{
    auto shapeInfo = MakeShapes(32, 32, 3, 3);
    ConvAscendcDescInfo descInfo;
    descInfo.fMapDtype = ge::DT_FLOAT16;
    descInfo.weightDtype = ge::DT_FLOAT16;
    uint64_t m = 0;
    uint64_t n = 0;
    GetInitBasicBlockMN(shapeInfo, descInfo, 64, m, n);
    EXPECT_EQ(m, BB_M_1024);
    EXPECT_EQ(n, BB_N_64);
}

// GetInitBasicBlockMN: 64<co<=128档
TEST_F(ConvBasicBlockExtraTest, GetInit_co_in_64_128)
{
    auto shapeInfo = MakeShapes(32, 32, 3, 3);
    ConvAscendcDescInfo descInfo;
    uint64_t m = 0;
    uint64_t n = 0;
    GetInitBasicBlockMN(shapeInfo, descInfo, 100, m, n);
    EXPECT_EQ(m, BB_M_512);
    EXPECT_EQ(n, BB_N_128);
}

// GetInitBasicBlockMN: howo<=64档
TEST_F(ConvBasicBlockExtraTest, GetInit_howo_le_64)
{
    auto shapeInfo = MakeShapes(8, 8, 3, 3);
    ConvAscendcDescInfo descInfo;
    uint64_t m = 0;
    uint64_t n = 0;
    GetInitBasicBlockMN(shapeInfo, descInfo, 512, m, n);
    EXPECT_EQ(m, BB_M_64);
    EXPECT_EQ(n, BB_N_1024);
}

// GetInitBasicBlockMN: 64<howo<=128档
TEST_F(ConvBasicBlockExtraTest, GetInit_howo_in_64_128)
{
    auto shapeInfo = MakeShapes(8, 16, 3, 3);
    ConvAscendcDescInfo descInfo;
    uint64_t m = 0;
    uint64_t n = 0;
    GetInitBasicBlockMN(shapeInfo, descInfo, 512, m, n);
    EXPECT_EQ(m, BB_M_128);
    EXPECT_EQ(n, BB_N_512);
}

// GetInitBasicBlockMN: 兜底档
TEST_F(ConvBasicBlockExtraTest, GetInit_default)
{
    auto shapeInfo = MakeShapes(32, 64, 3, 3);
    ConvAscendcDescInfo descInfo;
    uint64_t m = 0;
    uint64_t n = 0;
    GetInitBasicBlockMN(shapeInfo, descInfo, 512, m, n);
    EXPECT_EQ(m, BB_M_256);
    EXPECT_EQ(n, BB_N_256);
}
