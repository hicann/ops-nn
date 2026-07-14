/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <iostream>
#include "tiling_case_executor.h"

#include "../../../op_kernel/relu_grad_v3_tiling_data.h"

using namespace std;
using namespace ge;

class ReluGradV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ReluGradV3Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ReluGradV3Tiling TearDown" << std::endl; }
};

static gert::TilingContextPara MakeTilingPara(ge::DataType dtype, const gert::StorageShape& xShape,
                                              const gert::StorageShape& yShape, const gert::StorageShape& zShape)
{
    struct ReluGradV3CompileInfo {
    } compileInfo;
    return gert::TilingContextPara("ReluGradV3",
                                   {
                                       {xShape, dtype, ge::FORMAT_ND},
                                       {yShape, dtype, ge::FORMAT_ND},
                                   },
                                   {
                                       {zShape, dtype, ge::FORMAT_ND},
                                   },
                                   {}, &compileInfo, 64, 262144, 4096);
}

static const ReluGradV3TilingData* GetTilingData(const TilingInfo& tilingInfo)
{
    EXPECT_EQ(tilingInfo.tilingDataSize, sizeof(ReluGradV3TilingData));
    return reinterpret_cast<const ReluGradV3TilingData*>(tilingInfo.tilingData.get());
}

TEST_F(ReluGradV3Tiling, relu_grad_v3_float32_success)
{
    TilingInfo tilingInfo;
    auto para = MakeTilingPara(ge::DT_FLOAT, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}});
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    auto tiling = GetTilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tilingInfo.tilingKey, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    EXPECT_EQ(tiling->totalLength, 8192);
    EXPECT_EQ(tiling->broadcastMode, 0);
    EXPECT_EQ(tiling->dimNum, 2);
    EXPECT_EQ(tiling->xElementNum, 8192);
    EXPECT_EQ(tiling->yElementNum, 8192);
    EXPECT_EQ(tiling->outShape[0], 256);
    EXPECT_EQ(tiling->outShape[1], 32);
    EXPECT_EQ(tiling->xStrides[0], 32);
    EXPECT_EQ(tiling->xStrides[1], 1);
    EXPECT_EQ(tiling->yStrides[0], 32);
    EXPECT_EQ(tiling->yStrides[1], 1);
}

TEST_F(ReluGradV3Tiling, relu_grad_v3_float16_success)
{
    TilingInfo tilingInfo;
    auto para = MakeTilingPara(ge::DT_FLOAT16, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}});
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    auto tiling = GetTilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalLength, 8192);
    EXPECT_EQ(tiling->broadcastMode, 0);
    EXPECT_EQ(tiling->dimNum, 2);
}

TEST_F(ReluGradV3Tiling, relu_grad_v3_bfloat16_success)
{
    TilingInfo tilingInfo;
    auto para = MakeTilingPara(ge::DT_BF16, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}});
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    auto tiling = GetTilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalLength, 8192);
    EXPECT_EQ(tiling->broadcastMode, 0);
    EXPECT_EQ(tiling->dimNum, 2);
}

TEST_F(ReluGradV3Tiling, relu_grad_v3_int32_success)
{
    TilingInfo tilingInfo;
    auto para = MakeTilingPara(ge::DT_INT32, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}});
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    auto tiling = GetTilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalLength, 8192);
    EXPECT_EQ(tiling->broadcastMode, 0);
}

TEST_F(ReluGradV3Tiling, relu_grad_v3_uint8_success)
{
    TilingInfo tilingInfo;
    auto para = MakeTilingPara(ge::DT_UINT8, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}});
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    auto tiling = GetTilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalLength, 8192);
    EXPECT_EQ(tiling->broadcastMode, 0);
}

TEST_F(ReluGradV3Tiling, relu_grad_v3_int8_success)
{
    TilingInfo tilingInfo;
    auto para = MakeTilingPara(ge::DT_INT8, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}}, {{256, 32}, {256, 32}});
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    auto tiling = GetTilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalLength, 8192);
    EXPECT_EQ(tiling->broadcastMode, 0);
}

TEST_F(ReluGradV3Tiling, relu_grad_v3_float32_y_scalar_broadcast_success)
{
    TilingInfo tilingInfo;
    auto para = MakeTilingPara(ge::DT_FLOAT, {{256, 32}, {256, 32}}, {{1}, {1}}, {{256, 32}, {256, 32}});
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    auto tiling = GetTilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalLength, 8192);
    EXPECT_EQ(tiling->broadcastMode, 1);
    EXPECT_EQ(tiling->dimNum, 2);
    EXPECT_EQ(tiling->xElementNum, 8192);
    EXPECT_EQ(tiling->yElementNum, 1);
    EXPECT_EQ(tiling->outShape[0], 256);
    EXPECT_EQ(tiling->outShape[1], 32);
    EXPECT_EQ(tiling->xStrides[0], 32);
    EXPECT_EQ(tiling->xStrides[1], 1);
    EXPECT_EQ(tiling->yStrides[0], 0);
    EXPECT_EQ(tiling->yStrides[1], 0);
}

TEST_F(ReluGradV3Tiling, relu_grad_v3_float32_4d_broadcast_success)
{
    TilingInfo tilingInfo;
    auto para = MakeTilingPara(ge::DT_FLOAT, {{8, 32, 1, 64}, {8, 32, 1, 64}}, {{1, 32, 128, 64}, {1, 32, 128, 64}},
                               {{8, 32, 128, 64}, {8, 32, 128, 64}});
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    auto tiling = GetTilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->totalLength, 2097152);
    EXPECT_EQ(tiling->broadcastMode, 1);
    EXPECT_EQ(tiling->dimNum, 4);
    EXPECT_EQ(tiling->xElementNum, 16384);
    EXPECT_EQ(tiling->yElementNum, 262144);
    EXPECT_EQ(tiling->outShape[0], 8);
    EXPECT_EQ(tiling->outShape[1], 32);
    EXPECT_EQ(tiling->outShape[2], 128);
    EXPECT_EQ(tiling->outShape[3], 64);
    EXPECT_EQ(tiling->xStrides[0], 2048);
    EXPECT_EQ(tiling->xStrides[1], 64);
    EXPECT_EQ(tiling->xStrides[2], 0);
    EXPECT_EQ(tiling->xStrides[3], 1);
    EXPECT_EQ(tiling->yStrides[0], 0);
    EXPECT_EQ(tiling->yStrides[1], 8192);
    EXPECT_EQ(tiling->yStrides[2], 64);
    EXPECT_EQ(tiling->yStrides[3], 1);
}
