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
#include "cube_utils/cube_math_util.h"
#include <limits>

namespace ops {

// ============================================================================
// Test Suite: FP32 to FP16 Conversion
// ============================================================================

TEST(CubeMathUtilTest, Fp32ToFp16_NormalValue_ConvertsCorrectly) {
    float fp32 = 1.5f;
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // 验证转换结果
    // 注意：FP16 精度有限，使用近似比较
    // 1.5 的 FP16 表示为 0x3E00
    EXPECT_EQ(fp16, 0x3E00);
}

TEST(CubeMathUtilTest, Fp32ToFp16_Zero_ConvertsCorrectly) {
    float fp32 = 0.0f;
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    EXPECT_EQ(fp16, 0x0000);
}

TEST(CubeMathUtilTest, Fp32ToFp16_NegativeZero_ConvertsCorrectly) {
    float fp32 = -0.0f;
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // -0.0 的 FP16 表示为 0x8000
    EXPECT_EQ(fp16, 0x8000);
}

TEST(CubeMathUtilTest, Fp32ToFp16_One_ConvertsCorrectly) {
    float fp32 = 1.0f;
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // 1.0 的 FP16 表示为 0x3C00
    EXPECT_EQ(fp16, 0x3C00);
}

TEST(CubeMathUtilTest, Fp32ToFp16_NegativeOne_ConvertsCorrectly) {
    float fp32 = -1.0f;
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // -1.0 的 FP16 表示为 0xBC00
    EXPECT_EQ(fp16, 0xBC00);
}

TEST(CubeMathUtilTest, Fp32ToFp16_SmallValue_ConvertsCorrectly) {
    float fp32 = 0.0001f;
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // 验证转换后的值不为0（非下溢）
    EXPECT_NE(fp16, 0x0000);
}

TEST(CubeMathUtilTest, Fp32ToFp16_LargeValue_ConvertsCorrectly) {
    float fp32 = 1000.0f;
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // 验证转换后的值不为无穷大
    EXPECT_NE(fp16 & 0x7C00, 0x7C00);
}

TEST(CubeMathUtilTest, Fp32ToFp16_NaN_HandlesCorrectly) {
    float fp32 = std::numeric_limits<float>::quiet_NaN();
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // 验证 NaN 处理：FP16 NaN 指数位为 0x7C00
    EXPECT_EQ(fp16 & 0x7C00, 0x7C00);
}

TEST(CubeMathUtilTest, Fp32ToFp16_Infinity_HandlesCorrectly) {
    float fp32 = std::numeric_limits<float>::infinity();
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // FP16 +Infinity 为 0x7C00
    EXPECT_EQ(fp16, 0x7C00);
}

TEST(CubeMathUtilTest, Fp32ToFp16_NegativeInfinity_HandlesCorrectly) {
    float fp32 = -std::numeric_limits<float>::infinity();
    uint16_t fp16 = Fp32ToFp16(fp32);
    
    // FP16 -Infinity 为 0xFC00
    EXPECT_EQ(fp16, 0xFC00);
}
}  // namespace ops