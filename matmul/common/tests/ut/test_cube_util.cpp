/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "../../op_host/op_api/cube_util.cpp"

using namespace Ops::NN;
using namespace op;

class CubeUtilPromoteTypeTest : public testing::Test {};

// CalcUseFp16PromoteType
TEST_F(CubeUtilPromoteTypeTest, UseFp16_Fp16)
{
    EXPECT_EQ(CalcUseFp16PromoteType(DataType::DT_FLOAT16), DataType::DT_FLOAT16);
}

TEST_F(CubeUtilPromoteTypeTest, UseFp16_Fp32)
{
    EXPECT_EQ(CalcUseFp16PromoteType(DataType::DT_FLOAT), DataType::DT_FLOAT16);
}

TEST_F(CubeUtilPromoteTypeTest, UseFp16_Bf16)
{
    EXPECT_EQ(CalcUseFp16PromoteType(DataType::DT_BF16), DataType::DT_FLOAT16);
}

TEST_F(CubeUtilPromoteTypeTest, UseFp16_Hif8)
{
    EXPECT_EQ(CalcUseFp16PromoteType(DataType::DT_HIFLOAT8), DataType::DT_HIFLOAT8);
}

TEST_F(CubeUtilPromoteTypeTest, UseFp16_Fp8E4m3fn)
{
    EXPECT_EQ(CalcUseFp16PromoteType(DataType::DT_FLOAT8_E4M3FN), DataType::DT_FLOAT8_E4M3FN);
}

TEST_F(CubeUtilPromoteTypeTest, UseFp16_Unsupported)
{
    EXPECT_EQ(CalcUseFp16PromoteType(DataType::DT_INT8), DataType::DT_UNDEFINED);
}

// CalcUseHf32PromoteType
TEST_F(CubeUtilPromoteTypeTest, UseHf32_Fp16)
{
    EXPECT_EQ(CalcUseHf32PromoteType(DataType::DT_FLOAT16), DataType::DT_FLOAT16);
}

TEST_F(CubeUtilPromoteTypeTest, UseHf32_Bf16)
{
    EXPECT_EQ(CalcUseHf32PromoteType(DataType::DT_BF16), DataType::DT_BF16);
}

TEST_F(CubeUtilPromoteTypeTest, UseHf32_Hif8)
{
    EXPECT_EQ(CalcUseHf32PromoteType(DataType::DT_HIFLOAT8), DataType::DT_HIFLOAT8);
}

TEST_F(CubeUtilPromoteTypeTest, UseHf32_Fp8E4m3fn)
{
    EXPECT_EQ(CalcUseHf32PromoteType(DataType::DT_FLOAT8_E4M3FN), DataType::DT_FLOAT8_E4M3FN);
}

TEST_F(CubeUtilPromoteTypeTest, UseHf32_Fp32)
{
    EXPECT_EQ(CalcUseHf32PromoteType(DataType::DT_FLOAT), DataType::DT_FLOAT);
}

TEST_F(CubeUtilPromoteTypeTest, UseHf32_Unsupported)
{
    EXPECT_EQ(CalcUseHf32PromoteType(DataType::DT_INT8), DataType::DT_UNDEFINED);
}

// CalcAllowFp32DownPrecisionPromoteType
TEST_F(CubeUtilPromoteTypeTest, AllowFp32Down_Fp16)
{
    EXPECT_EQ(CalcAllowFp32DownPrecisionPromoteType(DataType::DT_FLOAT16), DataType::DT_FLOAT16);
}

TEST_F(CubeUtilPromoteTypeTest, AllowFp32Down_Bf16)
{
    EXPECT_EQ(CalcAllowFp32DownPrecisionPromoteType(DataType::DT_BF16), DataType::DT_BF16);
}

TEST_F(CubeUtilPromoteTypeTest, AllowFp32Down_Hif8)
{
    EXPECT_EQ(CalcAllowFp32DownPrecisionPromoteType(DataType::DT_HIFLOAT8), DataType::DT_HIFLOAT8);
}

TEST_F(CubeUtilPromoteTypeTest, AllowFp32Down_Fp8E4m3fn)
{
    EXPECT_EQ(CalcAllowFp32DownPrecisionPromoteType(DataType::DT_FLOAT8_E4M3FN), DataType::DT_FLOAT8_E4M3FN);
}

TEST_F(CubeUtilPromoteTypeTest, AllowFp32Down_Unsupported)
{
    EXPECT_EQ(CalcAllowFp32DownPrecisionPromoteType(DataType::DT_INT8), DataType::DT_UNDEFINED);
}

// CalcKeepDtypePromoteType
TEST_F(CubeUtilPromoteTypeTest, KeepDtype_Fp16)
{
    EXPECT_EQ(CalcKeepDtypePromoteType(DataType::DT_FLOAT16), DataType::DT_FLOAT16);
}

TEST_F(CubeUtilPromoteTypeTest, KeepDtype_Bf16)
{
    EXPECT_EQ(CalcKeepDtypePromoteType(DataType::DT_BF16), DataType::DT_BF16);
}

TEST_F(CubeUtilPromoteTypeTest, KeepDtype_Fp32)
{
    EXPECT_EQ(CalcKeepDtypePromoteType(DataType::DT_FLOAT), DataType::DT_FLOAT);
}

TEST_F(CubeUtilPromoteTypeTest, KeepDtype_Hif8)
{
    EXPECT_EQ(CalcKeepDtypePromoteType(DataType::DT_HIFLOAT8), DataType::DT_HIFLOAT8);
}

TEST_F(CubeUtilPromoteTypeTest, KeepDtype_Fp8E4m3fn)
{
    EXPECT_EQ(CalcKeepDtypePromoteType(DataType::DT_FLOAT8_E4M3FN), DataType::DT_FLOAT8_E4M3FN);
}

TEST_F(CubeUtilPromoteTypeTest, KeepDtype_Unsupported)
{
    EXPECT_EQ(CalcKeepDtypePromoteType(DataType::DT_INT8), DataType::DT_INT8);
}

// CalcForceGrpAccForFp32PromoteType
TEST_F(CubeUtilPromoteTypeTest, ForceGrpAcc_Fp32)
{
    EXPECT_EQ(CalcForceGrpAccForFp32PromoteType(DataType::DT_FLOAT), DataType::DT_FLOAT);
}

TEST_F(CubeUtilPromoteTypeTest, ForceGrpAcc_NonFp32)
{
    EXPECT_EQ(CalcForceGrpAccForFp32PromoteType(DataType::DT_FLOAT16), DataType::DT_FLOAT16);
}

TEST_F(CubeUtilPromoteTypeTest, ForceGrpAcc_Bf16)
{
    EXPECT_EQ(CalcForceGrpAccForFp32PromoteType(DataType::DT_BF16), DataType::DT_BF16);
}
