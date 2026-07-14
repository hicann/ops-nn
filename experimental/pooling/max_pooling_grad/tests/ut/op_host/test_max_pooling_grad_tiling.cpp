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
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include "tiling_case_executor.h"
#include "experimental/pooling/max_pooling_grad/op_kernel/max_pooling_grad_tiling_data.h"

using namespace ge;

namespace {
// max_pooling_grad 的 dtype 由编译期 DTYPE_DY 宏决定（见 PR #4948 检视 #1），tiling 函数不按 dtype 设置
// tiling key，故 tiling key 恒为默认值 0。对照 average_pooling_grad 的 field-by-field 校验写法。
struct MaxPoolingGradCompileInfo {};

struct TilingCaseParam {
    ge::DataType dtype = ge::DT_FLOAT;
    uint64_t expectTilingKey = 0U;
    uint64_t expectSmallCoreDataNum = 0;
    uint64_t expectBigCoreDataNum = 0;
    uint64_t expectUbPartDataNum = 0;
    uint64_t expectSmallCoreTailDataNum = 0;
    uint64_t expectBigCoreTailDataNum = 0;
    uint64_t expectSmallCoreLoopNum = 0;
    uint64_t expectBigCoreLoopNum = 0;
    uint64_t expectTailBlockNum = 0;
    uint64_t expectLastCoreValidDataNum = 0;
};

// 3 输入 (dy, x, y) + 1 输出 (dx)，形状统一 [2,1,4,6]；platform: coreNum=64, ubSize=256KiB。
static gert::TilingContextPara BuildTilingContext(const TilingCaseParam& param, MaxPoolingGradCompileInfo& compileInfo)
{
    return gert::TilingContextPara("MaxPoolingGrad",
                                   {
                                       {{{2, 1, 4, 6}, {2, 1, 4, 6}}, param.dtype, ge::FORMAT_ND}, // dy
                                       {{{2, 1, 4, 6}, {2, 1, 4, 6}}, param.dtype, ge::FORMAT_ND}, // x
                                       {{{2, 1, 4, 6}, {2, 1, 4, 6}}, param.dtype, ge::FORMAT_ND}, // y
                                   },
                                   {
                                       {{{2, 1, 4, 6}, {2, 1, 4, 6}}, param.dtype, ge::FORMAT_ND}, // dx
                                   },
                                   {}, &compileInfo, 64, 262144, 4096);
}

static void RunSuccessCase(const TilingCaseParam& param)
{
    MaxPoolingGradCompileInfo compileInfo;
    auto tilingContextPara = BuildTilingContext(param, compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, param.expectTilingKey);
    // workspace 为 CompareScalar/Select 高级向量 API 的系统 workspace (GetLibApiWorkSpaceSize)，
    // 取值平台相关，仅校验 tiling 设置了 1 个 workspace 槽位。
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MaxPoolingGradTilingData));

    const auto* tilingData = reinterpret_cast<const MaxPoolingGradTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->smallCoreDataNum, param.expectSmallCoreDataNum);
    EXPECT_EQ(tilingData->bigCoreDataNum, param.expectBigCoreDataNum);
    EXPECT_EQ(tilingData->ubPartDataNum, param.expectUbPartDataNum);
    EXPECT_EQ(tilingData->smallCoreTailDataNum, param.expectSmallCoreTailDataNum);
    EXPECT_EQ(tilingData->bigCoreTailDataNum, param.expectBigCoreTailDataNum);
    EXPECT_EQ(tilingData->smallCoreLoopNum, param.expectSmallCoreLoopNum);
    EXPECT_EQ(tilingData->bigCoreLoopNum, param.expectBigCoreLoopNum);
    EXPECT_EQ(tilingData->tailBlockNum, param.expectTailBlockNum);
    EXPECT_EQ(tilingData->lastCoreValidDataNum, param.expectLastCoreValidDataNum);
}
} // namespace

class MaxPoolingGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolingGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaxPoolingGradTiling TearDown" << std::endl; }
};

// [2,1,4,6]=48 元素，FP32 (typeLength=4)：数据仅占 1 个 256B block，coreNum 被钳到 1（单核）。
TEST_F(MaxPoolingGradTiling, max_pooling_grad_float32_success)
{
    TilingCaseParam param;
    param.dtype = ge::DT_FLOAT;
    param.expectTilingKey = 0U;
    param.expectSmallCoreDataNum = 64;
    param.expectBigCoreDataNum = 128;
    param.expectUbPartDataNum = 8192;
    param.expectSmallCoreTailDataNum = 64;
    param.expectBigCoreTailDataNum = 128;
    param.expectSmallCoreLoopNum = 1;
    param.expectBigCoreLoopNum = 1;
    param.expectTailBlockNum = 0;
    param.expectLastCoreValidDataNum = 48;
    RunSuccessCase(param);
}

// [2,1,4,6]=48 元素，FP16 (typeLength=2)：同样 coreNum 钳到 1（单核）。
TEST_F(MaxPoolingGradTiling, max_pooling_grad_float16_success)
{
    TilingCaseParam param;
    param.dtype = ge::DT_FLOAT16;
    param.expectTilingKey = 0U;
    param.expectSmallCoreDataNum = 128;
    param.expectBigCoreDataNum = 256;
    param.expectUbPartDataNum = 16384;
    param.expectSmallCoreTailDataNum = 128;
    param.expectBigCoreTailDataNum = 256;
    param.expectSmallCoreLoopNum = 1;
    param.expectBigCoreLoopNum = 1;
    param.expectTailBlockNum = 0;
    param.expectLastCoreValidDataNum = 48;
    RunSuccessCase(param);
}
