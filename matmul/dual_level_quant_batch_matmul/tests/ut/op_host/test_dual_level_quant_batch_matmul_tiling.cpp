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
#include <stdlib.h>

#include <iostream>
#include <thread>
#include <vector>

#include "log/log.h"

#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"


void TestOneParamCase(int param);
class TestDualLevelQuantBatchMatmulTling
    : public testing::TestWithParam<int>
{
};

TEST_P(TestDualLevelQuantBatchMatmulTling, generalTilingTest)
{
    int param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_CASE_P(A4W4Tiling, TestDualLevelQuantBatchMatmulTling, testing::Values(1));

void TestOneParamCase(int param)
{
    std::string opType("DualLevelQuantBatchMatmul");
    auto kernelHold = gert::KernelRunContextFaker().Build();
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    gert::TilingContext* tilingContext;
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}