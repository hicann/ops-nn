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

#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

class TestDualLevelQuantBatchMatmulInferShape : public testing::Test
{
};

TEST_F(TestDualLevelQuantBatchMatmulInferShape, InferShape)
{
    gert::InferShapeContext nullContext;
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DualLevelQuantBatchMatmul")->infer_shape;
    ASSERT_EQ(inferShapeFunc(&nullContext), ge::GRAPH_SUCCESS);
}