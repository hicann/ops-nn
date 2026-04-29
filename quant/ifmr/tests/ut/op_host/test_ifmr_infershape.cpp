/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_ifmr_infershape.cpp
 * \brief
 */
#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "array_ops.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class IfmrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ifmr test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ifmr test TearDown" << std::endl; }
};

TEST_F(IfmrTest, ifmr_test_case_infershape)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("IFMR")->infer_shape;
    gert::StorageShape dataShape = {{3, 4, 5, 6}, {3, 4, 5, 6}};
    gert::StorageShape dataMinShape = {{}, {}};
    gert::StorageShape dataMaxShape = {{}, {}};
    gert::StorageShape cumsumShape = {{2048}, {2048}};
    gert::StorageShape scaleShape = {{}, {}};
    gert::StorageShape offsetShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&dataShape, &dataMinShape, &dataMaxShape, &cumsumShape})
                      .OutputShapes({&scaleShape, &offsetShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto scale_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    auto offset_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    gert::Shape expected_output_shape = {1};
    ASSERT_EQ(Ops::Base::ToString(*scale_desc), Ops::Base::ToString(expected_output_shape));
    ASSERT_EQ(Ops::Base::ToString(*offset_desc), Ops::Base::ToString(expected_output_shape));
}