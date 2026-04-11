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
 * \file test_softplus_v2_infershape.cpp
 * \brief SoftplusV2 op_host UT - InferShape unit tests
 */

#include <iostream>
#include <gtest/gtest.h>
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"


class SoftplusV2Infershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftplusV2Infershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SoftplusV2Infershape TearDown" << std::endl;
    }
};

TEST_F(SoftplusV2Infershape, infershape_1d)
{
    gert::StorageShape x_shape = {{1024}, {1024}};
    gert::Shape y_shape = {{1024}};
    gert::Shape expected_output_shape = {1024};

    ge::DataType dtype = ge::DT_FLOAT;

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftplusV2")->infer_shape;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&y_shape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expected_output_shape));
}

