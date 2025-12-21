/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_proto.h
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_graph/repeat_interleave_grad_proto.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"

class RepeatInterleaveGrad : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RepeatInterleaveGrad SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RepeatInterleaveGrad TearDown" << std::endl;
    }
};


TEST_F(RepeatInterleaveGrad, RepeatInterleaveGrad_infershape_test)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("RepeatInterleaveGrad")->infer_shape;

    gert::StorageShape xShape = {{64}, {}};
    gert::StorageShape wShape = {{1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}
                      })
                      .InputShapes({&xShape, &wShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}