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
#include <iostream>
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/lamb_next_m_v_with_decay_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;

class LambNextMVWithDecayProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(LambNextMVWithDecayProtoTest, lamb_next_mv_with_decay_case_2d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMVWithDecay")->infer_shape;
    gert::Shape inShape = {-1, -1};
    gert::Shape outShape = {};
    gert::Shape expShape = {-1, -1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape, &outShape, &outShape, &outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto od0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*od0), Ops::Base::ToString(expShape));
    auto od1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*od1), Ops::Base::ToString(expShape));
    auto od2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    ASSERT_EQ(Ops::Base::ToString(*od2), Ops::Base::ToString(expShape));
    auto od3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);
    ASSERT_EQ(Ops::Base::ToString(*od3), Ops::Base::ToString(expShape));
}

TEST_F(LambNextMVWithDecayProtoTest, lamb_next_mv_with_decay_case_fp16_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambNextMVWithDecay")->infer_shape;
    gert::Shape inShape = {-1, -1, -1, -1};
    gert::Shape outShape = {};
    gert::Shape expShape = {-1, -1, -1, -1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(13, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape, &outShape, &outShape, &outShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto od0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*od0), Ops::Base::ToString(expShape));
    auto od1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*od1), Ops::Base::ToString(expShape));
    auto od2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    ASSERT_EQ(Ops::Base::ToString(*od2), Ops::Base::ToString(expShape));
    auto od3 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(3);
    ASSERT_EQ(Ops::Base::ToString(*od3), Ops::Base::ToString(expShape));
}

