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
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "op_impl_registry.h"
#include "log/log.h"
#include "ut_op_common.h"

class AdamInferShapeTest : public testing::Test {};

TEST_F(AdamInferShapeTest, null_context_reports_stage_error_once)
{
    auto implementation = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->infer_shape, nullptr);
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    errors.GetRawErrorMessages();
    EXPECT_EQ(implementation->infer_shape(nullptr), ge::GRAPH_FAILED);
    auto reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, "EZ1014");
}

// 常规场景：10输入同shape，三输出均为该shape
TEST_F(AdamInferShapeTest, infershape_normal)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape t = {{4, 8}, {4, 8}};
    gert::StorageShape s = {{1}, {1}};
    std::vector<gert::StorageShape*> in{&t, &t, &t, &t, &s, &s, &s, &s, &s, &s};
    gert::StorageShape out = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9]})
                      .OutputShapes({&out, &out, &out})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape expectedShape = {4, 8};
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              Ops::Base::ToString(expectedShape));
}

// 部分未知shape（-1）场景：10输入广播结果含-1，in1/2/3同形通过
TEST_F(AdamInferShapeTest, infershape_partial_unknown_dim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape t = {{-1, 8}, {-1, 8}};
    gert::StorageShape s = {{1}, {1}};
    std::vector<gert::StorageShape*> in{&t, &t, &t, &t, &s, &s, &s, &s, &s, &s};
    gert::StorageShape out = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9]})
                      .OutputShapes({&out, &out, &out})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape expectedShape = {-1, 8};
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              Ops::Base::ToString(expectedShape));
}

// 未知rank（-2）场景：全输入-2，广播结果-2，in1/2/3同形通过
TEST_F(AdamInferShapeTest, infershape_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape u = {{-2}, {}};
    gert::StorageShape s = {{-2}, {}};
    std::vector<gert::StorageShape*> in{&u, &u, &u, &u, &s, &s, &s, &s, &s, &s};
    gert::StorageShape out = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9]})
                      .OutputShapes({&out, &out, &out})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape expectedShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              Ops::Base::ToString(expectedShape));
}

// 部分输入未知rank场景：input0为-2而in1/2/3已知时，广播结果-2，跳过校验不误拦
TEST_F(AdamInferShapeTest, infershape_partial_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape u = {{-2}, {}};
    gert::StorageShape t = {{4, 8}, {4, 8}};
    gert::StorageShape s = {{1}, {1}};
    std::vector<gert::StorageShape*> in{&u, &t, &t, &t, &t, &s, &s, &s, &s, &s};
    gert::StorageShape out = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(10, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1})
                      .InputShapes({in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9]})
                      .OutputShapes({&out, &out, &out})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape expectedShape = {-2};
    ASSERT_EQ(Ops::Base::ToString(*holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0)),
              Ops::Base::ToString(expectedShape));
}
