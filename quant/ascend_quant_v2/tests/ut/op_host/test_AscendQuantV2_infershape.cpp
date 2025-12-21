/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/ascend_quant_v2_proto.h"

using namespace ge;
using namespace op;

class AscendQuantV2Proto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AscendQuantV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AscendQuantV2Proto TearDown" << std::endl;
    }
};

TEST_F(AscendQuantV2Proto, AscendQuantV2_proto_0)
{
    // set input info
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendQuantV2")->infer_shape;

    gert::StorageShape xShape = {{-1, 64}, {}};
    gert::StorageShape sShape = {{64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs(
                          {{"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("round")},
                           {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .InputShapes({&xShape, &sShape, &sShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-1, 64]");
}