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

#include "../../../op_graph/ascend_dequant_proto.h"

TEST(AscendDequantProtoTest, RegistersInputsOutputsAndDefaultAttrs)
{
    ge::op::AscendDequant op;

    EXPECT_EQ(op.GetInputsSize(), 2U);
    EXPECT_EQ(op.GetOutputsSize(), 1U);
    EXPECT_FALSE(op.get_attr_sqrt_mode());
    EXPECT_FALSE(op.get_attr_relu_flag());
    EXPECT_EQ(op.get_attr_dtype(), 0);
}

TEST(AscendDequantProtoTest, UpdatesAttrValues)
{
    ge::op::AscendDequant op;

    op.set_attr_sqrt_mode(true);
    op.set_attr_relu_flag(true);
    op.set_attr_dtype(1);

    EXPECT_TRUE(op.get_attr_sqrt_mode());
    EXPECT_TRUE(op.get_attr_relu_flag());
    EXPECT_EQ(op.get_attr_dtype(), 1);
}

TEST(AscendDequantProtoTest, UpdatesInputAndOutputDesc)
{
    ge::op::AscendDequant op;

    ge::TensorDesc input_desc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_INT32);
    EXPECT_EQ(op.update_input_desc_x(input_desc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_input_desc_x().GetDataType(), ge::DT_INT32);

    ge::TensorDesc scale_desc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_UINT64);
    EXPECT_EQ(op.update_input_desc_deq_scale(scale_desc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_input_desc_deq_scale().GetDataType(), ge::DT_UINT64);

    ge::TensorDesc output_desc(ge::Shape({2, 3}), ge::FORMAT_ND, ge::DT_FLOAT16);
    EXPECT_EQ(op.update_output_desc_y(output_desc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(op.get_output_desc_y().GetDataType(), ge::DT_FLOAT16);
}
