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
 * \file test_multi_add_rms_norm_dynamic_quant_infershape.cpp
 * \brief
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/multi_add_rms_norm_dynamic_quant_proto.h"
#include "runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"

class MultiAddRmsNormDynamicQuant : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MultiAddRmsNormDynamicQuant Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MultiAddRmsNormDynamicQuant Proto Test TearDown" << std::endl;
    }
};

TEST_F(MultiAddRmsNormDynamicQuant, MultiAddRmsNormDynamicQuant_infershape_case_dynamic)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MultiAddRmsNormDynamicQuant"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MultiAddRmsNormDynamicQuant")->infer_shape;
    if (infer_shape_func != nullptr) {
        gert::StorageShape input_shape = {{24, 1, 11264}, {24, 1, 11264}};
        gert::StorageShape gamma_shape = {
            {
                11264,
            },
            {
                11264,
            }};
        gert::StorageShape out_shape = {{24, 1, 11264}, {24, 1, 11264}};
        gert::StorageShape reduce_shape = {{24, 1}, {24, 1}};

        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(5, 6)
                          .IrInstanceNum({1, 1, 1, 1, 1, 1})
                          .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                          .OutputShapes({&out_shape, &out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                          .NodeAttrs({
                              {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                          })
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();

        auto context = holder.GetContext<gert::InferShapeContext>();
        EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

        EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), 24);
        EXPECT_EQ(context->GetOutputShape(0)->GetDim(1), 1);
        EXPECT_EQ(context->GetOutputShape(0)->GetDim(2), 11264);
    }
}
