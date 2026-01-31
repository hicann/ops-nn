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
#include <iostream>
#include "infershape_test_util.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "runtime/infer_shape_range_context.h"
#include "test_cube_util.h"

namespace ge {

class TestQuantBatchMatmulInplaceAddInferShape : public testing::Test {
    protected:
        static void SetUpTestCase() {
            std::cout << "TestQuantBatchMatmulInplaceAddInferShape Proto Test SetUp" << std::endl;
        }

        static void TearDownTestCase() {
            std::cout << "TestQuantBatchMatmulInplaceAddInferShape Proto Test TearDown" << std::endl;
        }
};


TEST_F(TestQuantBatchMatmulInplaceAddInferShape, QuantBatchMatmulInplaceAdd_InferShapeCheck) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulInplaceAdd"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulInplaceAdd")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x1 = {{64, 128}, {64, 128}};
    gert::StorageShape x2 = {{64, 256}, {64, 256}};
    gert::StorageShape x1_scale = {{1, 128, 2}, {1, 128, 2}};
    gert::StorageShape x2_scale = {{1, 256, 2}, {1, 256, 2}};
    gert::StorageShape y = {{64, 256}, {64, 256}};
    gert::StorageShape outputShape = {{}, {}};
    int64_t groupSize = 4295032864;
    auto context_holder = gert::InferShapeContextFaker()
        .NodeIoNum(5, 1)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&x1, &x2, &x2_scale, &y, &x1_scale})
        .OutputShapes({&outputShape})
        .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                    {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
        .Build();
    ASSERT_EQ(infer_shape_func(context_holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = context_holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[64, 256]");
}

}