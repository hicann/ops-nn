/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_augru_grad_infershape.cpp
 * \brief DynamicAUGRUGrad InferShape/InferDataType UT
 */

#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_context.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"

// InferDataType注册已按目录规范拆分至op_graph，UT通过registry取注册需引入该编译单元
#include "../../../op_graph/dynamic_augru_grad_graph_infer.cpp"

namespace {
constexpr int64_t T = 4;
constexpr int64_t B = 2;
constexpr int64_t H = 8;
constexpr int64_t I = 16;
constexpr int64_t THREE_H = 3 * H;
} // namespace

class DynamicAUGRUGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DynamicAUGRUGradInferShapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "DynamicAUGRUGradInferShapeTest TearDown" << std::endl; }

    // 构造16个输入shape（x/w_i/w_h/att/y/init_h/h/dy/dh/update/update_att/reset/new/hidden_new/seq_length/mask）
    static void BuildShapes(std::vector<gert::StorageShape>& inputs, std::vector<gert::StorageShape>& outputs,
                            int64_t hDim1Override = -1, bool badXDim = false, bool badSeqLen = false,
                            bool badDy = false, bool badWInputRow = false, int64_t wHiddenLeadingDim = 0)
    {
        inputs.clear();
        outputs.clear();
        gert::StorageShape xShape = badXDim ? gert::StorageShape{{T, B}, {T, B}} :
                                              gert::StorageShape{{T, B, I}, {T, B, I}};
        inputs.push_back(xShape); // x
        int64_t wiRows = badWInputRow ? (I + 1) : I;
        inputs.push_back(gert::StorageShape{{wiRows, THREE_H}, {wiRows, THREE_H}}); // weight_input
        if (wHiddenLeadingDim == 0) {
            inputs.push_back(gert::StorageShape{{H, THREE_H}, {H, THREE_H}}); // weight_hidden
        } else {
            inputs.push_back(
                gert::StorageShape{{wHiddenLeadingDim, H, THREE_H}, {wHiddenLeadingDim, H, THREE_H}}); // weight_hidden
        }
        inputs.push_back(gert::StorageShape{{T, B, H}, {T, B, H}}); // weight_att
        inputs.push_back(gert::StorageShape{{T, B, H}, {T, B, H}}); // y（前向输出占位）
        inputs.push_back(gert::StorageShape{{B, H}, {B, H}});       // init_h
        int64_t hT = (hDim1Override >= 0) ? hDim1Override : T;
        inputs.push_back(gert::StorageShape{{hT, B, H}, {hT, B, H}}); // h
        int64_t dyT = badDy ? (T + 1) : T;
        inputs.push_back(gert::StorageShape{{dyT, B, H}, {dyT, B, H}}); // dy
        inputs.push_back(gert::StorageShape{{B, H}, {B, H}});           // dh
        inputs.push_back(gert::StorageShape{{T, B, H}, {T, B, H}});     // update
        inputs.push_back(gert::StorageShape{{T, B, H}, {T, B, H}});     // update_att
        inputs.push_back(gert::StorageShape{{T, B, H}, {T, B, H}});     // reset
        inputs.push_back(gert::StorageShape{{T, B, H}, {T, B, H}});     // new
        inputs.push_back(gert::StorageShape{{T, B, H}, {T, B, H}});     // hidden_new
        int64_t seq0 = badSeqLen ? (B + 1) : B;
        inputs.push_back(gert::StorageShape{{seq0}, {seq0}}); // seq_length
        inputs.push_back(gert::StorageShape{{T, B}, {T, B}}); // mask（占位）

        for (int i = 0; i < 7; i++) {
            outputs.push_back(gert::StorageShape{{}, {}});
        }
    }

    static std::vector<void*> ToRefVec(std::vector<gert::StorageShape>& shapes)
    {
        std::vector<void*> refs(shapes.size());
        for (size_t i = 0; i < shapes.size(); i++) {
            refs[i] = &shapes[i];
        }
        return refs;
    }
};

// 正常shape：全部输出shape正确
TEST_F(DynamicAUGRUGradInferShapeTest, infer_shape_normal)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::vector<gert::StorageShape> inputs;
    std::vector<gert::StorageShape> outputs;
    BuildShapes(inputs, outputs);
    auto inputRefs = ToRefVec(inputs);
    auto outputRefs = ToRefVec(outputs);

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputRefs)
                      .OutputShapes(outputRefs)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);

    // dw_input [I, 3H]
    std::vector<int64_t> expect0 = {I, THREE_H};
    auto* out0 = context->GetOutputShape(0);
    ASSERT_EQ(out0->GetDimNum(), 2U);
    EXPECT_EQ((*out0)[0], expect0[0]);
    EXPECT_EQ((*out0)[1], expect0[1]);
    // dw_hidden [H, 3H]
    auto* out1 = context->GetOutputShape(1);
    ASSERT_EQ(out1->GetDimNum(), 2U);
    EXPECT_EQ((*out1)[0], H);
    EXPECT_EQ((*out1)[1], THREE_H);
    // db_input/db_hidden [3H]
    auto* out2 = context->GetOutputShape(2);
    ASSERT_EQ(out2->GetDimNum(), 1U);
    EXPECT_EQ((*out2)[0], THREE_H);
    auto* out3 = context->GetOutputShape(3);
    ASSERT_EQ(out3->GetDimNum(), 1U);
    EXPECT_EQ((*out3)[0], THREE_H);
    // dx [T, B, I]
    auto* out4 = context->GetOutputShape(4);
    ASSERT_EQ(out4->GetDimNum(), 3U);
    EXPECT_EQ((*out4)[0], T);
    EXPECT_EQ((*out4)[1], B);
    EXPECT_EQ((*out4)[2], I);
    // dh_prev [B, H]
    auto* out5 = context->GetOutputShape(5);
    ASSERT_EQ(out5->GetDimNum(), 2U);
    EXPECT_EQ((*out5)[0], B);
    EXPECT_EQ((*out5)[1], H);
    // dw_att [T, B]
    auto* out6 = context->GetOutputShape(6);
    ASSERT_EQ(out6->GetDimNum(), 2U);
    EXPECT_EQ((*out6)[0], T);
    EXPECT_EQ((*out6)[1], B);
}

TEST_F(DynamicAUGRUGradInferShapeTest, infer_shape_weight_hidden_leading_singleton)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    std::vector<gert::StorageShape> inputs;
    std::vector<gert::StorageShape> outputs;
    BuildShapes(inputs, outputs, -1, false, false, false, false, 1);
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(ToRefVec(inputs))
                      .OutputShapes(ToRefVec(outputs))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputShape(1)->GetDimNum(), 2U);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(0), H);
    EXPECT_EQ(context->GetOutputShape(1)->GetDim(1), THREE_H);
}

TEST_F(DynamicAUGRUGradInferShapeTest, infer_shape_weight_hidden_non_singleton_leading_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);
    std::vector<gert::StorageShape> inputs;
    std::vector<gert::StorageShape> outputs;
    BuildShapes(inputs, outputs, -1, false, false, false, false, 2);
    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(ToRefVec(inputs))
                      .OutputShapes(ToRefVec(outputs))
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// h与x的T不一致：失败
TEST_F(DynamicAUGRUGradInferShapeTest, infer_shape_h_mismatch_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::vector<gert::StorageShape> inputs;
    std::vector<gert::StorageShape> outputs;
    BuildShapes(inputs, outputs, T + 1);
    auto inputRefs = ToRefVec(inputs);
    auto outputRefs = ToRefVec(outputs);

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputRefs)
                      .OutputShapes(outputRefs)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// x为2维：失败
TEST_F(DynamicAUGRUGradInferShapeTest, infer_shape_x_2dim_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::vector<gert::StorageShape> inputs;
    std::vector<gert::StorageShape> outputs;
    BuildShapes(inputs, outputs, -1, true);
    auto inputRefs = ToRefVec(inputs);
    auto outputRefs = ToRefVec(outputs);

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputRefs)
                      .OutputShapes(outputRefs)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// seq_length长度非[B]：失败
TEST_F(DynamicAUGRUGradInferShapeTest, infer_shape_seq_len_mismatch_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::vector<gert::StorageShape> inputs;
    std::vector<gert::StorageShape> outputs;
    BuildShapes(inputs, outputs, -1, false, true);
    auto inputRefs = ToRefVec(inputs);
    auto outputRefs = ToRefVec(outputs);

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputRefs)
                      .OutputShapes(outputRefs)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// dy的T维不一致：失败
TEST_F(DynamicAUGRUGradInferShapeTest, infer_shape_dy_mismatch_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::vector<gert::StorageShape> inputs;
    std::vector<gert::StorageShape> outputs;
    BuildShapes(inputs, outputs, -1, false, false, true);
    auto inputRefs = ToRefVec(inputs);
    auto outputRefs = ToRefVec(outputs);

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputRefs)
                      .OutputShapes(outputRefs)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// weight_input行数非I：失败
TEST_F(DynamicAUGRUGradInferShapeTest, infer_shape_w_input_row_mismatch_failed)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    std::vector<gert::StorageShape> inputs;
    std::vector<gert::StorageShape> outputs;
    BuildShapes(inputs, outputs, -1, false, false, false, true);
    auto inputRefs = ToRefVec(inputs);
    auto outputRefs = ToRefVec(outputs);

    auto holder = gert::InferShapeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(inputRefs)
                      .OutputShapes(outputRefs)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
}

// dtype推导：全部浮点输出与x一致
TEST_F(DynamicAUGRUGradInferShapeTest, infer_dtype_follow_x)
{
    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad")->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    ge::DataType xDtype = ge::DT_FLOAT16;
    std::vector<ge::DataType> inDtypes(16, ge::DT_FLOAT16);
    inDtypes[14] = ge::DT_INT32; // seq_length
    inDtypes[15] = ge::DT_UINT8; // mask
    std::vector<ge::DataType> outDtypes(7, ge::DT_UNDEFINED);

    std::vector<void*> inRefs(inDtypes.size());
    for (size_t i = 0; i < inDtypes.size(); i++) {
        inRefs[i] = &inDtypes[i];
    }
    std::vector<void*> outRefs(outDtypes.size());
    for (size_t i = 0; i < outDtypes.size(); i++) {
        outRefs[i] = &outDtypes[i];
    }

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputDataTypes(inRefs)
                      .OutputDataTypes(outRefs)
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    for (int i = 0; i < 7; i++) {
        EXPECT_EQ(context->GetOutputDataType(i), xDtype);
    }
}

// 浮点输入dtype与x不一致时应被推导阶段拒绝（proto契约，防GE precision_reduce静默cast）
TEST_F(DynamicAUGRUGradInferShapeTest, infer_dtype_mixed_failed)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicAUGRUGrad");
    ASSERT_NE(opImpl, nullptr);
    auto inferDtypeFunc = opImpl->infer_datatype;
    ASSERT_NE(inferDtypeFunc, nullptr);

    // x为fp16、weight_input为fp32（其余合法）
    std::vector<ge::DataType> inDtypes(16, ge::DT_FLOAT16);
    inDtypes[1] = ge::DT_FLOAT;
    inDtypes[14] = ge::DT_INT32; // seq_length
    inDtypes[15] = ge::DT_UINT8; // mask
    std::vector<ge::DataType> outDtypes(7, ge::DT_UNDEFINED);

    std::vector<void*> inRefs(inDtypes.size());
    for (size_t i = 0; i < inDtypes.size(); i++) {
        inRefs[i] = &inDtypes[i];
    }
    std::vector<void*> outRefs(outDtypes.size());
    for (size_t i = 0; i < outDtypes.size(); i++) {
        outRefs[i] = &outDtypes[i];
    }

    auto holder = gert::InferDataTypeContextFaker()
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1})
                      .InputDataTypes(inRefs)
                      .OutputDataTypes(outRefs)
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_FAILED);
}
