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
 * \file test_single_layer_lstm_infershape.cpp
 * \brief SingleLayerLstm InferShape unit tests.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "op_impl_registry.h"

namespace {
constexpr int64_t T = 8;
constexpr int64_t B = 16;
constexpr int64_t I = 32;
constexpr int64_t H = 64;
constexpr int64_t G4H = 4 * H;
} // namespace

class SingleLayerLstmInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SingleLayerLstmInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SingleLayerLstmInfershape TearDown" << std::endl; }
};

namespace {
void CheckOutputDataTypes(ge::DataType weightType)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("SingleLayerLstm");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    ge::DataType outputs[8] = {ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                               ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED};
    auto holder = gert::InferDataTypeContextFaker()
                      .IrInputNum(5)
                      .NodeIoNum(5, 8)
                      .NodeInputTd(0, weightType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, weightType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, weightType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, weightType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, weightType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&weightType, &weightType, &weightType, &weightType, &weightType})
                      .OutputDataTypes({&outputs[0], &outputs[1], &outputs[2], &outputs[3], &outputs[4], &outputs[5],
                                        &outputs[6], &outputs[7]})
                      .Build();
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), weightType);
    for (size_t k = 1; k < 8; ++k) {
        EXPECT_EQ(context->GetOutputDataType(k), weightType) << "saved output index " << k;
    }
}
} // namespace

TEST_F(SingleLayerLstmInfershape, single_layer_lstm_inferdtype_fp32) { CheckOutputDataTypes(ge::DT_FLOAT); }

TEST_F(SingleLayerLstmInfershape, single_layer_lstm_inferdtype_fp16_all_outputs)
{
    CheckOutputDataTypes(ge::DT_FLOAT16);
}

TEST_F(SingleLayerLstmInfershape, single_layer_lstm_inferdtype_bf16_all_outputs) { CheckOutputDataTypes(ge::DT_BF16); }

/* All eight outputs are [T, B, H]. H is read off init_h, not divided out of w's second dimension:
 * inferring it from w would make the same number arrive from two directions, and one of them could be
 * wrong without contradicting the other. */
TEST_F(SingleLayerLstmInfershape, single_layer_lstm_infershape_all_outputs_are_t_b_h)
{
    gert::InfershapeContextPara para("SingleLayerLstm",
                                     {
                                         {{{T, B, I}, {T, B, I}}, ge::DT_FLOAT, ge::FORMAT_ND},       // x
                                         {{{I + H, G4H}, {I + H, G4H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // w
                                         {{{G4H}, {G4H}}, ge::DT_FLOAT, ge::FORMAT_ND},               // b
                                         {{{B, H}, {B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},             // init_h
                                         {{{B, H}, {B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},             // init_c
                                     },
                                     {
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // y
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output_h
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // output_c
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // i
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // j
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // f
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // o
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // tanhc
                                     },
                                     {
                                         {"direction", Ops::NN::AnyValue::CreateFrom<std::string>("UNIDIRECTIONAL")},
                                         {"gate_order", Ops::NN::AnyValue::CreateFrom<std::string>("ifjo")},
                                     });
    std::vector<std::vector<int64_t>> expect(8, {T, B, H});
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

/* A rank-2 x is refused rather than reinterpreted. The kernel indexes x as [T, B, I] with one stride
 * computed from those three extents, so a different rank would read past the tensor without any
 * bound being crossed. */
TEST_F(SingleLayerLstmInfershape, single_layer_lstm_infershape_refuses_wrong_x_rank)
{
    gert::InfershapeContextPara para("SingleLayerLstm",
                                     {
                                         {{{B, I}, {B, I}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x, rank 2
                                         {{{I + H, G4H}, {I + H, G4H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{G4H}, {G4H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{B, H}, {B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{B, H}, {B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {"direction", Ops::NN::AnyValue::CreateFrom<std::string>("UNIDIRECTIONAL")},
                                         {"gate_order", Ops::NN::AnyValue::CreateFrom<std::string>("ifjo")},
                                     });
    std::vector<std::vector<int64_t>> expect;
    ExecuteTestCase(para, ge::GRAPH_FAILED, expect);
}
