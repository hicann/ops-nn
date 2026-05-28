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
 * \file test_adaptive_avg_pool2d_infershape.cpp
 * \brief InferShape测试 - AdaptiveAvgPool2d
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

#include "infershape_case_executor.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

using namespace std;
using namespace ge;

class AdaptiveAvgPool2dInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool2dInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool2dInfershape TearDown" << std::endl;
    }
};

// 正常用例: 4D 输入 + 合法 output_size
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_4d_basic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{2, 3, 224, 224}, {2, 3, 224, 224}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({7, 7})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 7, 7}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 正常用例: 3D 输入 + 合法 output_size
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_3d_basic)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{3, 224, 224}, {3, 224, 224}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 正常用例: FP16 数据类型
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{1, 64, 7, 7}, {1, 64, 7, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 64, 1, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 正常用例: BF16 数据类型
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{1, 32, 14, 14}, {1, 32, 14, 14}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({7, 7})},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 32, 7, 7}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 异常用例: output_size 为空 (size=0, !=2)
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_invalid_output_size_empty)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{2, 3, 224, 224}, {2, 3, 224, 224}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 异常用例: output_size 元素数量为 3 (不为 2)
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_invalid_output_size_3_elements)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{2, 3, 224, 224}, {2, 3, 224, 224}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({7, 7, 7})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 异常用例: output_size 元素数量为 1 (不为 2)
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_invalid_output_size_1_element)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{2, 3, 224, 224}, {2, 3, 224, 224}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({7})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 异常用例: 输入为 2D (不支持，仅支持 3D/4D)
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_invalid_input_dims_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{224, 224}, {224, 224}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({7, 7})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// 异常用例: 输入为 5D (不支持，仅支持 3D/4D)
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_invalid_input_dims_5d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{2, 3, 4, 224, 224}, {2, 3, 4, 224, 224}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({7, 7})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

// ======================== 异常用例: Unknown Rank ========================
TEST_F(AdaptiveAvgPool2dInfershape, test_infershape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "AdaptiveAvgPool2d",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({7, 7})},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS);
}

// ======================== InferDataType ========================
TEST_F(AdaptiveAvgPool2dInfershape, test_infer_dtype_fp16)
{
    ge::DataType inputDtype = ge::DT_FLOAT16;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&inputDtype})
                      .OutputDataTypes({&outputDtype})
                      .Build();

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool2d")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(AdaptiveAvgPool2dInfershape, test_infer_dtype_bfloat16)
{
    ge::DataType inputDtype = ge::DT_BF16;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputDataTypes({&inputDtype})
                      .OutputDataTypes({&outputDtype})
                      .Build();

    auto inferDtypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AdaptiveAvgPool2d")->infer_datatype;
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    ASSERT_EQ(inferDtypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
