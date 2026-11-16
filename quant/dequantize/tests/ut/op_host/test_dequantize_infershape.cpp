/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "log/log.h"
#include "ut_op_common.h"
#include "infershape_case_executor.h"

using namespace ge;
using namespace gert;

class DequantizeInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DequantizeInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "DequantizeInfershape TearDown" << std::endl; }
};

static void RunDequantizeCase(const std::vector<std::vector<int64_t>>& inShapes,
                              const std::vector<ge::DataType>& inDtypes,
                              const std::vector<std::vector<int64_t>>& expectOut)
{
    std::vector<InfershapeContextPara::TensorDescription> inputs;
    for (size_t i = 0; i < inShapes.size(); i++) {
        gert::StorageShape s;
        for (int64_t d : inShapes[i]) {
            s.MutableOriginShape().AppendDim(d);
            s.MutableStorageShape().AppendDim(d);
        }
        inputs.emplace_back(s, inDtypes[i], ge::FORMAT_ND);
    }
    std::vector<InfershapeContextPara::TensorDescription> outputs;
    for (size_t i = 0; i < expectOut.size(); i++) {
        outputs.emplace_back(gert::StorageShape(), inDtypes[0], ge::FORMAT_ND);
    }
    InfershapeContextPara para("Dequantize", inputs, outputs);
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOut);
}

// 常规场景
TEST_F(DequantizeInfershape, dequantize_infershape_test_1)
{
    RunDequantizeCase({{4, 8}, {}, {}}, {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}, {{4, 8}});
}

// 部分未知shape（-1）场景：输出透传
TEST_F(DequantizeInfershape, dequantize_infershape_partial_unknown)
{
    RunDequantizeCase({{4, -1}, {}, {}}, {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}, {{4, -1}});
}

// 全未知shape（-1）场景
TEST_F(DequantizeInfershape, dequantize_infershape_all_unknown_dim)
{
    RunDequantizeCase({{-1, -1}, {}, {}}, {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}, {{-1, -1}});
}

// 未知rank（-2）场景
TEST_F(DequantizeInfershape, dequantize_infershape_unknown_rank)
{
    RunDequantizeCase({{
                           -2,
                       },
                       {
                           -2,
                       },
                       {
                           -2,
                       }},
                      {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
                      {{
                          -2,
                      }});
}
