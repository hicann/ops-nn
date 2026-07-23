/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "hard_swish_grad_tiling_data.h"

namespace HardSwishGradUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "HardSwishGrad";

struct HardSwishGradTestParam {
    std::string caseName;
    std::initializer_list<int64_t> gradShape;
    ge::DataType gradDtype;
    ge::Format gradFormat;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static HardSwishGradTestParam testCases[] = {
    {"fp32_success",
     {7, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     1UL,
     "21 1 5440 ",
     {0},
     64,
     262144,
     4096},
    {"fp16_success",
     {7, 3},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     0UL,
     "21 1 5440 ",
     {0},
     64,
     262144,
     4096},
    {"bf16_success",
     {7, 3},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_BF16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     2UL,
     "21 1 5440 ",
     {0},
     64,
     262144,
     4096},
    {"shape_mismatch",
     {7, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {21},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {},
     64,
     262144,
     4096},
    {"dtype_mismatch",
     {7, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {},
     64,
     262144,
     4096},
    {"unsupported_dtype",
     {7, 3},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {7, 3},
     ge::DT_INT32,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_FAILED,
     0UL,
     "",
     {},
     64,
     262144,
     4096},
};

class HardSwishGradTilingTest : public testing::TestWithParam<HardSwishGradTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "HardSwishGradTilingTest SetUp." << std::endl; }
    static void TearDownTestCase() { std::cout << "HardSwishGradTilingTest TearDown." << std::endl; }
};

struct HardSwishGradCompileInfo {
} compileInfo;

static void TestOneParamCase(const HardSwishGradTestParam& param)
{
    gert::StorageShape gradShape = {param.gradShape, param.gradShape};
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{gradShape, param.gradDtype, param.gradFormat}, {xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.maxAIVNum, param.ubSize, param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces);
}

TEST_P(HardSwishGradTilingTest, tiling_test)
{
    const HardSwishGradTestParam& param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(HardSwishGradTilingTests, HardSwishGradTilingTest, testing::ValuesIn(testCases));

} // namespace HardSwishGradUT
