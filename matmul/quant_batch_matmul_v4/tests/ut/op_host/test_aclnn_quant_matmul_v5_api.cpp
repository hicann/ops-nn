/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <float.h>
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_quant_matmul_v5.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

struct QuantBatchMatmulV5TestParam {
    string caseName;
    vector<int64_t> x1Shape;
    vector<int64_t> x2Shape;
    vector<int64_t> scaleShape;
    vector<int64_t> offsetShape;
    vector<int64_t> pertokenScaleShape;
    vector<int64_t> biasShape;
    vector<int64_t> yScaleShape;
    vector<int64_t> x1OffsetShape;
    vector<int64_t> yOffsetShape;
    vector<int64_t> outShape;
    vector<int64_t> x1_stride;
    vector<int64_t> x2_stride;
    aclDataType x1Type;
    aclDataType x2Type;
    aclDataType scaleType;
    aclDataType offsetType;
    aclDataType pertokenType;
    aclDataType biasType;
    aclDataType yScaleType;
    aclDataType x1OffsetType;
    aclDataType yOffsetType;
    aclDataType outType;
    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    // out
    aclnnStatus expect_ret;
    aclFormat x2Format = aclFormat::ACL_FORMAT_ND;
};

class l2_QuantBatchMatmulV5_test_910_95 : public testing::TestWithParam<QuantBatchMatmulV5TestParam> {
protected:
    static void SetUpTestCase() { cout << "l2_QuantBatchMatmulV5_test_910_95 SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_QuantBatchMatmulV5_test_910_95 TearDown" << endl; }
};

class l2_QuantBatchMatmulV5_test_910B2 : public testing::TestWithParam<QuantBatchMatmulV5TestParam> {
protected:
    static void SetUpTestCase() { cout << "l2_QuantBatchMatmulV5_test_910B2 SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_QuantBatchMatmulV5_test_910B2 TearDown" << endl; }
};

vector<int64_t> GenerateNZShape(const vector<int64_t> &viewShape, const aclDataType &dtype) {
    if (viewShape.size() < 2) {
        throw invalid_argument("size of viewShape must >= 2 when create fractalNz shape, actual is " +
                               viewShape.size());
    }
    if (dtype != ACL_FLOAT8_E4M3FN) {
        throw invalid_argument("only support dtype float8_e4m3fn when create fractalNz shape, actual is " +
                               static_cast<int32_t>(dtype));
    }
    int64_t c0 = 32;
    vector<int64_t> storageShape(viewShape.size() + 2, 0);
    copy(viewShape.begin(), viewShape.end() - 2, storageShape.begin());
    // nd: (b1, b2, x, y)
    // Nz: (b1, b2, ceil(y / c0), ceil(x / 16), 16, c0)
    int64_t x = viewShape[viewShape.size() - 2];
    int64_t y = viewShape[viewShape.size() - 1];
    storageShape[storageShape.size() - 4] = ceil(static_cast<double>(y) / c0);
    storageShape[storageShape.size() - 3] = ceil(static_cast<double>(x) / 16);
    storageShape[storageShape.size() - 2] = 16;
    storageShape[storageShape.size() - 1] = c0;
    return storageShape;
}

static void TestOneParamCase(const QuantBatchMatmulV5TestParam &param)
{
    std::cout << "run case " << param.caseName << std::endl;
    vector<int64_t> stride;
    void* deviceAddr = nullptr;
    aclTensor* x1Desc = nullptr;
    if (param.x1Shape.size() > 0) {
        x1Desc = aclCreateTensor(param.x1Shape.data(), param.x1Shape.size(), param.x1Type, param.x1_stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.x1Shape.data(), param.x1Shape.size(), deviceAddr);
    }
    aclTensor* x2Desc = nullptr;
    if (param.x2Shape.size() > 0) {
        aclFormat format = aclFormat::ACL_FORMAT_ND;
        vector<int64_t> storageShape;
        if (param.x2Format == aclFormat::ACL_FORMAT_FRACTAL_NZ_C0_32 || param.x2Format == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
            format = param.x2Format;
            storageShape = GenerateNZShape(param.x2Shape, param.x1Type);
        } else {
            storageShape = param.x2Shape;
        }

        x2Desc = aclCreateTensor(param.x2Shape.data(), param.x2Shape.size(), param.x2Type, param.x2_stride.data(), 0, format, storageShape.data(), storageShape.size(), deviceAddr);
    }
    aclTensor* scaleDesc = nullptr;
    if (param.scaleShape.size() > 0) {
        scaleDesc = aclCreateTensor(param.scaleShape.data(), param.scaleShape.size(), param.scaleType, stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.scaleShape.data(), param.scaleShape.size(), deviceAddr);
    }
    aclTensor* offsetDesc = nullptr;
    if (param.offsetShape.size() > 0) {
        offsetDesc = aclCreateTensor(param.offsetShape.data(), param.offsetShape.size(), param.offsetType, stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.offsetShape.data(), param.offsetShape.size(), deviceAddr);
    }
    aclTensor* pertokenDesc = nullptr;
    if (param.pertokenScaleShape.size() > 0) {
        pertokenDesc = aclCreateTensor(param.pertokenScaleShape.data(), param.pertokenScaleShape.size(), param.pertokenType, stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.pertokenScaleShape.data(), param.pertokenScaleShape.size(), deviceAddr);
    }
    aclTensor* biasDesc = nullptr;
    if (param.biasShape.size() > 0) {
        biasDesc = aclCreateTensor(param.biasShape.data(), param.biasShape.size(), param.biasType, stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.biasShape.data(), param.biasShape.size(), deviceAddr);
    }
    aclTensor* outDesc = nullptr;
    if (param.outShape.size() > 0) {
        outDesc = aclCreateTensor(param.outShape.data(), param.outShape.size(), param.outType, stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.outShape.data(), param.outShape.size(), deviceAddr);
    }
    aclTensor* yScaleDesc = nullptr;
    if (param.yScaleShape.size() > 0) {
        yScaleDesc = aclCreateTensor(param.yScaleShape.data(), param.yScaleShape.size(), param.yScaleType, stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.yScaleShape.data(), param.yScaleShape.size(), deviceAddr);
    }
    aclTensor* x1OffsetDesc = nullptr;
    if (param.x1OffsetShape.size() > 0) {
        x1OffsetDesc = aclCreateTensor(param.x1OffsetShape.data(), param.x1OffsetShape.size(), param.x1OffsetType, stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.x1OffsetShape.data(), param.x1OffsetShape.size(), deviceAddr);
    }
    aclTensor* yOffsetDesc = nullptr;
    if (param.yOffsetShape.size() > 0) {
        yOffsetDesc = aclCreateTensor(param.yOffsetShape.data(), param.yOffsetShape.size(), param.yOffsetType, stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.yOffsetShape.data(), param.yOffsetShape.size(), deviceAddr);
    }

    aclnnStatus aclRet;
    auto ut = OP_API_UT(aclnnQuantMatmulV5,
                        INPUT(x1Desc, x2Desc, pertokenDesc, scaleDesc, yScaleDesc, x1OffsetDesc, offsetDesc, yOffsetDesc, biasDesc, param.transposeX1,
                              param.transposeX2, param.groupSize),
                        OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    ASSERT_EQ(aclRet, param.expect_ret);
    std::cout << "end case " << param.caseName << std::endl;
}

TEST_P(l2_QuantBatchMatmulV5_test_910_95, ascend910_95_generalTest)
{
    QuantBatchMatmulV5TestParam param = GetParam();
    TestOneParamCase(param);
}

TEST_P(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_generalTest)
{
    QuantBatchMatmulV5TestParam param = GetParam();
    TestOneParamCase(param);
}

static QuantBatchMatmulV5TestParam casesParams910B2[] = {
    {"ascend910B2_test_quant_bmm_v5_A8W8FP16_pertensor_batch0", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8FP16_pertensor_batch1", {2, 16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {2, 16, 16}, {512, 32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8FP16_pertensor_batch2", {2, 2, 16, 32}, {32, 16}, {1}, {}, {}, {}, {}, {}, {}, {2, 2, 16, 16}, {1024, 512, 32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8BF16_pertensor", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8C8_pertensor", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8C8_pertensor_offset", {16, 32}, {32, 16}, {1}, {1}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8C8_perchannel_offset", {16, 32}, {32, 16}, {16}, {1}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8FP16_perchannel", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8BF16_perchannel", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8FP16_pertoken_pertensor", {16, 32}, {32, 16}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8BF16_pertoken_pertensor", {16, 32}, {32, 16}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910B2_test_quant_bmm_v5_A8W8BF16_pertoken_perchannel", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
};

static QuantBatchMatmulV5TestParam casesParams910_95[] = {
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertensor_batch0", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertensor_batch1", {2, 16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {2, 16, 16}, {512, 32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertensor_batch2", {2, 2, 16, 32}, {32, 16}, {1}, {}, {}, {}, {}, {}, {}, {2, 2, 16, 16}, {1024, 512, 32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8BF16_pertensor", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_offset", {16, 32}, {32, 16}, {1}, {1}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_perchannel_offset", {16, 32}, {32, 16}, {16}, {1}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_perchannel", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8BF16_perchannel", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_pertensor", {16, 32}, {32, 16}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8BF16_pertoken_pertensor", {16, 32}, {32, 16}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8BF16_pertoken_perchannel", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP16_pertensor", {32, 16}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_pertensor", {16, 32}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP16_pertensor", {32, 16}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP16_pertensor", {32, 16}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3BF16_pertensor", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2BF16_pertensor", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2BF16_pertensor", {32, 16}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3BF16_pertensor", {32, 16}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_pertensor", {32, 16}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP32_pertensor", {32, 16}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP32_pertensor", {32, 16}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP32_pertensor", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP8E4M3_pertensor", {32, 16}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT8_E4M3FN, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP16_pertensor", {32, 16}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8BF16_pertensor", {16, 32}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP32_pertensor", {16, 32}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8HIF8_pertensor", {32, 16}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_HIFLOAT8, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP16_doublescale", {32, 16}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_doublescale_batch0", {16, 32}, {16, 32}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_doublescale_batch1", {2, 16, 32}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {2, 16, 16}, {512, 32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_doublescale_batch2", {2, 2, 16, 32}, {32, 16}, {1}, {}, {1}, {}, {}, {}, {}, {2, 2, 16, 16}, {1024, 512, 32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP16_doublescale", {32, 16}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP16_doublescale", {32, 16}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3BF16_doublescale", {16, 32}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2BF16_doublescale", {16, 32}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2BF16_doublescale", {32, 16}, {16, 32}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3BF16_doublescale", {32, 16}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_doublescale", {32, 16}, {16, 32}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP32_doublescale", {32, 16}, {16, 32}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP32_doublescale", {32, 16}, {16, 32}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP32_doublescale", {16, 32}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP16_doublescale", {32, 16}, {16, 32}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8BF16_doublescale", {16, 32}, {16, 32}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP32_doublescale", {16, 32}, {16, 32}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8HIF8_doublescale", {32, 16}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E2M1FP4E2M1FP16_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E2M1, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E2M1FP4E1M2FP16_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E2M1, ACL_FLOAT4_E1M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E1M2FP16_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E1M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP16_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E2M1FP4E2M1BF16_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E2M1, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E2M1FP4E1M2BF16_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E2M1, ACL_FLOAT4_E1M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E1M2BF16_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E1M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1BF16_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E2M1FP4E2M1FP32_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E2M1, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E2M1FP4E1M2FP32_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E2M1, ACL_FLOAT4_E1M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E1M2FP32_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E1M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},

    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_even_number_error", {16, 63}, {16, 63}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {63, 1}, {63, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 4295032864, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_shape_error", {15, 64}, {15, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 4295032864, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_scale_shape_error", {16, 66}, {16, 66}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {66, 1}, {66, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 4295032864, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_scale_even_number_error", {16, 96}, {16, 96}, {16, 3}, {}, {16, 3}, {16}, {}, {}, {}, {16, 16}, {96, 1}, {96, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 4295032864, ACLNN_ERR_PARAM_INVALID},

    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP32_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP32_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP32_mxfp8", {6, 74}, {9, 74}, {9, 2, 2}, {}, {6, 2, 2}, {9}, {}, {}, {}, {6, 9}, {74, 1}, {74, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP16_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP16_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP16_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3BF16_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2BF16_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3BF16_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2BF16_mxfp8", {16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_mxfp8_batch", {2, 16, 64}, {16, 64}, {16, 1, 2}, {}, {16, 1, 2}, {16}, {}, {}, {}, {2, 16, 16}, {1024, 64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_mxfp8_false_false", {16, 31}, {31, 99}, {1, 99, 2}, {}, {16, 1, 2}, {99}, {}, {}, {}, {16, 99}, {31, 1}, {99, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP16_mxfp8_true_true", {64, 12}, {88, 64}, {88, 1, 2}, {}, {1, 12, 2}, {}, {}, {}, {}, {12, 88}, {12, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3BF16_mxfp8_true_false", {192, 122}, {192, 16}, {3, 16, 2}, {}, {3, 122, 2}, {16}, {}, {}, {}, {122, 16}, {122, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 32, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertoken_pertensor_output_dtype_error", {16, 32}, {32, 16}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_x1_dtype_error", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_HIFLOAT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_x2_dtype_error", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_scale_dtype_error", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_bias_dtype_error", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_illegal_bias_with_batch_error", {2, 2, 16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {1024, 512,32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_offset_dtype_error", {16, 32}, {32, 16}, {1}, {1}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_INT32, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_pertoken_dtype_error", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_HIFLOAT8, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_x1_min_dim_error", {32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_x2_min_dim_error", {16, 32}, {16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_k_error", {16, 32}, {33, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_perchannel_input_batch_dim_error", {3, 16, 32}, {2, 32, 16}, {16}, {16}, {}, {16}, {}, {}, {}, {2, 16, 16}, {512, 32, 1}, {512, 16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_x1_max_dim_error", {1, 1, 1, 1, 1, 16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {1, 1, 1, 1, 1, 16, 16}, {512, 512, 512, 512, 512, 32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_x2_max_dim_error", {16, 32}, {1, 1, 1, 1, 1, 32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {1, 1, 1, 1, 1, 16, 16}, {32, 1}, {512, 512, 512, 512, 512, 16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_perchannel_inner_axis_error", {16, 70000}, {70000, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {70000, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_scale_dim_error", {16, 32}, {32, 16}, {16, 16}, {}, {16}, {16}, {}, {}, {}, {16, 32}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_scale_shape_error", {16, 32}, {32, 16}, {2}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_offset_dim_error", {16, 32}, {32, 16}, {1}, {16, 16}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_offset_shape_error", {16, 32}, {32, 16}, {1}, {2}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_output_dim_error", {16, 32}, {32, 16}, {1}, {16}, {}, {16}, {}, {}, {}, {16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_bias_max_dim_error", {16, 32}, {32, 16}, {1}, {16}, {}, {1, 1, 1, 16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_bias_three_dim_error", {16, 32}, {32, 16}, {1}, {16}, {}, {1, 1, 16}, {}, {}, {}, {1, 16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_bias_batch_dim_infered_error", {1, 16, 32}, {2, 32, 16}, {1}, {16}, {}, {3, 1, 16}, {}, {}, {}, {2, 16, 16}, {512, 32, 1}, {512, 16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_bias_second_dim_error", {1, 16, 32}, {2, 32, 16}, {16}, {1}, {}, {2, 2, 16}, {}, {}, {}, {2, 16, 16}, {512, 32, 1}, {512, 16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_bias_third_dim_error", {1, 16, 32}, {2, 32, 16}, {1}, {16}, {}, {2, 1, 10}, {}, {}, {}, {2, 16, 16}, {512, 32, 1}, {512, 16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_bias_shape_error", {16, 32}, {32, 16}, {1}, {16}, {}, {15}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_output_shape_m_error", {16, 32}, {32, 16}, {1}, {16}, {}, {16}, {}, {}, {}, {10, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8C8_pertensor_output_shape_n_error", {16, 32}, {32, 16}, {1}, {16}, {}, {16}, {}, {}, {}, {16, 10}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_pertoken_dim_error", {16, 32}, {32, 16}, {16}, {16}, {16, 16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_perchannel_pertoken_shape_error", {16, 32}, {32, 16}, {16}, {16}, {15}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP8E4M3HIF8FP16_pertensor_input_dtype_error", {32, 16}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_pertensor_bias_dtype_error", {16, 32}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP8E5M2_pertensor_output_dtype_error", {32, 16}, {16, 32}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT8_E5M2, true, true, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP16_doublescale_bias_dtype_error", {16, 32}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP16_doublescale_scale_dtype_error", {32, 16}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP16_doublescale_pertoken_dtype_error", {32, 16}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_HIF8HIF8C8_doublescale_output_dtype_error", {16, 32}, {32, 16}, {1}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_HIF8HIF8HIF8_scale_shape_error", {16, 32}, {32, 16}, {2}, {}, {1}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_HIFLOAT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_HIF8HIF8HIF8_pertoken_shape_error", {16, 32}, {32, 16}, {1}, {}, {2}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_HIFLOAT8, false, false, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP8FP4FP16_mxfp4_pergroup_x1_dtype_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4FP8FP16_mxfp4_pergroup_x2_dtype_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_INT8, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4FP4FP16_mxfp4_pergroup_scale_dtype_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4FP4FP16_mxfp4_pergroup_pertoken_dtype_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4FP4FP16_mxfp4_pergroup_bias_dtype_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4FP4C8_mxfp4_pergroup_output_dtype_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT8, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_pergroup_scale_dim_error", {16, 64}, {16, 64}, {16}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_pergroup_scale_shape_m_error", {16, 64}, {16, 64}, {15, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_pergroup_scale_shape_k_error", {16, 64}, {16, 64}, {16, 3}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_pergroup_pertoken_dim_error", {16, 64}, {16, 64}, {16, 2}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_pergroup_pertoken_shape_m_error", {16, 64}, {16, 64}, {16, 2}, {}, {15, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1FP32_mxfp4_pergroup_pertoken_shape_k_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 3}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1BF16_mxfp4_pergroup_transpose_true_true_error", {64, 16}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {64, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, true, 32, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_FP4E1M2FP4E2M1BF16_mxfp4_pergroup_transpose_false_false_error", {16, 64}, {64, 16}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {16, 1}, ACL_FLOAT4_E1M2, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 32, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_HIFLOAT8FP8E4M3FP32_mxfp8_x1_dtype_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_mxfp8_x2_dtype_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_INT8, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_yScale_not_null_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {16}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_INT8, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_NULLPTR},
    // {"ascend910_95_test_quant_bmm_v5_x1Offset_not_null_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {16}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_INT8, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_NULLPTR},
    // {"ascend910_95_test_quant_bmm_v5_yOffset_not_null_error", {16, 64}, {16, 64}, {16, 2}, {}, {16, 2}, {16}, {}, {}, {16}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 32, ACLNN_ERR_PARAM_NULLPTR},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertensor_scale_bfloat16_bias_none", {16, 32}, {32, 16}, {1}, {}, {}, {}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_perchannel_scale_bfloat16_bias_int32", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertensor_scale_bfloat16_bias_float", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_perchannel_scale_bfloat16_bias_bfloat16", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertensor_scale_float_bias_none", {16, 32}, {32, 16}, {1}, {}, {}, {}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_perchannel_scale_float_bias_int32", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertensor_scale_float_bias_float", {16, 32}, {32, 16}, {1}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_perchannel_scale_float_bias_bfloat16", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_perchannel_scale_float_bias_bfloat16_groupsize0", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8CB16_perchannel_scale_float_bias_bfloat16_groupsizeerror", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 1, ACLNN_ERR_PARAM_INVALID},
    {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertoken_scale_bfloat16_bias_none", {16, 32}, {32, 16}, {1}, {}, {16}, {}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertoken_scale_bfloat16_bias_int32", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertoken_scale_bfloat16_bias_float", {16, 32}, {32, 16}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertoken_scale_bfloat16_bias_bfloat16", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertoken_scale_float_bias_float", {16, 32}, {32, 16}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8CB16_pertoken_scale_float_bias_bfloat16", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_scale_float_bias_float", {16, 32}, {32, 16}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_A8W8FP16_pertoken_scale_float_bias_float16", {16, 32}, {32, 16}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_INT8, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, false, 0, ACLNN_SUCCESS},

    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_pertoken", {16, 64}, {16, 64}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP32_pertoken", {16, 64}, {16, 64}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP32_pertoken", {16, 64}, {16, 64}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP32_pertoken", {16, 64}, {16, 64}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP16_pertoken", {16, 64}, {16, 64}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_pertoken", {16, 64}, {16, 64}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP16_pertoken", {16, 64}, {16, 64}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP16_pertoken", {16, 64}, {16, 64}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3BF16_pertoken", {16, 64}, {16, 64}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2BF16_pertoken", {16, 64}, {16, 64}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3BF16_pertoken", {16, 64}, {16, 64}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2BF16_pertoken", {16, 64}, {16, 64}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_pertoken_batch", {2, 16, 64}, {16, 64}, {1}, {}, {16}, {16}, {}, {}, {}, {2, 16, 16}, {1024, 64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP16_pertoken", {32, 16}, {16, 32}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8BF16_pertoken", {16, 32}, {16, 32}, {1}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP32_pertoken", {16, 32}, {16, 32}, {16}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_pertensor_perchannel", {16, 64}, {16, 64}, {16}, {}, {1}, {}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_pertensor_perchannel", {16, 64}, {16, 64}, {16}, {}, {1}, {}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP16_pertensor_perchannel", {16, 64}, {16, 64}, {16}, {}, {1}, {}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2BF16_pertensor_perchannel", {16, 64}, {16, 64}, {16}, {}, {1}, {}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP32_pertensor_perchannel", {16, 32}, {16, 32}, {16}, {}, {1}, {}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP16_perchannel", {32, 16}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_perchannel", {16, 32}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP16_perchannel", {32, 16}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP16_perchannel", {32, 16}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3BF16_perchannel", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2BF16_perchannel", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2BF16_perchannel", {32, 16}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3BF16_perchannel", {32, 16}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_perchannel", {32, 16}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP32_perchannel", {32, 16}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP32_perchannel", {32, 16}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP32_perchannel", {16, 32}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {16, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP8E4M3_perchannel", {32, 16}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT8_E4M3FN, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP16_perchannel", {32, 16}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8BF16_perchannel", {16, 32}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8FP32_perchannel", {16, 32}, {16, 32}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {32, 1}, {32, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIF8HIF8HIF8_perchannel", {32, 16}, {32, 16}, {16}, {}, {}, {16}, {}, {}, {}, {16, 16}, {16, 1}, {16, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_UINT64, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_HIFLOAT8, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_pertoken_dim_error", {16, 64}, {16, 64}, {16}, {}, {16, 2}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_scale_dim_error", {16, 64}, {16, 64}, {16, 2}, {}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_x2Offset_not_null_error", {16, 64}, {16, 64}, {16}, {16}, {16}, {16}, {}, {}, {}, {16, 16}, {64, 1}, {64, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 0, ACLNN_ERR_PARAM_INVALID},

    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP32_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP32_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP32_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2FP16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3FP16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2FP16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3BF16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E5M2BF16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E4M3BF16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E5M2FP8E5M2BF16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIFP8HIFP8FP32_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIFP8HIFP8FP16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIFP8HIFP8BF16_perblock", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIFP8HIFP8BF16_perblock_group_input_0", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 0, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIFP8HIFP8BF16_perblock_wrong_group", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 16, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_HIFP8HIFP8BF16_perblock_cant_infer_group", {255, 255}, {255, 127}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {255, 127}, {255, 1}, {127, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 0, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_HIFP8HIFP8BF16_perblock_cant_infer_group_scale_0", {255, 255}, {255, 127}, {0, 1}, {}, {0, 2}, {}, {}, {}, {}, {255, 127}, {255, 1}, {127, 1}, ACL_HIFLOAT8, ACL_HIFLOAT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 0, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_perblock_batch", {2, 256, 256}, {256, 128}, {2, 1}, {}, {2, 2, 2}, {}, {}, {}, {}, {2, 256, 128}, {65536, 256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_perblock_batch2", {10, 2048, 1024}, {1024, 512}, {8, 4}, {}, {10, 16, 8}, {}, {}, {}, {}, {10, 2048, 512}, {2097152, 1024, 1}, {512, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_perblock_batch3", {10, 2048, 1024}, {512, 1024}, {4, 8}, {}, {10, 16, 8}, {}, {}, {}, {}, {10, 2048, 512}, {2097152, 1024, 1}, {1024, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, true, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3FP8E4M3FP32_perblock_batch4", {2048, 1024}, {10, 1024, 512}, {10, 4, 8}, {}, {16, 8}, {}, {}, {}, {}, {10, 2048, 512}, {1024, 1}, {524288, 1, 1024}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, false, false, 549764202624, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_HIFLOAT8FP8E4M3FP32_perblock_x1_dtype_error", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_HIFLOAT8, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_FP8E4M3INT8FP32_perblock_x2_dtype_error", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_perblock_yScale_not_null_error", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {128}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_ERR_PARAM_NULLPTR},
    // {"ascend910_95_test_quant_bmm_v5_perblock_x1Offset_not_null_error", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {128}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_ERR_PARAM_NULLPTR},
    // {"ascend910_95_test_quant_bmm_v5_perblock_yOffset_not_null_error", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {128}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, true, false, 549764202624, ACLNN_ERR_PARAM_NULLPTR},
    // {"ascend910_95_test_quant_bmm_v5_perblock_out_dtype_error", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT8_E4M3FN, true, false, 549764202624, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_perblock_bias_not_null_error", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {128}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_perblock_groupsize_error", {256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {256, 128}, {256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_perblock_x1_x1scale_dim_error", {2, 256, 256}, {256, 128}, {2, 1}, {}, {2, 2}, {}, {}, {}, {}, {2, 256, 128}, {65536, 256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_perblock_x1_x1scale_batch_error", {2, 256, 256}, {256, 128}, {2, 1}, {}, {4, 2, 2}, {}, {}, {}, {}, {2, 256, 128}, {65536, 256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_perblock_x1scale_shape_error", {2, 256, 256}, {256, 128}, {2, 1}, {}, {2, 2, 1}, {}, {}, {}, {}, {2, 256, 128}, {65536, 256, 1}, {128, 1}, ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, ACL_BF16, true, false, 549764202624, ACLNN_ERR_PARAM_INVALID},
    // // A8W4 case
    // {"ascend910_95_test_quant_bmm_v5_A8W4_Pertensor_Pergroup", {16, 128}, {128, 128}, {128, 4}, {}, {}, {}, {1, 128}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_Bias", {16, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {1, 128}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_No_Bias", {16, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_SUCCESS},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_True_TransX1", {128, 16}, {128, 128}, {128, 4}, {}, {16, 4}, {}, {}, {}, {}, {16, 128}, {16, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, true, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_False_TransX2", {16, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, false, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_With_OffsetX2", {16, 128}, {128, 128}, {128, 4}, {128, 4}, {16, 4}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_NULLPTR},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_X1_With_Batch", {2, 16, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {}, {}, {}, {}, {2, 16, 128}, {2048, 128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_X2_With_Batch", {16, 128}, {2, 128, 128}, {128, 4}, {}, {16, 4}, {}, {}, {}, {}, {2, 16, 128}, {128, 1}, {16384, 128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_X2_Scale_Error_Shape", {16, 128}, {128, 128}, {4, 128, 4}, {}, {16, 4}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_X1_Scale_Error_Shape", {16, 128}, {128, 128}, {128, 4}, {}, {16, 16, 4}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_A8W4_Y_Scale_Error_Shape", {16, 128}, {128, 128}, {128, 4}, {}, {}, {}, {1, 128, 16}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_A8W4_Y_Error_Shape", {16, 128}, {128, 128}, {128, 4}, {}, {}, {}, {1, 128}, {}, {}, {2, 16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_A8W4_With_Bias", {16, 128}, {128, 128}, {128, 4}, {}, {}, {1, 128}, {1, 128}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_A8W4_WithOut_Y_Scale", {16, 128}, {128, 128}, {128, 4}, {}, {}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_With_Y_Scale", {16, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {}, {1, 128}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_ERROR_Scale_Dtype", {16, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_BF16, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_Error_K_Dim", {16, 128}, {128, 1280}, {128, 40}, {}, {16, 4}, {}, {}, {}, {}, {16, 128}, {128, 1}, {1280, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_Error_M_Dim", {-1, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {}, {}, {}, {}, {-1, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_Error_Scale_Dim", {16, 128}, {128, 128}, {128, 2}, {}, {128, 2}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_Error_GroupSize", {16, 128}, {128, 128}, {128, 2}, {}, {128, 2}, {}, {}, {}, {}, {16, 128}, {128, 1}, {128, 1}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 64, ACLNN_ERR_PARAM_INVALID},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_NZ_C0_32", {16, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {1, 128}, {}, {}, {}, {16, 128}, {}, {}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_SUCCESS, ACL_FORMAT_FRACTAL_NZ_C0_32},
    // {"ascend910_95_test_quant_bmm_v5_MxA8W4_NZ_C0_32_X2_No_TransX2_ERROR", {16, 128}, {128, 128}, {128, 4}, {}, {16, 4}, {1, 128}, {}, {}, {}, {16, 128}, {}, {}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E8M0, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, false, 32, ACLNN_ERR_PARAM_INVALID, ACL_FORMAT_FRACTAL_NZ_C0_32},
    // {"ascend910_95_test_quant_bmm_v5_A8W4_NZ_TransX2_ERROR", {16, 128}, {128, 128}, {128, 4}, {}, {}, {}, {1, 128}, {}, {}, {16, 128}, {}, {}, ACL_FLOAT8_E4M3FN, ACL_FLOAT4_E2M1, ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16, ACL_UINT64, ACL_BF16, ACL_BF16, ACL_BF16, false, true, 32, ACLNN_ERR_PARAM_INVALID, ACL_FORMAT_FRACTAL_NZ},
};

INSTANTIATE_TEST_SUITE_P(Ascend910_95_QuantBatchMatmulV5, l2_QuantBatchMatmulV5_test_910_95, testing::ValuesIn(casesParams910_95));
INSTANTIATE_TEST_SUITE_P(Ascend910B2_QuantBatchMatmulV5, l2_QuantBatchMatmulV5_test_910B2, testing::ValuesIn(casesParams910B2));

static void ThreadFunc(const QuantBatchMatmulV5TestParam *params, size_t testcase_num, size_t thread_idx,
                       size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const QuantBatchMatmulV5TestParam *params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

// TEST_F(l2_QuantBatchMatmulV5_test_910_95, ascend910_95_multi_thread)
// {
//     // 用3个线程测试
//     TestMultiThread(casesParams910_95, sizeof(casesParams910_95) / sizeof(QuantBatchMatmulV5TestParam), 3);
// }

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_multi_thread)
{
    // 用3个线程测试
    TestMultiThread(casesParams910B2, sizeof(casesParams910B2) / sizeof(QuantBatchMatmulV5TestParam), 3);
}

// TEST_F(l2_QuantBatchMatmulV5_test_910_95, ascend910_95_case_torch_int8_in_pertensor)
// {
//     TensorDesc x1_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND, {32, 1}, 0, {12 * 32}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({32, 64}, ACL_INT8, ACL_FORMAT_ND, {64, 1}, 0, {32 * 64}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({16, 64}, ACL_INT8, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV5_test_910_95, ascend910_95_case_torch_fp8e4m3_in_fp8_out_pertensor)
// {
//     TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {32, 1}, 0, {12 * 32}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {64, 1}, 0, {32 * 64}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV5_test_910_95, ascend910_95_case_torch_fp8e5m2_in_fp8_out_pertensor)
// {
//     TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND, {32, 1}, 0, {12 * 32}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND, {64, 1}, 0, {32 * 64}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({16, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV5_test_910_95, ascend910_95_case_torch_hif8_in_hif8_out_pertensor)
// {
//     TensorDesc x1_desc = TensorDesc({16, 32}, ACL_HIFLOAT8, ACL_FORMAT_ND, {32, 1}, 0, {12 * 32}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({32, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND, {64, 1}, 0, {32 * 64}).ValueRange(-1, 1);
//     TensorDesc scale_desc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({16, 64}, ACL_HIFLOAT8, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

// TEST_F(l2_QuantBatchMatmulV5_test_910_95, ascend910_95_case_torch_fp8e4m3_in_fp32_out_double_scale)
// {
//     TensorDesc x1_desc = TensorDesc({16, 32}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {32, 1}, 0, {12 * 32}).ValueRange(-1, 1);
//     TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, {64, 1}, 0, {32 * 64}).ValueRange(-1, 1);
//     TensorDesc scale_x1_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc scale_x2_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
//     TensorDesc out_desc = TensorDesc({16, 64}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, scale_x1_desc, scale_x2_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
//                         OUTPUT(out_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_SUCCESS);
// }

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_0)
{
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_1)
{
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({280, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, true, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_2)
{
    // A4W4场景无pertoken_scale时不支持transposeX1=true
    TensorDesc x1_desc = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, true, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_3)
{
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_4)
{
    // A4W4场景scale_dim需要和展开成int4后的n_dim匹配
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({35}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_5)
{
    // A4W4场景不支持INT8输出
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_6)
{
    // A4W4场景无pertoken_scale时暂时只支持2维的x2输入
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({2, 8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_7)
{
    // A4W4场景无pertoken_scale时暂时只支持ND的x2输入
    TensorDesc x1_desc = TensorDesc({16, 32}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({32, 16}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {1, 1}, 0, {10, 10, 16, 32}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({16}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_8)
{
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({32}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_9)
{
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({64}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_10)
{
    // x1和x2的数据类型必须一致
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_11)
{
    // x1 int4输入时内轴需要为偶数
    TensorDesc x1_desc = TensorDesc({64, 7}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({7, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({8}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_12)
{
    // x2 int4输入时内轴需要为偶数
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 13}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({13}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 13}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_13)
{
    // x1 int4输入无pertoken_scale时bias只支持1维
    TensorDesc x1_desc = TensorDesc({2, 64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 14}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({14}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({2, 1, 14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_14)
{
    // 新增支持组合：A4W4_pertoken_scale_out_fp16
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, pertoken_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_15)
{
    // 新增支持组合：A4W4_pertoken_scale_out_bf16
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, pertoken_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_16)
{
    //A4W4有pertoken_scale不支持transposex1=True
    TensorDesc x1_desc = TensorDesc({8, 64}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, pertoken_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, true, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_17)
{
    // A4W4 pertoken_scale 不支持3维x2输入
    TensorDesc x1_desc = TensorDesc({64, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({2, 8, 35}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({2, 64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, pertoken_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_18)
{
    // A4W4 pertoken_scale 不支持NZ x2输入
    TensorDesc x1_desc = TensorDesc({128, 128}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({128, 128}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ, {128, 1}, 0, {2, 8, 16, 64}).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc out_desc = TensorDesc({128, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, pertoken_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_19)
{
    // A4W4 pertoken_scale 不支持3维bias
    TensorDesc x1_desc = TensorDesc({2, 64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 14}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc bias_desc = TensorDesc({2, 1, 14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, pertoken_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_20)
{
    // A4W4 pertoken_scale bf16 bias不支持fp16输出
    TensorDesc x1_desc = TensorDesc({2, 64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 14}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc bias_desc = TensorDesc({2, 1, 14}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, pertoken_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_21)
{
    // A4W4 pertoken_scale fp16 bias不支持bf16输出
    TensorDesc x1_desc = TensorDesc({2, 64, 8}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 14}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc pertoken_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc bias_desc = TensorDesc({2, 1, 14}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({2, 64, 14}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, pertoken_desc, scale_desc, nullptr, nullptr, nullptr, nullptr, bias_desc, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_22)
{
    TensorDesc x1_desc = TensorDesc({64, 8}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8, 280}, ACL_INT4, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({280}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({64, 280}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, nullptr, scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_1)
{
    // A8W4 msd方案 支持fp16输出
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_2)
{
    // A8W4 msd方案 支持bf16输出
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_3)
{
    // A8W4 msd方案 x1scale_desc shape 必须是[m,1]
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,100}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_4)
{
    // A8W4 msd方案 x1scale_desc shape 必须是[m,1]
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({100,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_5)
{
    // A8W4 msd方案 x1_desc shpae [1] 必须等于 x2_desc shape [0]
    TensorDesc x1_desc = TensorDesc({1, 8191}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_6)
{
    // A8W4 msd方案 x2scale_desc shape 必须为 [k // 256, n]
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({31,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_7)
{
    // A8W4 msd方案 x2scale_desc shape 必须为 [k // 256, n]
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1025}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_8)
{
    // A8W4 msd方案 out_desc shape 必须是 [m,n]
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({100, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_9)
{
    // A8W4 msd方案 out_desc shape 必须是 [m,n]
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1025}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_10)
{
    // A8W4 msd方案 x1scale_desc 类型必须是FLOAT
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_11)
{
    // A8W4 msd方案 x2scale_desc 类型必须是 UINT64
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_12)
{
    // A8W4 msd方案 yoffset_desc 类型必须是 FLOAT
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_13)
{
    // A8W4 msd方案 out_desc类型必须是 FLOAT16 / BF16
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 128}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_14)
{
    // A8W4 msd方案 x2 shape的n维度没有65535限制
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 10000}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,80000}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({80000}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 80000}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A8W4_MSD_15)
{
    // A8W4 msd方案 x2 shape的x2NDim 必须大于1
    TensorDesc x1_desc = TensorDesc({1, 8192}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({8192, 0}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1); // INT32 = INT4 * 8, 128 = 1024 / 8
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({32,1024}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc yoffset_desc = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 1024}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnQuantMatmulV5, INPUT(x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, nullptr, yoffset_desc, nullptr, false, false, 256),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_PERGROUP_1)
{
    // A4W4 pergroup方案, m轴=1
    TensorDesc x1_desc = TensorDesc({1, 5120}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({20480, 5120}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1scale_desc = TensorDesc({1,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({20,20480}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc x2offset_desc = TensorDesc({20,20480}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({1, 20480}, ACL_BF16, ACL_FORMAT_ND);
    int64_t groupSize = 256L | (1L << 16) | (1L << 32); // groupSizeK | (groupSizeN << 16) | (groupSizeM << 32)
    auto ut = OP_API_UT(
        aclnnQuantMatmulV5,
        INPUT(
            x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, x2offset_desc, nullptr, nullptr, false,
            true, groupSize),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_PERGROUP_2)
{
    // A4W4 pergroup方案, m轴=7
    TensorDesc x1_desc = TensorDesc({7, 5120}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({20480, 5120}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1scale_desc = TensorDesc({7,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({20,20480}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc x2offset_desc = TensorDesc({20,20480}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({7, 20480}, ACL_BF16, ACL_FORMAT_ND);
    int64_t groupSize = 256L | (1L << 16) | (1L << 32); // groupSizeK | (groupSizeN << 16) | (groupSizeM << 32)
    auto ut = OP_API_UT(
        aclnnQuantMatmulV5,
        INPUT(
            x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, x2offset_desc, nullptr, nullptr, false,
            true, groupSize),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_PERGROUP_3)
{
    // A4W4 pergroup方案, m轴=8
    TensorDesc x1_desc = TensorDesc({8, 5120}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({20480, 5120}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1scale_desc = TensorDesc({8,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({20,20480}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc x2offset_desc = TensorDesc({20,20480}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({8, 20480}, ACL_BF16, ACL_FORMAT_ND);
    int64_t groupSize = 256L | (1L << 16) | (1L << 32); // groupSizeK | (groupSizeN << 16) | (groupSizeM << 32)
    auto ut = OP_API_UT(
        aclnnQuantMatmulV5,
        INPUT(
            x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, x2offset_desc, nullptr, nullptr, false,
            true, groupSize),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_test_case_A4W4_PERGROUP_4)
{
    // A4W4 pergroup方案, m轴=9
    TensorDesc x1_desc = TensorDesc({9, 5120}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({20480, 5120}, ACL_INT4, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1scale_desc = TensorDesc({9,1}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2scale_desc = TensorDesc({20,20480}, ACL_UINT64, ACL_FORMAT_ND);
    TensorDesc x2offset_desc = TensorDesc({20,20480}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({9, 20480}, ACL_BF16, ACL_FORMAT_ND);
    int64_t groupSize = 256L | (1L << 16) | (1L << 32); // groupSizeK | (groupSizeN << 16) | (groupSizeM << 32)
    auto ut = OP_API_UT(
        aclnnQuantMatmulV5,
        INPUT(
            x1_desc, x2_desc, x1scale_desc, x2scale_desc, nullptr, nullptr, x2offset_desc, nullptr, nullptr, false,
            true, groupSize),
        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
