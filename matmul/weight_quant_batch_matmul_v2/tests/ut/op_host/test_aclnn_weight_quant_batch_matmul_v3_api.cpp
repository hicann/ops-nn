/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include <thread>
#include <vector>

#include "../../../op_host/op_api/aclnn_weight_quant_batch_matmul_v3.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

enum ContiguousType
{
    CONTIGUOUS,
    TRANSPOSE_LAST_TWO_DIMS,
    NOT_CONTIGUOUS
};

struct WeightQuantBatchMatmulV3TestParam {
    string caseName;
    vector<int64_t> x;
    vector<int64_t> weight;
    vector<int64_t> antiquantScale;
    vector<int64_t> antiquantOffsetoptional;
    vector<int64_t> quantScaleOptional;
    vector<int64_t> quantOffsetOptional;
    vector<int64_t> biasOptional;
    int antiquantGroupSize;
    int innerPrecise;
    vector<int64_t> y;
    aclDataType xType;
    aclDataType weightType;
    aclDataType antiquantScaleType;
    aclDataType antiquantOffsetType;
    aclDataType quantScaleOptionalType;
    aclDataType quantOffsetOptionalType;
    aclDataType biasOptionalType;
    aclDataType yType;
    aclFormat xFormat;
    aclFormat weightFormat;
    bool isAntiquantOffsetoptionalNotNull;
    bool isQuantScaleOptionalNotNull;
    bool isQuantOffsetOptionalNotNull;
    bool isBiasOptionalNotNull;
    aclnnStatus expectRet;
    ContiguousType ctgsX = CONTIGUOUS;
    ContiguousType ctgsWeight = CONTIGUOUS;
    ContiguousType ctgsAntiquantScale = CONTIGUOUS;
    ContiguousType ctgsAntiquantOffsetOptional = CONTIGUOUS;
    ContiguousType ctgsQuantScaleOptional = CONTIGUOUS;
    ContiguousType ctgsQuantOffsetOptional = CONTIGUOUS;
    ContiguousType ctgsBiasOptional = CONTIGUOUS;
    ContiguousType ctgsY = CONTIGUOUS;
};

class l2_weight_quant_batch_matmul_v3_test_910B2 : public testing::TestWithParam<WeightQuantBatchMatmulV3TestParam>
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v3_test_910B2 SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v3_test_910B2 TearDown" << endl;
    }
};

class l2_weight_quant_batch_matmul_v3_test_310P : public testing::TestWithParam<WeightQuantBatchMatmulV3TestParam>
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v3_test_310P SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v3_test_310P TearDown" << endl;
    }
};

class l2_weight_quant_batch_matmul_v3_test_910 : public testing::TestWithParam<WeightQuantBatchMatmulV3TestParam>
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v3_test_910 SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_weight_quant_batch_matmul_v3_test_910 TearDown" << endl;
    }
};

static vector<int64_t> CreateFractalNZShape(const vector<int64_t>& viewShape, const aclDataType& dtype)
{
    if (viewShape.size() < 2) { // 维度小于2维无法转Nz
        throw invalid_argument(
            "size of viewShape must >= 2 when create fractalNz shape, actual is " + viewShape.size());
    }
    if (dtype != ACL_INT8) {
        throw invalid_argument(
            "only support dtype int8 when create fractalNz shape, actual is " + static_cast<int32_t>(dtype));
    }
    // nd转Nz时会把viewShape最后2维拆成4维
    vector<int64_t> storageShape(viewShape.size() + 2, 0);
    // nd转Nz时除了viewShape最后2维，其余维度保持不变
    copy(viewShape.begin(), viewShape.end() - 2, storageShape.begin());
    // nd: (b1, b2, x, y)
    // Nz: (b1, b2, ceil(y / 32), ceil(x / 16), 16, 32) int8
    int64_t x = viewShape[viewShape.size() - 2];
    int64_t y = viewShape[viewShape.size() - 1];
    storageShape[storageShape.size() - 4] = ceil(static_cast<double>(y) / 32); // 倒数第4维度，32为新的倒数第1维大小
    storageShape[storageShape.size() - 3] = ceil(static_cast<double>(x) / 16); // 倒数第3维度，16为新的倒数第2维大小
    storageShape[storageShape.size() - 2] = 16;                                // 倒数第2维度，固定为16
    storageShape[storageShape.size() - 1] = 32; // 倒数第1维度，需要32B对齐，1个int8占用1B，所以对齐需要32个
    return storageShape;
}

static vector<int64_t> CreateContiguousStride(const vector<int64_t>& viewShape)
{
    vector<int64_t> strides(viewShape.size(), 1);
    if (strides.size() < 2) { // 把每个维度累乘起来，小于2维时无变化
        return strides;
    }
    // 从倒数第2维开始更新每个维度实际在存储上的偏移，倒数第1维偏移为1不用算
    for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
        strides[i] = viewShape[i] * strides[i + 1];
    }
    return strides;
}

static aclTensor* CreateAclTensor(
    const vector<int64_t>& viewShape, const aclDataType& dtype, const aclFormat& format, const vector<int64_t>& stride,
    int64_t offset, const vector<int64_t>& originalShape)
{
    vector<int64_t> storageShape;
    if (format == ACL_FORMAT_FRACTAL_NZ) {
        storageShape = CreateFractalNZShape(viewShape, dtype);
    } else {
        storageShape = originalShape;
    }

    auto tensor = aclCreateTensor(
        viewShape.data(), viewShape.size(), dtype, stride.data(), offset, format, storageShape.data(),
        storageShape.size(), nullptr);
    if (originalShape.size() == 3) { // 创建3维的tensor
        tensor->SetOriginalShape(op::Shape{originalShape[0], originalShape[1], originalShape[2]});
    } else if (originalShape.size() == 2) { // 创建2维的tensor
        tensor->SetOriginalShape(op::Shape{originalShape[0], originalShape[1]});
    } else if (originalShape.size() == 1) { // 创建1维的tensor
        tensor->SetOriginalShape(op::Shape{originalShape[0]});
    } else {
        throw invalid_argument("only support originalShape.size() in (1,2,3), actual is " + originalShape.size());
    }
    return tensor;
}

static aclTensor* CreateTensorDesc(
    const vector<int64_t>& viewShape, const aclDataType& dtype, const aclFormat& format, const ContiguousType& ctgsType)
{
    vector<int64_t> strides;
    vector<int64_t> originalShape;
    int sum = 0;
    switch (ctgsType) {
        case CONTIGUOUS:
            return CreateAclTensor(viewShape, dtype, format, strides, 0, viewShape);
            break;
        case TRANSPOSE_LAST_TWO_DIMS:
            if (viewShape.size() < 2) { // 最后2维已交换的数据的viewShape的维度数量不应小于2
                throw invalid_argument(
                    "size of viewShape must >= 2 when ContiguousType is TRANSPOSE_LAST_TWO_DIMS, actual is " +
                    viewShape.size());
            }
            // viewShape     (y, x)
            // stride        (1, y)
            // storage_shape (x, y)
            strides = CreateContiguousStride(viewShape);
            // 交换步长数组的最后2维
            swap(strides[viewShape.size() - 1], strides[viewShape.size() - 2]);
            originalShape = viewShape;
            // 交换形状的最后2维
            swap(originalShape[viewShape.size() - 1], originalShape[viewShape.size() - 2]);
            return CreateAclTensor(viewShape, dtype, format, strides, 0, originalShape);
            break;
        case NOT_CONTIGUOUS:
            sum = std::accumulate(viewShape.begin(), viewShape.end(), 0);
            if (sum <= 1) {
                throw invalid_argument(
                    "sum of viewShape must > 1 when ContiguousType is NOT_CONTIGUOUS, actual is " + viewShape.size());
            }
            if (format == ACL_FORMAT_FRACTAL_NZ) {
                throw invalid_argument("not support format is FRACTAL_NZ with NOT_CONTIGUOUS");
            }
            strides = CreateContiguousStride(viewShape);
            strides.back() = 2; // 2是为了验证步长数组最后1维不为1的非连续用例
            originalShape = viewShape;
            originalShape.back() *= 2; // 因为步长数组最后1维为2，所以实际在内存上的形状最后1维扩大到2倍
            // viewShape     (x, y)
            // stride        (y, 2)
            // storage_shape (x, 2*y)
            return CreateAclTensor(viewShape, dtype, format, strides, 0, originalShape);
            break;
        default:
            throw invalid_argument("not support ContiguousType " + static_cast<int>(ctgsType));
            break;
    }
}

static void TestOneParamCase(const WeightQuantBatchMatmulV3TestParam& param)
{
    std::cout << "run case start: " << param.caseName << std::endl;
    aclTensor* x = CreateTensorDesc(param.x, param.xType, param.xFormat, param.ctgsX);
    aclTensor* weight = CreateTensorDesc(param.weight, param.weightType, param.weightFormat, param.ctgsWeight);
    aclTensor* antiquantScale =
        CreateTensorDesc(param.antiquantScale, param.antiquantScaleType, param.xFormat, param.ctgsAntiquantScale);
    aclTensor* antiquantOffsetOptional = nullptr;
    if (param.isAntiquantOffsetoptionalNotNull) {
        antiquantOffsetOptional = CreateTensorDesc(
            param.antiquantOffsetoptional, param.antiquantOffsetType, param.xFormat, param.ctgsAntiquantOffsetOptional);
    }
    aclTensor* quantScaleOptional = nullptr;
    if (param.isQuantScaleOptionalNotNull) {
        quantScaleOptional = CreateTensorDesc(
            param.quantScaleOptional, param.quantScaleOptionalType, param.xFormat, param.ctgsQuantScaleOptional);
    }
    aclTensor* quantOffsetOptional = nullptr;
    if (param.isQuantOffsetOptionalNotNull) {
        quantOffsetOptional = CreateTensorDesc(
            param.quantOffsetOptional, param.quantOffsetOptionalType, param.xFormat, param.ctgsQuantOffsetOptional);
    }
    aclTensor* biasOptional = nullptr;
    if (param.isBiasOptionalNotNull) {
        biasOptional =
            CreateTensorDesc(param.biasOptional, param.biasOptionalType, param.xFormat, param.ctgsBiasOptional);
    }
    aclTensor* y = CreateTensorDesc(param.y, param.yType, param.xFormat, param.ctgsY);

    uint64_t workspaceSize = 0U;
    aclOpExecutor* exe = nullptr;
    aclnnStatus aclRet = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
        x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional, biasOptional,
        param.antiquantGroupSize, param.innerPrecise, y, &workspaceSize, &exe);
    EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "run case end:" << param.caseName << std::endl;
    if (param.expectRet == ACLNN_SUCCESS) {
        EXPECT_NE(exe, nullptr);
    }
}

TEST_P(l2_weight_quant_batch_matmul_v3_test_910B2, ascend910B2_generalTest)
{
    WeightQuantBatchMatmulV3TestParam param = GetParam();
    TestOneParamCase(param);
}

TEST_P(l2_weight_quant_batch_matmul_v3_test_310P, ascend310P_generalTest)
{
    WeightQuantBatchMatmulV3TestParam param = GetParam();
    TestOneParamCase(param);
}

TEST_P(l2_weight_quant_batch_matmul_v3_test_910, ascend910_generalTest)
{
    WeightQuantBatchMatmulV3TestParam param = GetParam();
    TestOneParamCase(param);
}

static WeightQuantBatchMatmulV3TestParam casesParamsAscend910B2[] = {
    {"testWeightQuantBatchMatmulV3AntiqauntOffsetDtypeInvalid01",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3AntiqauntOffsetDtypeInvalid02",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_BF16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3BiasDtypeInvalid01",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3BiasDtypeInvalid02",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_BF16,
     ACL_INT8,
     ACL_BF16,
     ACL_BF16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_BF16,
     ACL_BF16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3XDtypeInvalid",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_INT8,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3XFormatInvalid",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_NCHW,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3XDimInvalid",
     {1, 96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3QuantParamInvalid01",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     false,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3QuantParamInvalid02",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FLOAT,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3KInvalid01",
     {96, 11264},
     {11263, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FLOAT,
     ACL_UINT64,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3KInvalid02",
     {96, 65536},
     {65536, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FLOAT,
     ACL_UINT64,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3MInvalid01",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {97, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3NInvalid02",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1663},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3AntiquantGroupSizeInvalid01",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     2,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3AntiquantGroupSizeInvalid02",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     1664,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT4,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3AntiquantShapeInvalid01",
     {96, 11264},
     {11264, 1664},
     {1663},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3AntiquantShapeInvalid02",
     {96, 11264},
     {11264, 1664},
     {1664},
     {1663},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3AntiquantShapeInvalid03",
     {96, 11264},
     {11264, 1664},
     {1, 1663},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3AntiquantShapeInvalid04",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1663},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3AntiquantShapeInvalid05",
     {96, 11264},
     {11264, 1664},
     {52, 1664},
     {52, 1663},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     32,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT4,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3QuantShapeInvalid01",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1663},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3QuantShapeInvalid02",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1664},
     {1663},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3QuantShapeInvalid03",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {2, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3QuantShapeInvalid04",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {2, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulVBiasShapeInvalid01",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1663},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulVBiasShapeInvalid02",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1663},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulKdiff",
     {96, 11264},
     {512, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulshape0",
     {0, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {0, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3Int4ShapeInvalid",
     {96, 11264},
     {11264, 1663},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT4,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3WeightNZInvalid",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3Exceed65536",
     {96, 65536},
     {65536, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID},
    {"testWeightQuantBatchMatmulV3Int4PackInInt32InvalidDimNum",
     {96, 11264},
     {11264},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT32,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     false,
     false,
     true,
     ACLNN_ERR_PARAM_INVALID},
    // 非连续用例
    {"xNotContiguous",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID,
     NOT_CONTIGUOUS},
    {"weightNotContiguous",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     NOT_CONTIGUOUS},
    {"antiquantScaleNotContiguous",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     CONTIGUOUS,
     NOT_CONTIGUOUS},
    {"antiquantOffsetOptionalNotContiguous",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     NOT_CONTIGUOUS},
    {"quantScaleOptionalNotContiguous",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     NOT_CONTIGUOUS},
    {"quantOffsetOptionalNotContiguous",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     NOT_CONTIGUOUS},
    {"biasOptionalNotContiguous",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     NOT_CONTIGUOUS},
    {"yNotContiguous",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     true,
     true,
     true,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     CONTIGUOUS,
     NOT_CONTIGUOUS},
    {"fixpAntiquantscaleDtypeInt64XDtypeError",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_BF16,
     ACL_INT8,
     ACL_INT64,
     ACL_INT32,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     false,
     false,
     false,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     TRANSPOSE_LAST_TWO_DIMS},
    {"fixpAntiquantscaleDtypeInt64AntiquantOffsetDtypeError",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_INT64,
     ACL_FLOAT16,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     false,
     false,
     false,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     TRANSPOSE_LAST_TWO_DIMS},
    {"fixpAntiquantweightDtypeInt4Error",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT4,
     ACL_UINT64,
     ACL_INT32,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     false,
     false,
     false,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     TRANSPOSE_LAST_TWO_DIMS},
    {"fixpAntiquantscaleDtypefp16AntiquantOffsetDtypeInt32Error",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FLOAT16,
     ACL_INT32,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     false,
     false,
     false,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     TRANSPOSE_LAST_TWO_DIMS},
    {"fixpYDtypeInt8Error",
     {96, 11264},
     {11264, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     {1, 1664},
     0,
     0,
     {96, 1664},
     ACL_FLOAT16,
     ACL_INT8,
     ACL_UINT64,
     ACL_INT32,
     ACL_UINT64,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     true,
     false,
     false,
     false,
     ACLNN_ERR_PARAM_INVALID,
     CONTIGUOUS,
     TRANSPOSE_LAST_TWO_DIMS},
};

INSTANTIATE_TEST_SUITE_P(
    Ascend910B2_WeightQuantBatchMatmulV3, l2_weight_quant_batch_matmul_v3_test_910B2,
    testing::ValuesIn(casesParamsAscend910B2));

static void ThreadFunc(
    const WeightQuantBatchMatmulV3TestParam* params, size_t testcase_num, size_t thread_idx, size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const WeightQuantBatchMatmulV3TestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(l2_weight_quant_batch_matmul_v3_test_910B2, ascend910B2_multi_thread)
{
    // 用3个线程测试
    TestMultiThread(
        casesParamsAscend910B2, sizeof(casesParamsAscend910B2) / sizeof(WeightQuantBatchMatmulV3TestParam), 3);
}
