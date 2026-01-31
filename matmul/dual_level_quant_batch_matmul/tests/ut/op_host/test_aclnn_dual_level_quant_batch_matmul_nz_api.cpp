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
#include "../../../op_host/op_api/aclnn_dual_level_quant_batch_matmul_nz.h"
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

struct DualLevelQuantMatmulNzTestParam {
    string caseName;
    vector<int64_t> x1;
    vector<int64_t> x2;
    vector<int64_t> x1Level0Scale;
    vector<int64_t> x1Level1Scale;
    vector<int64_t> x2Level0Scale;
    vector<int64_t> x2Level1Scale;
    vector<int64_t> biasOptional;
    bool transposeX1;
    bool transposeX2;
    int64_t level0GroupSize;
    int64_t level1GroupSize;
    vector<int64_t> y;
    aclDataType x1Type;
    aclDataType x2Type;
    aclDataType x1Level0ScaleType;
    aclDataType x1Level1ScaleType;
    aclDataType x2Level0ScaleType;
    aclDataType x2Level1ScaleType;
    aclDataType biasOptionalType;
    aclDataType yType;
    aclFormat x1Format;
    aclFormat x2Format;
    bool isBiasOptionalNotNull;
    aclnnStatus expectRet;
    ContiguousType ctgsX1 = CONTIGUOUS;
    ContiguousType ctgsX2 = CONTIGUOUS;
    ContiguousType ctgsX1Level0Scale = CONTIGUOUS;
    ContiguousType ctgsX1Level1Scale = CONTIGUOUS;
    ContiguousType ctgsX2Level0Scale = CONTIGUOUS;
    ContiguousType ctgsX2Level1Scale = CONTIGUOUS;
    ContiguousType ctgsBiasOptional = CONTIGUOUS;
    ContiguousType ctgsY = CONTIGUOUS;
};

class l2_dual_level_quant_batch_matmul_nz_test_950 : public testing::TestWithParam<DualLevelQuantMatmulNzTestParam>
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_dual_level_quant_batch_matmul_nz_test_950 SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_dual_level_quant_batch_matmul_nz_test_950 TearDown" << endl;
    }
};

vector<int64_t> CreateFractalNZShape(const vector<int64_t>& viewShape, const aclDataType& dtype)
{
    if (viewShape.size() < 2) { // 维度小于2维无法转Nz
        throw invalid_argument(
            "size of viewShape must >= 2 when create fractalNz shape, actual is " + viewShape.size());
    }
    if (dtype != ACL_INT8 && dtype != ACL_INT4 && dtype != ACL_INT32) {
        throw invalid_argument(
            "only support dtype int8/int4/int32 when create fractalNz shape, actual is " + static_cast<int32_t>(dtype));
    }

    // nd转Nz时会把viewShape最后2维拆成4维
    vector<int64_t> storageShape(viewShape.size() + 2, 0);
    // nd转Nz时除了viewShape最后2维，其余维度保持不变
    copy(viewShape.begin(), viewShape.end() - 2, storageShape.begin());
    // nd: (b1, b2, x, y)
    // Nz: (b1, b2, ceil(y / 32), ceil(x / 16), 16, 32) int8
    int64_t x = viewShape[viewShape.size() - 2];
    int64_t y = viewShape[viewShape.size() - 1];
    storageShape[storageShape.size() - 4] = ceil(static_cast<double>(y) / 32); // 倒数第4维度，16为新的倒数第1维大小
    storageShape[storageShape.size() - 3] = ceil(static_cast<double>(x) / 16); // 倒数第3维度，16为新的倒数第2维大小
    storageShape[storageShape.size() - 2] = 16;                                // 倒数第2维度，固定为16
    storageShape[storageShape.size() - 1] = 32;                                // 倒数第1维度，需要32B对齐
    return storageShape;
}

vector<int64_t> CreateContiguousStride(const vector<int64_t>& viewShape)
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

aclTensor* CreateAclTensor(
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

aclTensor* CreateTensorDesc(
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
            if (viewShape.size() < 2) { // 最后2维已交换的数据的viewShape的维度尺寸不应小于2
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

static void TestOneParamCase(const DualLevelQuantMatmulNzTestParam& param)
{
    std::cout << "run case start: " << param.caseName << std::endl;
    aclTensor* x1 = CreateTensorDesc(param.x1, param.x1Type, param.x1Format, param.ctgsX1);
    aclTensor* x2 = CreateTensorDesc(param.x2, param.x2Type, param.x2Format, param.ctgsX2);
    aclTensor* x1Level0Scale =
        CreateTensorDesc(param.x1Level0Scale, param.x1Level0ScaleType, param.x1Format, param.ctgsX1Level0Scale);
    aclTensor* x1Level1Scale =
        CreateTensorDesc(param.x1Level1Scale, param.x1Level1ScaleType, ACL_FORMAT_NCL, param.ctgsX1Level1Scale);
    aclTensor* x2Level0Scale =
        CreateTensorDesc(param.x2Level0Scale, param.x2Level0ScaleType, param.x1Format, param.ctgsX2Level0Scale);
    clTensor* x2Level1Scale =
        CreateTensorDesc(param.x2Level1Scale, param.x2Level1ScaleType, ACL_FORMAT_NCL, param.ctgsX2Level1Scale);
    aclTensor* biasOptional = nullptr;
    if (param.isBiasOptionalNotNull) {
        biasOptional =
            CreateTensorDesc(param.biasOptional, param.biasOptionalType, param.xFormat, param.ctgsBiasOptional);
    }
    aclTensor* y = CreateTensorDesc(param.y, param.yType, param.xFormat, param.ctgsY);

    uint64_t workspaceSize = 0U;
    aclOpExecutor* exe = nullptr;
    aclnnStatus aclRet = aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(
        x1, x2, x1Level0Scale, x2Level0Scale, x1Level1Scale, x2Level1Scale, biasOptional,
        param.transposeX1, param.transposeX2, param.level0GroupSize, param.level1GroupSize,
        y, &workspaceSize, &exe);
    EXPECT_EQ(aclRet, param.expectRet);
    std::cout << "run case end:" << param.caseName << std::endl;
    if (param.expectRet == ACLNN_SUCCESS) {
        EXPECT_NE(exe, nullptr);
    }
}

TEST_P(l2_dual_level_quant_batch_matmul_nz_test_950, ascend950_generalTest)
{
    DualLevelQuantMatmulNzTestParam param = GetParam();
    TestOneParamCase(param);
}

static DualLevelQuantMatmulNzTestParam casesParamsAscend950[] = {
    {"Ascend950_case_dual_a4w4_weight_nz",
     {1, 7168},
     {1536, 3584},
     {1, 14},
     {1, 112,2},
     {14, 1536},
     {11536, 112, 2},
     {1536},
     false,
     true,
     512,
     32,
     {1, 1536},
     ACL_FLOAT4_E2M1,
     ACL_INT8,
     ACL_FLOAT,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FLOAT8_E8M0,
     ACL_FLOAT,
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_FRACTAL_NZ,
     true,
    },
};

INSTANTIATE_TEST_SUITE_P(
    Ascend950_DualLevelQuantMatmulNz, l2_dual_level_quant_batch_matmul_nz_test_950,
    testing::ValuesIn(casesParamsAscend950));

static void ThreadFunc(
    const DualLevelQuantMatmulNzTestParam* params, size_t testcase_num, size_t thread_idx, size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const DualLevelQuantMatmulNzTestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(l2_dual_level_quant_batch_matmul_nz_test_950, ascend950_multi_thread)
{
    // 用3个线程测试
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    TestMultiThread(
        casesParamsAscend950, sizeof(casesParamsAscend950) / sizeof(DualLevelQuantMatmulNzTestParam), 3);
}