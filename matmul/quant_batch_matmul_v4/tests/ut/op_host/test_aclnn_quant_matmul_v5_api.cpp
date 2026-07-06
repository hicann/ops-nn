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
#include <fstream>
#include <gmock/gmock.h>
#include <algorithm>
#include <vector>
#include <array>
#include <map>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_quant_matmul_v5.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"

using namespace ut_str;

using namespace std;
using namespace op;

struct QuantBatchMatmulV5TestParam {
    string caseName;
    vector<int64_t> x1Shape;
    vector<int64_t> x2Shape;
    vector<int64_t> x1StorageShape;
    vector<int64_t> x2StorageShape;
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
    aclnnStatus expect_ret;
    bool check_ret = true;
    aclFormat x1Format = aclFormat::ACL_FORMAT_ND;
    aclFormat x2Format = aclFormat::ACL_FORMAT_ND;
};

class l2_QuantBatchMatmulV5_test_950 : public testing::TestWithParam<QuantBatchMatmulV5TestParam> {
protected:
    static void SetUpTestCase() { cout << "l2_QuantBatchMatmulV5_test_950 SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_QuantBatchMatmulV5_test_950 TearDown" << endl; }
};

class l2_QuantBatchMatmulV5_test_910B2 : public testing::TestWithParam<QuantBatchMatmulV5TestParam> {
protected:
    static void SetUpTestCase() { cout << "l2_QuantBatchMatmulV5_test_910B2 SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_QuantBatchMatmulV5_test_910B2 TearDown" << endl; }
};

vector<int64_t> GenerateNZShape(const vector<int64_t>& viewShape, const aclDataType& dtype)
{
    if (viewShape.size() < 2) {
        throw invalid_argument("size of viewShape must >= 2 when create fractalNz shape, actual is " +
                               std::to_string(viewShape.size()));
    }
    if (dtype != ACL_FLOAT8_E4M3FN) {
        throw invalid_argument("only support dtype float8_e4m3fn when create fractalNz shape, actual is " +
                               std::to_string(static_cast<int32_t>(dtype)));
    }
    int64_t c0 = 32;
    vector<int64_t> storageShape(viewShape.size() + 2, 0);
    copy(viewShape.begin(), viewShape.end() - 2, storageShape.begin());
    int64_t x = viewShape[viewShape.size() - 2];
    int64_t y = viewShape[viewShape.size() - 1];
    storageShape[storageShape.size() - 4] = ceil(static_cast<double>(y) / c0);
    storageShape[storageShape.size() - 3] = ceil(static_cast<double>(x) / 16);
    storageShape[storageShape.size() - 2] = 16;
    storageShape[storageShape.size() - 1] = c0;
    return storageShape;
}

static std::string GetRepoRootPath() { return ut_str::GetExeDirPath() + "../../../../"; }

static std::vector<QuantBatchMatmulV5TestParam> LoadParamsFromCsv(const string& socVersion)
{
    std::vector<QuantBatchMatmulV5TestParam> params;
    const std::string rootPath = GetRepoRootPath();
    const std::string casePath(rootPath +
                               "matmul/quant_batch_matmul_v4/tests/ut/op_host/test_aclnn_quant_matmul_v5_api.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    std::string line;
    std::vector<std::string> headers;
    while (std::getline(csvData, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::vector<std::string> cols;
        SplitStr2Vec(line, ",", cols);
        if (headers.empty()) {
            for (const auto& col : cols) {
                headers.push_back(Trim(col));
            }
            continue;
        }
        std::map<std::string, std::string> row;
        for (size_t i = 0; i < headers.size(); ++i) {
            row[headers[i]] = i < cols.size() ? Trim(cols[i]) : "";
        }
        if (row["socVersion"] != socVersion) {
            continue;
        }
        QuantBatchMatmulV5TestParam param;
        param.caseName = row["caseName"];
        param.x1Shape = ParseInt64Vec(row["x1Shape"]);
        param.x2Shape = ParseInt64Vec(row["x2Shape"]);
        if (row.find("x1StorageShape") != row.end()) {
            param.x1StorageShape = ParseInt64Vec(row["x1StorageShape"]);
        }
        if (row.find("x2StorageShape") != row.end()) {
            param.x2StorageShape = ParseInt64Vec(row["x2StorageShape"]);
        }
        param.scaleShape = ParseInt64Vec(row["scaleShape"]);
        param.offsetShape = ParseInt64Vec(row["offsetShape"]);
        param.pertokenScaleShape = ParseInt64Vec(row["pertokenScaleShape"]);
        param.biasShape = ParseInt64Vec(row["biasShape"]);
        param.yScaleShape = ParseInt64Vec(row["yScaleShape"]);
        param.x1OffsetShape = ParseInt64Vec(row["x1OffsetShape"]);
        param.yOffsetShape = ParseInt64Vec(row["yOffsetShape"]);
        param.outShape = ParseInt64Vec(row["outShape"]);
        param.x1_stride = ParseInt64Vec(row["x1Stride"]);
        param.x2_stride = ParseInt64Vec(row["x2Stride"]);
        param.x1Type = ParseAclDataType(row["x1Type"], ACL_FLOAT);
        param.x2Type = ParseAclDataType(row["x2Type"], ACL_FLOAT);
        param.scaleType = ParseAclDataType(row["scaleType"], ACL_FLOAT);
        param.offsetType = ParseAclDataType(row["offsetType"], ACL_FLOAT);
        param.pertokenType = ParseAclDataType(row["pertokenType"], ACL_FLOAT);
        param.biasType = ParseAclDataType(row["biasType"], ACL_FLOAT);
        param.yScaleType = ParseAclDataType(row["yScaleType"], ACL_FLOAT);
        param.x1OffsetType = ParseAclDataType(row["x1OffsetType"], ACL_FLOAT);
        param.yOffsetType = ParseAclDataType(row["yOffsetType"], ACL_FLOAT);
        param.outType = ParseAclDataType(row["outType"], ACL_FLOAT);
        param.transposeX1 = ParseBool(row["transposeX1"]);
        param.transposeX2 = ParseBool(row["transposeX2"]);
        param.groupSize = row["groupSize"].empty() ? 0 : stoll(row["groupSize"]);
        param.expect_ret = ParseAclnnStatus(row["expectRet"]);
        if (row.find("checkRet") != row.end() && !row["checkRet"].empty()) {
            param.check_ret = ParseBool(row["checkRet"]);
        }
        if (row.find("x1Format") != row.end() && !row["x1Format"].empty()) {
            param.x1Format = ParseAclFormat(row["x1Format"]);
        }
        param.x2Format = ParseAclFormat(row["x2Format"]);
        params.push_back(param);
    }
    return params;
}

static void TestOneParamCase(const QuantBatchMatmulV5TestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    const bool isDav3510 = param.caseName.rfind("ascend950", 0) == 0 || param.caseName.rfind("ascend3510", 0) == 0 ||
                           param.caseName.rfind("ascend910D", 0) == 0;
    op::NpuArchManager archManager(isDav3510 ? NpuArch::DAV_3510 : NpuArch::DAV_2201);
    vector<int64_t> stride;
    void* deviceAddr = nullptr;
    aclTensor* x1Desc = nullptr;
    if (param.x1Shape.size() > 0) {
        const vector<int64_t> x1Stride = param.x1_stride.empty() ? MakeDefaultStride(param.x1Shape) : param.x1_stride;
        const vector<int64_t> x1StorageShape = param.x1StorageShape.empty() ? param.x1Shape : param.x1StorageShape;
        x1Desc = aclCreateTensor(param.x1Shape.data(), param.x1Shape.size(), param.x1Type, x1Stride.data(), 0,
                                 param.x1Format, x1StorageShape.data(), x1StorageShape.size(), deviceAddr);
    }
    aclTensor* x2Desc = nullptr;
    if (param.x2Shape.size() > 0) {
        const aclFormat format = param.x2Format;
        vector<int64_t> storageShape = param.x2StorageShape;
        if (storageShape.empty()) {
            if ((format == aclFormat::ACL_FORMAT_FRACTAL_NZ_C0_32 || format == aclFormat::ACL_FORMAT_FRACTAL_NZ) &&
                param.x2Type == ACL_FLOAT8_E4M3FN) {
                storageShape = GenerateNZShape(param.x2Shape, param.x2Type);
            } else {
                storageShape = param.x2Shape;
            }
        }

        const vector<int64_t> x2Stride = param.x2_stride.empty() ? MakeDefaultStride(param.x2Shape) : param.x2_stride;
        x2Desc = aclCreateTensor(param.x2Shape.data(), param.x2Shape.size(), param.x2Type, x2Stride.data(), 0, format,
                                 storageShape.data(), storageShape.size(), deviceAddr);
    }
    aclTensor* scaleDesc = nullptr;
    if (param.scaleShape.size() > 0) {
        scaleDesc = aclCreateTensor(param.scaleShape.data(), param.scaleShape.size(), param.scaleType, stride.data(), 0,
                                    aclFormat::ACL_FORMAT_ND, param.scaleShape.data(), param.scaleShape.size(),
                                    deviceAddr);
    }
    aclTensor* offsetDesc = nullptr;
    if (param.offsetShape.size() > 0) {
        offsetDesc = aclCreateTensor(param.offsetShape.data(), param.offsetShape.size(), param.offsetType,
                                     stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.offsetShape.data(),
                                     param.offsetShape.size(), deviceAddr);
    }
    aclTensor* pertokenDesc = nullptr;
    if (param.pertokenScaleShape.size() > 0) {
        pertokenDesc = aclCreateTensor(param.pertokenScaleShape.data(), param.pertokenScaleShape.size(),
                                       param.pertokenType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                                       param.pertokenScaleShape.data(), param.pertokenScaleShape.size(), deviceAddr);
    }
    aclTensor* biasDesc = nullptr;
    if (param.biasShape.size() > 0) {
        biasDesc = aclCreateTensor(param.biasShape.data(), param.biasShape.size(), param.biasType, stride.data(), 0,
                                   aclFormat::ACL_FORMAT_ND, param.biasShape.data(), param.biasShape.size(),
                                   deviceAddr);
    }
    aclTensor* outDesc = nullptr;
    if (param.outShape.size() > 0) {
        outDesc = aclCreateTensor(param.outShape.data(), param.outShape.size(), param.outType, stride.data(), 0,
                                  aclFormat::ACL_FORMAT_ND, param.outShape.data(), param.outShape.size(), deviceAddr);
    }
    aclTensor* yScaleDesc = nullptr;
    if (param.yScaleShape.size() > 0) {
        yScaleDesc = aclCreateTensor(param.yScaleShape.data(), param.yScaleShape.size(), param.yScaleType,
                                     stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.yScaleShape.data(),
                                     param.yScaleShape.size(), deviceAddr);
    }
    aclTensor* x1OffsetDesc = nullptr;
    if (param.x1OffsetShape.size() > 0) {
        x1OffsetDesc = aclCreateTensor(param.x1OffsetShape.data(), param.x1OffsetShape.size(), param.x1OffsetType,
                                       stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.x1OffsetShape.data(),
                                       param.x1OffsetShape.size(), deviceAddr);
    }
    aclTensor* yOffsetDesc = nullptr;
    if (param.yOffsetShape.size() > 0) {
        yOffsetDesc = aclCreateTensor(param.yOffsetShape.data(), param.yOffsetShape.size(), param.yOffsetType,
                                      stride.data(), 0, aclFormat::ACL_FORMAT_ND, param.yOffsetShape.data(),
                                      param.yOffsetShape.size(), deviceAddr);
    }

    aclnnStatus aclRet;
    auto ut = OP_API_UT(aclnnQuantMatmulV5,
                        INPUT(x1Desc, x2Desc, pertokenDesc, scaleDesc, yScaleDesc, x1OffsetDesc, offsetDesc,
                              yOffsetDesc, biasDesc, param.transposeX1, param.transposeX2, param.groupSize),
                        OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (param.check_ret) {
        ASSERT_EQ(aclRet, param.expect_ret);
    } else {
        std::cout << "skip ret check for case " << param.caseName << ", actual ret " << aclRet << std::endl;
    }
    std::cout << "end case " << param.caseName << std::endl;
}

TEST_P(l2_QuantBatchMatmulV5_test_950, ascend950_generalTest)
{
    QuantBatchMatmulV5TestParam param = GetParam();
    TestOneParamCase(param);
}

TEST_P(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_generalTest)
{
    QuantBatchMatmulV5TestParam param = GetParam();
    TestOneParamCase(param);
}

static const std::vector<QuantBatchMatmulV5TestParam> kCasesParams950 = LoadParamsFromCsv("Ascend950");
static const std::vector<QuantBatchMatmulV5TestParam> kCasesParams910B2 = LoadParamsFromCsv("Ascend910B2");

INSTANTIATE_TEST_SUITE_P(Ascend950_QuantBatchMatmulV5, l2_QuantBatchMatmulV5_test_950,
                         testing::ValuesIn(kCasesParams950));
INSTANTIATE_TEST_SUITE_P(Ascend910B2_QuantBatchMatmulV5, l2_QuantBatchMatmulV5_test_910B2,
                         testing::ValuesIn(kCasesParams910B2));

static void ThreadFunc(const QuantBatchMatmulV5TestParam* params, size_t testcase_num, size_t thread_idx,
                       size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const QuantBatchMatmulV5TestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(l2_QuantBatchMatmulV5_test_910B2, ascend910B2_multi_thread)
{
    TestMultiThread(kCasesParams910B2.data(), kCasesParams910B2.size(), 3);
}
