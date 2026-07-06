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
#include <array>
#include <fstream>
#include <vector>
#include <thread>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_quant_matmul_v3.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"
using namespace ut_str;

struct QuantBatchMatmulV3TestParam {
    string caseName;
    vector<int64_t> x1;
    vector<int64_t> x2;
    vector<int64_t> scale;
    vector<int64_t> offset;
    vector<int64_t> bias;
    vector<int64_t> out;
    vector<int64_t> x1_stride;
    vector<int64_t> x2_stride;
    aclDataType scaleType;
    aclDataType outType;
    // out
    aclnnStatus expect_ret;
};
struct QuantBatchMatmulV3CsvRow {
    string caseGroup;
    string caseName;
    vector<int64_t> x1;
    vector<int64_t> x2;
    vector<int64_t> scale;
    vector<int64_t> offset;
    vector<int64_t> bias;
    vector<int64_t> out;
    vector<int64_t> x1Stride;
    vector<int64_t> x2Stride;
    aclDataType scaleType;
    aclDataType outType;
    aclnnStatus expectRet;
    aclDataType x1Type;
    aclFormat x1Format;
    vector<int64_t> x1StorageShape;
    aclDataType x2Type;
    aclFormat x2Format;
    vector<int64_t> x2StorageShape;
    bool scaleIsNull;
    aclDataType offsetType;
    aclDataType biasType;
    bool transposeX1;
    bool transposeX2;
    bool checkRet;
};

static vector<QuantBatchMatmulV3CsvRow> GetCsvRows()
{
    vector<QuantBatchMatmulV3CsvRow> rows;
    string rootPath(ut_str::GetExeDirPath() + "../../../../");
    string casePath(rootPath + "matmul/quant_batch_matmul_v3/tests/ut/op_api/test_aclnn_quant_matmul_v3.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        return rows;
    }

    string line;
    bool skipHeader = true;
    constexpr size_t kExpectedCols = 25UL;
    while (std::getline(csvData, line)) {
        const string trimLine = Trim(line);
        if (trimLine.empty() || trimLine[0] == '#') {
            continue;
        }
        if (skipHeader) {
            skipHeader = false;
            continue;
        }
        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() < kExpectedCols) {
            cols.resize(kExpectedCols);
        }
        size_t idx = 0UL;
        QuantBatchMatmulV3CsvRow row;
        row.caseGroup = Trim(cols[idx++]);
        row.caseName = Trim(cols[idx++]);
        row.x1 = ParseInt64Vec(cols[idx++]);
        row.x2 = ParseInt64Vec(cols[idx++]);
        row.scale = ParseInt64Vec(cols[idx++]);
        row.offset = ParseInt64Vec(cols[idx++]);
        row.bias = ParseInt64Vec(cols[idx++]);
        row.out = ParseInt64Vec(cols[idx++]);
        row.x1Stride = ParseInt64Vec(cols[idx++]);
        row.x2Stride = ParseInt64Vec(cols[idx++]);
        row.scaleType = ParseAclDataType(cols[idx++]);
        row.outType = ParseAclDataType(cols[idx++]);
        row.expectRet = ParseAclnnStatus(cols[idx++]);
        row.x1Type = ParseAclDataType(cols[idx++]);
        row.x1Format = ParseAclFormat(cols[idx++]);
        row.x1StorageShape = ParseInt64Vec(cols[idx++]);
        row.x2Type = ParseAclDataType(cols[idx++]);
        row.x2Format = ParseAclFormat(cols[idx++]);
        row.x2StorageShape = ParseInt64Vec(cols[idx++]);
        row.scaleIsNull = ParseBool(cols[idx++]);
        row.offsetType = ParseAclDataType(cols[idx++]);
        row.biasType = ParseAclDataType(cols[idx++]);
        row.transposeX1 = ParseBool(cols[idx++]);
        row.transposeX2 = ParseBool(cols[idx++]);
        row.checkRet = ParseBool(cols[idx++]);
        rows.push_back(row);
    }
    return rows;
}

static vector<QuantBatchMatmulV3TestParam> GetParams()
{
    vector<QuantBatchMatmulV3TestParam> params;
    const auto rows = GetCsvRows();
    for (const auto& row : rows) {
        if (row.caseGroup != "general") {
            continue;
        }
        QuantBatchMatmulV3TestParam param;
        param.caseName = row.caseName;
        param.x1 = row.x1;
        param.x2 = row.x2;
        param.scale = row.scale;
        param.offset = row.offset;
        param.bias = row.bias;
        param.out = row.out;
        param.x1_stride = row.x1Stride;
        param.x2_stride = row.x2Stride;
        param.scaleType = row.scaleType;
        param.outType = row.outType;
        param.expect_ret = row.expectRet;
        params.push_back(param);
    }
    return params;
}

struct QuantBatchMatmulV3SpecialTestParam {
    string caseName;
    vector<int64_t> x1;
    aclDataType x1Type;
    aclFormat x1Format;
    vector<int64_t> x1Stride;
    vector<int64_t> x1StorageShape;
    vector<int64_t> x2;
    aclDataType x2Type;
    aclFormat x2Format;
    vector<int64_t> x2Stride;
    vector<int64_t> x2StorageShape;
    vector<int64_t> scale;
    aclDataType scaleType;
    bool scaleIsNull;
    vector<int64_t> offset;
    aclDataType offsetType;
    vector<int64_t> bias;
    aclDataType biasType;
    vector<int64_t> out;
    aclDataType outType;
    bool transposeX1;
    bool transposeX2;
    aclnnStatus expectRet;
    bool checkRet;
};

static vector<QuantBatchMatmulV3SpecialTestParam> GetSpecialParams()
{
    vector<QuantBatchMatmulV3SpecialTestParam> params;
    const auto rows = GetCsvRows();
    for (const auto& row : rows) {
        if (row.caseGroup != "special") {
            continue;
        }
        QuantBatchMatmulV3SpecialTestParam param;
        param.caseName = row.caseName;
        param.x1 = row.x1;
        param.x1Type = row.x1Type;
        param.x1Format = row.x1Format;
        param.x1Stride = row.x1Stride;
        param.x1StorageShape = row.x1StorageShape;
        param.x2 = row.x2;
        param.x2Type = row.x2Type;
        param.x2Format = row.x2Format;
        param.x2Stride = row.x2Stride;
        param.x2StorageShape = row.x2StorageShape;
        param.scale = row.scale;
        param.scaleType = row.scaleType;
        param.scaleIsNull = row.scaleIsNull;
        param.offset = row.offset;
        param.offsetType = row.offsetType;
        param.bias = row.bias;
        param.biasType = row.biasType;
        param.out = row.out;
        param.outType = row.outType;
        param.transposeX1 = row.transposeX1;
        param.transposeX2 = row.transposeX2;
        param.expectRet = row.expectRet;
        param.checkRet = row.checkRet;
        params.push_back(param);
    }
    return params;
}

static TensorDesc BuildTensorDesc(const vector<int64_t>& shape, aclDataType dtype, aclFormat format,
                                  const vector<int64_t>& stride, const vector<int64_t>& storageShape)
{
    if (!storageShape.empty()) {
        return TensorDesc(shape, dtype, format, stride, 0, storageShape);
    }
    if (!stride.empty()) {
        return TensorDesc(shape, dtype, format, stride);
    }
    return TensorDesc(shape, dtype, format);
}

class l2_QuantBatchMatmulV3_test : public testing::TestWithParam<QuantBatchMatmulV3TestParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

class l2_QuantBatchMatmulV3_special_test : public testing::TestWithParam<QuantBatchMatmulV3SpecialTestParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};
static void TestOneParamCase(const QuantBatchMatmulV3TestParam& param)
{
    TensorDesc x1_desc = TensorDesc(param.x1, ACL_INT8, ACL_FORMAT_ND, param.x1_stride);
    TensorDesc x2_desc = TensorDesc(param.x2, ACL_INT8, ACL_FORMAT_ND, param.x2_stride);
    TensorDesc scale_desc = TensorDesc(param.scale, param.scaleType, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc(param.offset, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc(param.bias, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc(param.out, param.outType, ACL_FORMAT_ND);
    bool hasOffset = param.offset.size() == 0 ? false : true;
    bool hasBias = param.bias.size() == 0 ? false : true;
    auto mandtoryInput = std::make_tuple(x1_desc, x2_desc, scale_desc);
    aclnnStatus aclRet = ACLNN_ERR_PARAM_INVALID;
    uint64_t workspace_size = 0;
    if (hasOffset && hasBias) {
        auto ut = OP_API_UT(aclnnQuantMatmulV3,
                            std::tuple_cat(mandtoryInput, std::make_tuple(offset_desc, bias_desc, false, false)),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (hasOffset) {
        auto ut = OP_API_UT(aclnnQuantMatmulV3,
                            std::tuple_cat(mandtoryInput, std::make_tuple(offset_desc, nullptr, false, false)),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (hasBias) {
        auto ut = OP_API_UT(aclnnQuantMatmulV3,
                            std::tuple_cat(mandtoryInput, std::make_tuple(nullptr, bias_desc, false, false)),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else {
        auto ut = OP_API_UT(aclnnQuantMatmulV3,
                            std::tuple_cat(mandtoryInput, std::make_tuple(nullptr, nullptr, false, false)),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
    //   EXPECT_EQ(aclRet, param.expect_ret);
}

TEST_P(l2_QuantBatchMatmulV3_test, ascend910B2_generalTest)
{
    QuantBatchMatmulV3TestParam param = GetParam();
    TestOneParamCase(param);
}
INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulV3, l2_QuantBatchMatmulV3_test, testing::ValuesIn(GetParams()));

TEST_P(l2_QuantBatchMatmulV3_special_test, ascend_special_csv_test)
{
    const auto& param = GetParam();
    TensorDesc x1_desc = BuildTensorDesc(param.x1, param.x1Type, param.x1Format, param.x1Stride, param.x1StorageShape);
    TensorDesc x2_desc = BuildTensorDesc(param.x2, param.x2Type, param.x2Format, param.x2Stride, param.x2StorageShape);
    TensorDesc scale_desc = TensorDesc(param.scale, param.scaleType, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc(param.offset, param.offsetType, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc(param.bias, param.biasType, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc(param.out, param.outType, ACL_FORMAT_ND);

    const bool hasOffset = !param.offset.empty();
    const bool hasBias = !param.bias.empty();
    aclnnStatus aclRet = ACLNN_ERR_PARAM_INVALID;
    uint64_t workspace_size = 0;

    if (param.scaleIsNull) {
        if (hasOffset && hasBias) {
            auto ut = OP_API_UT(
                aclnnQuantMatmulV3,
                INPUT(x1_desc, x2_desc, nullptr, offset_desc, bias_desc, param.transposeX1, param.transposeX2),
                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (hasOffset) {
            auto ut = OP_API_UT(
                aclnnQuantMatmulV3,
                INPUT(x1_desc, x2_desc, nullptr, offset_desc, nullptr, param.transposeX1, param.transposeX2),
                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (hasBias) {
            auto ut = OP_API_UT(
                aclnnQuantMatmulV3,
                INPUT(x1_desc, x2_desc, nullptr, nullptr, bias_desc, param.transposeX1, param.transposeX2),
                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else {
            auto ut = OP_API_UT(
                aclnnQuantMatmulV3,
                INPUT(x1_desc, x2_desc, nullptr, nullptr, nullptr, param.transposeX1, param.transposeX2),
                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        }
    } else {
        if (hasOffset && hasBias) {
            auto ut = OP_API_UT(
                aclnnQuantMatmulV3,
                INPUT(x1_desc, x2_desc, scale_desc, offset_desc, bias_desc, param.transposeX1, param.transposeX2),
                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (hasOffset) {
            auto ut = OP_API_UT(
                aclnnQuantMatmulV3,
                INPUT(x1_desc, x2_desc, scale_desc, offset_desc, nullptr, param.transposeX1, param.transposeX2),
                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else if (hasBias) {
            auto ut = OP_API_UT(
                aclnnQuantMatmulV3,
                INPUT(x1_desc, x2_desc, scale_desc, nullptr, bias_desc, param.transposeX1, param.transposeX2),
                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        } else {
            auto ut = OP_API_UT(
                aclnnQuantMatmulV3,
                INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, param.transposeX1, param.transposeX2),
                OUTPUT(out_desc));
            aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        }
    }

    if (param.checkRet) {
        EXPECT_EQ(aclRet, param.expectRet);
    }
}

INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulV3Special, l2_QuantBatchMatmulV3_special_test,
                         testing::ValuesIn(GetSpecialParams()));

static void ThreadFunc(const QuantBatchMatmulV3TestParam* params, size_t testcase_num, size_t thread_idx,
                       size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const QuantBatchMatmulV3TestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

// TEST_F(l2_QuantBatchMatmulV3_test, ascend910B2_multi_thread)
// {
//     TestMultiThread(casesParams, sizeof(casesParams) / sizeof(QuantBatchMatmulV3TestParam), 3);
// }
