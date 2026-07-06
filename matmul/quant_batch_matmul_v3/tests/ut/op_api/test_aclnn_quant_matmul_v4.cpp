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
#include <fstream>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_quant_matmul_v4.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"
using namespace ut_str;

using namespace std;
using namespace op;

struct QuantBatchMatmulV4TestParam {
    string socVersion;
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

struct QuantBatchMatmulV4SpecialTestParam {
    string caseName;
    vector<int64_t> x1;
    aclDataType x1Type;
    aclFormat x1Format;
    vector<int64_t> x1StorageShape;
    vector<int64_t> x2;
    aclDataType x2Type;
    aclFormat x2Format;
    vector<int64_t> x2StorageShape;
    vector<int64_t> scale;
    aclDataType scaleType;
    bool offsetNull;
    vector<int64_t> offset;
    aclDataType offsetType;
    aclFormat offsetFormat;
    bool pertokenNull;
    vector<int64_t> pertoken;
    aclDataType pertokenType;
    aclFormat pertokenFormat;
    bool biasNull;
    vector<int64_t> bias;
    aclDataType biasType;
    aclFormat biasFormat;
    vector<int64_t> out;
    aclDataType outType;
    bool transposeX1;
    bool transposeX2;
    aclnnStatus expectRet;
    bool checkRet;
};

struct QuantBatchMatmulV4CsvRow {
    string caseGroup;
    string socVersion;
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
    aclDataType x1Type;
    aclFormat x1Format;
    vector<int64_t> x1StorageShape;
    aclDataType x2Type;
    aclFormat x2Format;
    vector<int64_t> x2StorageShape;
    bool offsetNull;
    aclDataType offsetType;
    aclFormat offsetFormat;
    bool pertokenNull;
    vector<int64_t> pertoken;
    aclDataType pertokenType;
    aclFormat pertokenFormat;
    bool biasNull;
    aclDataType biasType;
    aclFormat biasFormat;
    bool transposeX1;
    bool transposeX2;
    aclnnStatus expectRet;
    bool checkRet;
};

static vector<QuantBatchMatmulV4CsvRow> GetCsvRows()
{
    vector<QuantBatchMatmulV4CsvRow> rows;
    string rootPath(ut_str::GetExeDirPath() + "../../../../");
    string casePath(rootPath + "matmul/quant_batch_matmul_v3/tests/ut/op_api/test_aclnn_quant_matmul_v4.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        return rows;
    }

    string line;
    bool skipHeader = true;
    constexpr size_t kExpectedCols = 33UL;
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
        QuantBatchMatmulV4CsvRow row;
        row.caseGroup = Trim(cols[idx++]);
        row.socVersion = Trim(cols[idx++]);
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
        row.offsetNull = ParseBool(cols[idx++]);
        row.offsetType = ParseAclDataType(cols[idx++]);
        row.offsetFormat = ParseAclFormat(cols[idx++]);
        row.pertokenNull = ParseBool(cols[idx++]);
        row.pertoken = ParseInt64Vec(cols[idx++]);
        row.pertokenType = ParseAclDataType(cols[idx++]);
        row.pertokenFormat = ParseAclFormat(cols[idx++]);
        row.biasNull = ParseBool(cols[idx++]);
        row.biasType = ParseAclDataType(cols[idx++]);
        row.biasFormat = ParseAclFormat(cols[idx++]);
        row.transposeX1 = ParseBool(cols[idx++]);
        row.transposeX2 = ParseBool(cols[idx++]);
        const string checkRetStr = Trim(cols[idx++]);
        row.checkRet = checkRetStr.empty() ? true : ParseBool(checkRetStr);
        rows.push_back(row);
    }
    return rows;
}

static vector<QuantBatchMatmulV4TestParam> GetParams(const string& socVersion)
{
    vector<QuantBatchMatmulV4TestParam> params;
    const auto rows = GetCsvRows();
    for (const auto& row : rows) {
        if (row.caseGroup != "general" || row.socVersion != socVersion) {
            continue;
        }
        QuantBatchMatmulV4TestParam param;
        param.socVersion = row.socVersion;
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

static const vector<QuantBatchMatmulV4TestParam>& GetParams950Cache()
{
    static const vector<QuantBatchMatmulV4TestParam> params = GetParams("Ascend950");
    return params;
}

static vector<QuantBatchMatmulV4SpecialTestParam> GetSpecialParams()
{
    vector<QuantBatchMatmulV4SpecialTestParam> params;
    const auto rows = GetCsvRows();
    for (const auto& row : rows) {
        if (row.caseGroup != "special") {
            continue;
        }
        QuantBatchMatmulV4SpecialTestParam p;
        p.caseName = row.caseName;
        p.x1 = row.x1;
        p.x1Type = row.x1Type;
        p.x1Format = row.x1Format;
        p.x1StorageShape = row.x1StorageShape;
        p.x2 = row.x2;
        p.x2Type = row.x2Type;
        p.x2Format = row.x2Format;
        p.x2StorageShape = row.x2StorageShape;
        p.scale = row.scale;
        p.scaleType = row.scaleType;
        p.offsetNull = row.offsetNull;
        p.offset = row.offset;
        p.offsetType = row.offsetType;
        p.offsetFormat = row.offsetFormat;
        p.pertokenNull = row.pertokenNull;
        p.pertoken = row.pertoken;
        p.pertokenType = row.pertokenType;
        p.pertokenFormat = row.pertokenFormat;
        p.biasNull = row.biasNull;
        p.bias = row.bias;
        p.biasType = row.biasType;
        p.biasFormat = row.biasFormat;
        p.out = row.out;
        p.outType = row.outType;
        p.transposeX1 = row.transposeX1;
        p.transposeX2 = row.transposeX2;
        p.expectRet = row.expectRet;
        p.checkRet = row.checkRet;
        params.push_back(p);
    }
    return params;
}

class l2_QuantBatchMatmulV4_test : public testing::TestWithParam<QuantBatchMatmulV4TestParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

class l2_QuantBatchMatmulV4_test_310P : public testing::TestWithParam<QuantBatchMatmulV4TestParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

class l2_QuantBatchMatmulV4_test_950 : public testing::TestWithParam<QuantBatchMatmulV4TestParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

class l2_QuantBatchMatmulV4_special_test : public testing::TestWithParam<QuantBatchMatmulV4SpecialTestParam> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

static void TestOneParamCase(const QuantBatchMatmulV4TestParam& param)
{
    TensorDesc x1_desc = TensorDesc(param.x1, ACL_INT8, ACL_FORMAT_ND, param.x1_stride);
    TensorDesc x2_desc = TensorDesc(param.x2, ACL_INT8, ACL_FORMAT_ND, param.x2_stride);
    TensorDesc scale_desc = TensorDesc(param.scale, param.scaleType, ACL_FORMAT_ND);
    TensorDesc offset_desc = TensorDesc(param.offset, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc(param.bias, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc(param.out, param.outType, ACL_FORMAT_ND);
    bool hasOffset = param.offset.size() == 0 ? false : true;
    bool hasBias = param.bias.size() == 0 ? false : true;
    bool hasPertoken = param.bias.size() == 0 ? false : true;
    auto mandtoryInput = std::make_tuple(x1_desc, x2_desc, scale_desc);
    aclnnStatus aclRet = ACLNN_ERR_PARAM_INVALID;
    uint64_t workspace_size = 0;
    if (hasOffset && hasBias) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulV4,
            std::tuple_cat(mandtoryInput, std::make_tuple(offset_desc, nullptr, bias_desc, false, false)),
            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (hasOffset) {
        auto ut = OP_API_UT(aclnnQuantMatmulV4,
                            std::tuple_cat(mandtoryInput, std::make_tuple(offset_desc, nullptr, nullptr, false, false)),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (hasBias) {
        auto ut = OP_API_UT(aclnnQuantMatmulV4,
                            std::tuple_cat(mandtoryInput, std::make_tuple(nullptr, nullptr, bias_desc, false, false)),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else {
        auto ut = OP_API_UT(aclnnQuantMatmulV4,
                            std::tuple_cat(mandtoryInput, std::make_tuple(nullptr, nullptr, nullptr, false, false)),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
    // EXPECT_EQ(aclRet, param.expect_ret);
}

TEST_P(l2_QuantBatchMatmulV4_test, ascend910B2_generalTest)
{
    QuantBatchMatmulV4TestParam param = GetParam();
    TestOneParamCase(param);
}
TEST_P(l2_QuantBatchMatmulV4_test_310P, ascend310P_generalTest)
{
    QuantBatchMatmulV4TestParam param = GetParam();
    TestOneParamCase(param);
}
TEST_P(l2_QuantBatchMatmulV4_test_950, ascend950_generalTest)
{
    QuantBatchMatmulV4TestParam param = GetParam();
    TestOneParamCase(param);
}

TEST_P(l2_QuantBatchMatmulV4_special_test, ascend_special_csv_test)
{
    const auto& param = GetParam();
    TensorDesc x1_desc = param.x1StorageShape.empty() ?
                             TensorDesc(param.x1, param.x1Type, param.x1Format).ValueRange(-1, 1) :
                             TensorDesc(param.x1, param.x1Type, param.x1Format, {1, 1}, 0, param.x1StorageShape)
                                 .ValueRange(-1, 1);
    TensorDesc x2_desc = param.x2StorageShape.empty() ?
                             TensorDesc(param.x2, param.x2Type, param.x2Format).ValueRange(-1, 1) :
                             TensorDesc(param.x2, param.x2Type, param.x2Format, {1, 1}, 0, param.x2StorageShape)
                                 .ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc(param.scale, param.scaleType, ACL_FORMAT_ND).ValueRange(-5, 5);
    TensorDesc offset_desc = TensorDesc(param.offset, param.offsetType, param.offsetFormat).ValueRange(-1, 1);
    TensorDesc pertoken_desc = TensorDesc(param.pertoken, param.pertokenType, param.pertokenFormat).ValueRange(-5, 5);
    TensorDesc bias_desc = TensorDesc(param.bias, param.biasType, param.biasFormat).ValueRange(-1, 1);
    TensorDesc out_desc = TensorDesc(param.out, param.outType, ACL_FORMAT_ND).ValueRange(-1, 1);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ACLNN_ERR_PARAM_INVALID;
    if (param.offsetNull && param.pertokenNull && param.biasNull) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulV4,
            INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, nullptr, param.transposeX1, param.transposeX2),
            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (param.offsetNull && param.pertokenNull) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulV4,
            INPUT(x1_desc, x2_desc, scale_desc, nullptr, nullptr, bias_desc, param.transposeX1, param.transposeX2),
            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (param.offsetNull && param.biasNull) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulV4,
            INPUT(x1_desc, x2_desc, scale_desc, nullptr, pertoken_desc, nullptr, param.transposeX1, param.transposeX2),
            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (param.pertokenNull && param.biasNull) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulV4,
            INPUT(x1_desc, x2_desc, scale_desc, offset_desc, nullptr, nullptr, param.transposeX1, param.transposeX2),
            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (param.offsetNull) {
        auto ut = OP_API_UT(aclnnQuantMatmulV4,
                            INPUT(x1_desc, x2_desc, scale_desc, nullptr, pertoken_desc, bias_desc, param.transposeX1,
                                  param.transposeX2),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (param.pertokenNull) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulV4,
            INPUT(x1_desc, x2_desc, scale_desc, offset_desc, nullptr, bias_desc, param.transposeX1, param.transposeX2),
            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else if (param.biasNull) {
        auto ut = OP_API_UT(aclnnQuantMatmulV4,
                            INPUT(x1_desc, x2_desc, scale_desc, offset_desc, pertoken_desc, nullptr, param.transposeX1,
                                  param.transposeX2),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    } else {
        auto ut = OP_API_UT(aclnnQuantMatmulV4,
                            INPUT(x1_desc, x2_desc, scale_desc, offset_desc, pertoken_desc, bias_desc,
                                  param.transposeX1, param.transposeX2),
                            OUTPUT(out_desc));
        aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    }
    if (param.checkRet) {
        EXPECT_EQ(aclRet, param.expectRet);
    } else {
    }
}

INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulV4, l2_QuantBatchMatmulV4_test, testing::ValuesIn(GetParams("Ascend910B2")));
INSTANTIATE_TEST_SUITE_P(Ascend310P_QuantBatchMatmulV4, l2_QuantBatchMatmulV4_test_310P,
                         testing::ValuesIn(GetParams("Ascend310P3")));
INSTANTIATE_TEST_SUITE_P(Ascend950_QuantBatchMatmulV4, l2_QuantBatchMatmulV4_test_950,
                         testing::ValuesIn(GetParams("Ascend950")));
INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulV4Special, l2_QuantBatchMatmulV4_special_test,
                         testing::ValuesIn(GetSpecialParams()));

static void ThreadFunc(const QuantBatchMatmulV4TestParam* params, size_t testcase_num, size_t thread_idx,
                       size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const QuantBatchMatmulV4TestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(l2_QuantBatchMatmulV4_test_950, ascend950_multi_thread)
{
    const auto& params950 = GetParams950Cache();
    TestMultiThread(params950.data(), params950.size(), 3);
}
