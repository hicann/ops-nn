/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <vector>
#include <fstream>
#include <cctype>
#include <iostream>
#include <string>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_quant_batch_matmul_inplace_add.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"
#include "op_common/op_host/util/op_const_def.h"

using namespace ut_str;
using namespace std;
using namespace op;

struct QuantBatchMatmulInplaceAddCsvRow {
    string caseGroup;
    string socVersion;
    string caseName;
    vector<int64_t> x1Shape;
    aclDataType x1Type;
    aclFormat x1Format;
    vector<int64_t> x2Shape;
    aclDataType x2Type;
    aclFormat x2Format;
    vector<int64_t> x1ScaleShape;
    aclDataType x1ScaleType;
    aclFormat x1ScaleFormat;
    vector<int64_t> x2ScaleShape;
    aclDataType x2ScaleType;
    aclFormat x2ScaleFormat;
    vector<int64_t> yInputShape;
    aclDataType yInputType;
    aclFormat yInputFormat;
    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    aclnnStatus expectRet;
    bool checkRet;
    vector<int64_t> x1ScaleStorageShape;
    int64_t x1ScaleOffset;
    vector<int64_t> x2ScaleStorageShape;
    int64_t x2ScaleOffset;
};

using QuantBatchMatmulInplaceAddTestParam = QuantBatchMatmulInplaceAddCsvRow;

struct QuantBatchMatmulInplaceAddCsvLoadResult {
    vector<QuantBatchMatmulInplaceAddCsvRow> rows;
    vector<string> errors;
};

static QuantBatchMatmulInplaceAddCsvLoadResult LoadCsvRows()
{
    QuantBatchMatmulInplaceAddCsvLoadResult result;
    string rootPath(ut_str::GetExeDirPath() + "../../../../");
    string casePath(
        rootPath +
        "matmul/quant_batch_matmul_inplace_add/tests/ut/op_api/test_aclnn_quant_batch_matmul_inplace_add.csv");
    ifstream csvData(casePath, ios::in);
    if (!csvData.is_open()) {
        result.errors.push_back("cannot open case file: " + casePath);
        return result;
    }

    string line;
    bool skipHeader = true;
    constexpr size_t kExpectedCols = 27UL;
    size_t lineNo = 0UL;
    while (getline(csvData, line)) {
        ++lineNo;
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
            result.errors.push_back("skip invalid csv line " + std::to_string(lineNo) + " in " + casePath +
                                    ": expected at least " + std::to_string(kExpectedCols) + " columns, got " +
                                    std::to_string(cols.size()));
            continue;
        }

        size_t idx = 0UL;
        QuantBatchMatmulInplaceAddCsvRow row;
        row.caseGroup = Trim(cols[idx++]);
        row.socVersion = Trim(cols[idx++]);
        row.caseName = Trim(cols[idx++]);
        row.x1Shape = ParseInt64Vec(cols[idx++]);
        row.x1Type = ParseAclDataType(cols[idx++]);
        row.x1Format = ParseAclFormat(cols[idx++]);
        row.x2Shape = ParseInt64Vec(cols[idx++]);
        row.x2Type = ParseAclDataType(cols[idx++]);
        row.x2Format = ParseAclFormat(cols[idx++]);
        row.x1ScaleShape = ParseInt64Vec(cols[idx++]);
        row.x1ScaleType = ParseAclDataType(cols[idx++]);
        row.x1ScaleFormat = ParseAclFormat(cols[idx++]);
        row.x2ScaleShape = ParseInt64Vec(cols[idx++]);
        row.x2ScaleType = ParseAclDataType(cols[idx++]);
        row.x2ScaleFormat = ParseAclFormat(cols[idx++]);
        row.yInputShape = ParseInt64Vec(cols[idx++]);
        row.yInputType = ParseAclDataType(cols[idx++]);
        row.yInputFormat = ParseAclFormat(cols[idx++]);
        row.transposeX1 = ParseBool(cols[idx++]);
        row.transposeX2 = ParseBool(cols[idx++]);
        row.groupSize = ParseInt64OrDefault(cols[idx++], 0);
        row.expectRet = ParseAclnnStatus(cols[idx++]);
        const string checkRetStr = Trim(cols[idx++]);
        row.checkRet = checkRetStr.empty() ? true : ParseBool(checkRetStr);
        row.x1ScaleStorageShape = ParseInt64Vec(cols[idx++]);
        row.x1ScaleOffset = ParseInt64OrDefault(cols[idx++], 0);
        row.x2ScaleStorageShape = ParseInt64Vec(cols[idx++]);
        row.x2ScaleOffset = ParseInt64OrDefault(cols[idx++], 0);
        result.rows.push_back(row);
    }
    if (result.rows.empty()) {
        result.errors.push_back("no valid aclnn cases loaded from: " + casePath);
    }
    return result;
}

static const QuantBatchMatmulInplaceAddCsvLoadResult& GetCsvLoadResult()
{
    static const QuantBatchMatmulInplaceAddCsvLoadResult result = LoadCsvRows();
    return result;
}

static vector<QuantBatchMatmulInplaceAddCsvRow> GetCsvRows() { return GetCsvLoadResult().rows; }

static vector<QuantBatchMatmulInplaceAddTestParam> GetParams()
{
    vector<QuantBatchMatmulInplaceAddTestParam> params;
    const auto rows = GetCsvRows();
    for (const auto& row : rows) {
        if (row.socVersion != "Dav3510") {
            continue;
        }
        params.push_back(row);
    }
    return params;
}

static string SanitizeCaseName(const testing::TestParamInfo<QuantBatchMatmulInplaceAddTestParam>& info)
{
    string name = info.param.caseName;
    for (char& ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return name;
}

TEST(QuantBatchMatmulInplaceAddApiCsv, ShouldLoadValidCases)
{
    const auto& loadResult = GetCsvLoadResult();
    for (const auto& error : loadResult.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(loadResult.rows.empty());
}

class QuantBatchMatmulInplaceAddApiTest : public testing::TestWithParam<QuantBatchMatmulInplaceAddTestParam> {
protected:
    static void SetUpTestCase() { cout << "QuantBatchMatmulInplaceAddApiTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "QuantBatchMatmulInplaceAddApiTest TearDown" << endl; }
};

TEST_P(QuantBatchMatmulInplaceAddApiTest, dav3510_csv_test)
{
    const auto& param = GetParam();
    op::NpuArchManager archManager(Ops::Base::DAV_3510);

    TensorDesc x1Desc = TensorDesc(param.x1Shape, param.x1Type, param.x1Format).ValueRange(-1, 1);
    TensorDesc x2Desc = TensorDesc(param.x2Shape, param.x2Type, param.x2Format).ValueRange(-1, 1);
    TensorDesc x1ScaleDesc = TensorDesc(param.x1ScaleShape, param.x1ScaleType, param.x1ScaleFormat, {},
                                        param.x1ScaleOffset, param.x1ScaleStorageShape)
                                 .ValueRange(-1, 1);
    TensorDesc x2ScaleDesc = TensorDesc(param.x2ScaleShape, param.x2ScaleType, param.x2ScaleFormat, {},
                                        param.x2ScaleOffset, param.x2ScaleStorageShape)
                                 .ValueRange(-1, 1);
    TensorDesc yInputDesc = TensorDesc(param.yInputShape, param.yInputType, param.yInputFormat).ValueRange(-1, 1);

    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, yInputDesc, param.transposeX1,
                              param.transposeX2, param.groupSize),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (param.checkRet) {
        EXPECT_EQ(aclRet, param.expectRet);
    }
}

INSTANTIATE_TEST_SUITE_P(Dav3510QuantBatchMatmulInplaceAdd, QuantBatchMatmulInplaceAddApiTest,
                         testing::ValuesIn(GetParams()), SanitizeCaseName);

TEST_F(QuantBatchMatmulInplaceAddApiTest, dav3510_x2_nz_format_invalid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1Desc = TensorDesc({512, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2Desc = TensorDesc({512, 256}, ACL_HIFLOAT8, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    TensorDesc x1ScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2ScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc yInputDesc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, yInputDesc, true, false, 0), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(QuantBatchMatmulInplaceAddApiTest, dav3510_y_ref_nz_format_invalid)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1Desc = TensorDesc({512, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2Desc = TensorDesc({512, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1ScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2ScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc yInputDesc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_FRACTAL_NZ).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, yInputDesc, true, false, 0), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(QuantBatchMatmulInplaceAddApiTest, dav3510_hif8_success)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    TensorDesc x1Desc = TensorDesc({512, 128}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2Desc = TensorDesc({512, 256}, ACL_HIFLOAT8, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x1ScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2ScaleDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc yInputDesc = TensorDesc({128, 256}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto ut = OP_API_UT(aclnnQuantBatchMatmulInplaceAdd,
                        INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, yInputDesc, true, false, 0), OUTPUT());
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}
