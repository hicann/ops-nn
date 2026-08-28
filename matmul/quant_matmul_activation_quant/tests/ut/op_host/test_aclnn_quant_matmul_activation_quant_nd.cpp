/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_quant_matmul_activation_quant.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"

using namespace std;
using namespace ut_str;
using namespace op;

struct QuantMatmulActivationQuantNdTestParam {
    string caseName;
    string x1ShapeStr;
    string x1DtypeStr;
    string x1FormatStr;
    string x1StorageStr;
    string x2ShapeStr;
    string x2DtypeStr;
    string x2FormatStr;
    string x2StorageStr;
    string x1ScaleShapeStr;
    string x1ScaleDtypeStr;
    string x1ScaleFormatStr;
    string x1ScaleStorageStr;
    string x2ScaleShapeStr;
    string x2ScaleDtypeStr;
    string x2ScaleFormatStr;
    string x2ScaleStorageStr;
    string biasShapeStr;
    string biasDtypeStr;
    string yShapeStr;
    string yDtypeStr;
    string yFormatStr;
    string yStorageStr;
    string yScaleShapeStr;
    string yScaleDtypeStr;
    string yScaleFormatStr;
    string transposeX1Str;
    string transposeX2Str;
    string groupSizeStr;
    string activationType;
    string quantMode;
    string roundMode;
    string scaleAlgStr;
    string dstTypeMaxStr;
    string expectRetStr;
};

static TensorDesc BuildTensorDesc(const string& shapeStr, const string& dtypeStr, const string& formatStr,
                                  const string& storageStr)
{
    if (shapeStr.empty() && dtypeStr.empty()) {
        return TensorDesc();
    }
    auto shape = ParseInt64Vec(shapeStr);
    auto dtype = ParseAclDataType(dtypeStr);
    auto format = ParseAclFormat(formatStr);
    auto storage = ParseInt64Vec(storageStr);
    if (!storage.empty()) {
        return TensorDesc(shape, dtype, format, {}, 0, storage);
    }
    return TensorDesc(shape, dtype, format);
}

static vector<QuantMatmulActivationQuantNdTestParam> GetParams()
{
    vector<QuantMatmulActivationQuantNdTestParam> params;
    string rootPath(ut_str::GetExeDirPath() + "../../../../");
    string casePath(rootPath + "matmul/quant_matmul_activation_quant/tests/ut/op_host/"
                               "test_aclnn_quant_matmul_activation_quant_nd.csv");
    ifstream csvData(casePath, ios::in);
    if (!csvData.is_open()) {
        return params;
    }

    string line;
    bool skipHeader = true;
    while (getline(csvData, line)) {
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
        if (cols.size() < 35UL) {
            continue;
        }

        QuantMatmulActivationQuantNdTestParam param;
        size_t idx = 0UL;
        param.caseName = Trim(cols[idx++]);
        param.x1ShapeStr = Trim(cols[idx++]);
        param.x1DtypeStr = Trim(cols[idx++]);
        param.x1FormatStr = Trim(cols[idx++]);
        param.x1StorageStr = Trim(cols[idx++]);
        param.x2ShapeStr = Trim(cols[idx++]);
        param.x2DtypeStr = Trim(cols[idx++]);
        param.x2FormatStr = Trim(cols[idx++]);
        param.x2StorageStr = Trim(cols[idx++]);
        param.x1ScaleShapeStr = Trim(cols[idx++]);
        param.x1ScaleDtypeStr = Trim(cols[idx++]);
        param.x1ScaleFormatStr = Trim(cols[idx++]);
        param.x1ScaleStorageStr = Trim(cols[idx++]);
        param.x2ScaleShapeStr = Trim(cols[idx++]);
        param.x2ScaleDtypeStr = Trim(cols[idx++]);
        param.x2ScaleFormatStr = Trim(cols[idx++]);
        param.x2ScaleStorageStr = Trim(cols[idx++]);
        param.biasShapeStr = Trim(cols[idx++]);
        param.biasDtypeStr = Trim(cols[idx++]);
        param.yShapeStr = Trim(cols[idx++]);
        param.yDtypeStr = Trim(cols[idx++]);
        param.yFormatStr = Trim(cols[idx++]);
        param.yStorageStr = Trim(cols[idx++]);
        param.yScaleShapeStr = Trim(cols[idx++]);
        param.yScaleDtypeStr = Trim(cols[idx++]);
        param.yScaleFormatStr = Trim(cols[idx++]);
        param.transposeX1Str = Trim(cols[idx++]);
        param.transposeX2Str = Trim(cols[idx++]);
        param.groupSizeStr = Trim(cols[idx++]);
        param.activationType = Trim(cols[idx++]);
        param.quantMode = Trim(cols[idx++]);
        param.roundMode = Trim(cols[idx++]);
        param.scaleAlgStr = Trim(cols[idx++]);
        param.dstTypeMaxStr = Trim(cols[idx++]);
        param.expectRetStr = Trim(cols[idx++]);
        params.push_back(param);
    }
    return params;
}

class QuantMatmulActivationQuantNdAclnnTest : public testing::TestWithParam<QuantMatmulActivationQuantNdTestParam> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_P(QuantMatmulActivationQuantNdAclnnTest, CsvTest)
{
    const auto& param = GetParam();
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);

    auto x1Desc = BuildTensorDesc(param.x1ShapeStr, param.x1DtypeStr, param.x1FormatStr, param.x1StorageStr);
    auto x2Desc = BuildTensorDesc(param.x2ShapeStr, param.x2DtypeStr, param.x2FormatStr, param.x2StorageStr);
    auto x1ScaleDesc = BuildTensorDesc(param.x1ScaleShapeStr, param.x1ScaleDtypeStr, param.x1ScaleFormatStr,
                                       param.x1ScaleStorageStr);
    auto x2ScaleDesc = BuildTensorDesc(param.x2ScaleShapeStr, param.x2ScaleDtypeStr, param.x2ScaleFormatStr,
                                       param.x2ScaleStorageStr);
    auto yDesc = BuildTensorDesc(param.yShapeStr, param.yDtypeStr, param.yFormatStr, param.yStorageStr);
    auto yScaleDesc = BuildTensorDesc(param.yScaleShapeStr, param.yScaleDtypeStr, param.yScaleFormatStr, "");

    bool hasBias = !param.biasShapeStr.empty();
    auto biasDesc = hasBias ? TensorDesc(ParseInt64Vec(param.biasShapeStr), ParseAclDataType(param.biasDtypeStr),
                                         ACL_FORMAT_ND) :
                              TensorDesc();

    bool hasY = !param.yShapeStr.empty();
    bool hasYScale = !param.yScaleShapeStr.empty();

    bool transposeX1 = ParseBool(param.transposeX1Str);
    bool transposeX2 = ParseBool(param.transposeX2Str);
    int64_t groupSize = ParseInt64OrDefault(param.groupSizeStr, 0);
    int64_t scaleAlg = ParseInt64OrDefault(param.scaleAlgStr, 0);
    double dstTypeMax = param.dstTypeMaxStr.empty() ? 6.0 : stod(param.dstTypeMaxStr);
    aclnnStatus expectRet = ParseAclnnStatus(param.expectRetStr);

    aclnnStatus aclRet = ACLNN_ERR_PARAM_INVALID;
    uint64_t workspaceSize = 0;

    if (hasBias) {
        auto ut = OP_API_UT(
            aclnnQuantMatmulActivationQuant,
            INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, biasDesc, transposeX1, transposeX2, groupSize,
                  param.activationType.c_str(), param.quantMode.c_str(), param.roundMode.c_str(), scaleAlg, dstTypeMax),
            OUTPUT(hasY ? yDesc : TensorDesc(), hasYScale ? yScaleDesc : TensorDesc()));
        aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    } else {
        auto ut = OP_API_UT(
            aclnnQuantMatmulActivationQuant,
            INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, transposeX1, transposeX2, groupSize,
                  param.activationType.c_str(), param.quantMode.c_str(), param.roundMode.c_str(), scaleAlg, dstTypeMax),
            OUTPUT(hasY ? yDesc : TensorDesc(), hasYScale ? yScaleDesc : TensorDesc()));
        aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    }

    EXPECT_EQ(aclRet, expectRet) << "caseName=" << param.caseName;
}

INSTANTIATE_TEST_SUITE_P(QuantMatmulActivationQuantNdAclnn, QuantMatmulActivationQuantNdAclnnTest,
                         testing::ValuesIn(GetParams()));
