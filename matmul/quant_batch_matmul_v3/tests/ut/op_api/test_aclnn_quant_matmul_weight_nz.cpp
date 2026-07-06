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
#include <memory>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_quant_matmul_weight_nz.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"
using namespace ut_str;

using namespace std;
using namespace op;

struct QuantBatchMatmulWeightNzParam {
    string caseName;
    string arch;

    vector<int64_t> x1Shape;
    aclDataType x1Type;
    aclFormat x1Format;
    vector<int64_t> x1Stride;
    vector<int64_t> x1StorageShape;

    bool x2Null;
    vector<int64_t> x2Shape;
    aclDataType x2Type;
    aclFormat x2Format;
    vector<int64_t> x2Stride;
    vector<int64_t> x2StorageShape;

    bool x1ScaleNull;
    vector<int64_t> x1ScaleShape;
    aclDataType x1ScaleType;
    aclFormat x1ScaleFormat;

    bool x2ScaleNull;
    vector<int64_t> x2ScaleShape;
    aclDataType x2ScaleType;
    aclFormat x2ScaleFormat;

    bool yScaleNull;
    vector<int64_t> yScaleShape;
    aclDataType yScaleType;
    aclFormat yScaleFormat;

    bool x1OffsetNull;
    vector<int64_t> x1OffsetShape;
    aclDataType x1OffsetType;
    aclFormat x1OffsetFormat;

    bool groupScaleNull;
    vector<int64_t> groupScaleShape;
    aclDataType groupScaleType;
    aclFormat groupScaleFormat;

    bool yOffsetNull;
    vector<int64_t> yOffsetShape;
    aclDataType yOffsetType;
    aclFormat yOffsetFormat;

    bool biasNull;
    vector<int64_t> biasShape;
    aclDataType biasType;
    aclFormat biasFormat;

    vector<int64_t> outShape;
    aclDataType outType;
    aclFormat outFormat;

    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    aclnnStatus expectRet;
    bool checkRet;
};

static vector<QuantBatchMatmulWeightNzParam> GetParams()
{
    vector<QuantBatchMatmulWeightNzParam> params;
    string rootPath(ut_str::GetExeDirPath() + "../../../../");
    string casePath(rootPath + "matmul/quant_batch_matmul_v3/tests/ut/op_api/test_aclnn_quant_matmul_weight_nz.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        return params;
    }

    string line;
    bool skipHeader = true;
    constexpr size_t kExpectedCols = 49UL;
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
        QuantBatchMatmulWeightNzParam p;
        p.caseName = Trim(cols[idx++]);
        p.arch = Trim(cols[idx++]);

        p.x1Shape = ParseInt64Vec(cols[idx++]);
        p.x1Type = ParseAclDataType(cols[idx++]);
        p.x1Format = ParseAclFormat(cols[idx++]);
        p.x1Stride = ParseInt64Vec(cols[idx++]);
        p.x1StorageShape = ParseInt64Vec(cols[idx++]);

        p.x2Null = ParseBool(cols[idx++]);
        p.x2Shape = ParseInt64Vec(cols[idx++]);
        p.x2Type = ParseAclDataType(cols[idx++]);
        p.x2Format = ParseAclFormat(cols[idx++]);
        p.x2Stride = ParseInt64Vec(cols[idx++]);
        p.x2StorageShape = ParseInt64Vec(cols[idx++]);

        p.x1ScaleNull = ParseBool(cols[idx++]);
        p.x1ScaleShape = ParseInt64Vec(cols[idx++]);
        p.x1ScaleType = ParseAclDataType(cols[idx++]);
        p.x1ScaleFormat = ParseAclFormat(cols[idx++]);

        p.x2ScaleNull = ParseBool(cols[idx++]);
        p.x2ScaleShape = ParseInt64Vec(cols[idx++]);
        p.x2ScaleType = ParseAclDataType(cols[idx++]);
        p.x2ScaleFormat = ParseAclFormat(cols[idx++]);

        p.yScaleNull = ParseBool(cols[idx++]);
        p.yScaleShape = ParseInt64Vec(cols[idx++]);
        p.yScaleType = ParseAclDataType(cols[idx++]);
        p.yScaleFormat = ParseAclFormat(cols[idx++]);

        p.x1OffsetNull = ParseBool(cols[idx++]);
        p.x1OffsetShape = ParseInt64Vec(cols[idx++]);
        p.x1OffsetType = ParseAclDataType(cols[idx++]);
        p.x1OffsetFormat = ParseAclFormat(cols[idx++]);

        p.groupScaleNull = ParseBool(cols[idx++]);
        p.groupScaleShape = ParseInt64Vec(cols[idx++]);
        p.groupScaleType = ParseAclDataType(cols[idx++]);
        p.groupScaleFormat = ParseAclFormat(cols[idx++]);

        p.yOffsetNull = ParseBool(cols[idx++]);
        p.yOffsetShape = ParseInt64Vec(cols[idx++]);
        p.yOffsetType = ParseAclDataType(cols[idx++]);
        p.yOffsetFormat = ParseAclFormat(cols[idx++]);

        p.biasNull = ParseBool(cols[idx++]);
        p.biasShape = ParseInt64Vec(cols[idx++]);
        p.biasType = ParseAclDataType(cols[idx++]);
        p.biasFormat = ParseAclFormat(cols[idx++]);

        p.outShape = ParseInt64Vec(cols[idx++]);
        p.outType = ParseAclDataType(cols[idx++]);
        p.outFormat = ParseAclFormat(cols[idx++]);

        p.transposeX1 = ParseBool(cols[idx++]);
        p.transposeX2 = ParseBool(cols[idx++]);
        p.groupSize = ParseInt64OrDefault(cols[idx++], 0);
        p.expectRet = ParseAclnnStatus(cols[idx++]);
        p.checkRet = ParseBool(cols[idx++]);
        params.push_back(p);
    }
    return params;
}

class l2_QuantBatchMatmulWeightNz_test : public testing::TestWithParam<QuantBatchMatmulWeightNzParam> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

static TensorDesc CreateTensorDesc(const vector<int64_t>& shape, aclDataType type, aclFormat format,
                                   const vector<int64_t>& stride = {}, const vector<int64_t>& storage = {})
{
    if (!storage.empty()) {
        return TensorDesc(shape, type, format, stride, 0, storage);
    }
    if (!stride.empty()) {
        return TensorDesc(shape, type, format, stride);
    }
    return TensorDesc(shape, type, format);
}

static aclnnStatus RunCaseWithMask(const QuantBatchMatmulWeightNzParam& param)
{
    TensorDesc x1Desc = CreateTensorDesc(param.x1Shape, param.x1Type, param.x1Format, param.x1Stride,
                                         param.x1StorageShape);
    TensorDesc x2Desc = CreateTensorDesc(param.x2Shape, param.x2Type, param.x2Format, param.x2Stride,
                                         param.x2StorageShape);
    TensorDesc x1ScaleDesc = CreateTensorDesc(param.x1ScaleShape, param.x1ScaleType, param.x1ScaleFormat);
    TensorDesc x2ScaleDesc = CreateTensorDesc(param.x2ScaleShape, param.x2ScaleType, param.x2ScaleFormat);
    TensorDesc yScaleDesc = CreateTensorDesc(param.yScaleShape, param.yScaleType, param.yScaleFormat);
    TensorDesc x1OffsetDesc = CreateTensorDesc(param.x1OffsetShape, param.x1OffsetType, param.x1OffsetFormat);
    TensorDesc groupScaleDesc = CreateTensorDesc(param.groupScaleShape, param.groupScaleType, param.groupScaleFormat);
    TensorDesc yOffsetDesc = CreateTensorDesc(param.yOffsetShape, param.yOffsetType, param.yOffsetFormat);
    TensorDesc biasDesc = CreateTensorDesc(param.biasShape, param.biasType, param.biasFormat);
    TensorDesc outDesc = CreateTensorDesc(param.outShape, param.outType, param.outFormat);

    uint64_t workspace_size = 0;
    const uint32_t mask = (static_cast<uint32_t>(param.x2Null) << 7U) |
                          (static_cast<uint32_t>(param.x1ScaleNull) << 6U) |
                          (static_cast<uint32_t>(param.x2ScaleNull) << 5U) |
                          (static_cast<uint32_t>(param.yScaleNull) << 4U) |
                          (static_cast<uint32_t>(param.x1OffsetNull) << 3U) |
                          (static_cast<uint32_t>(param.groupScaleNull) << 2U) |
                          (static_cast<uint32_t>(param.yOffsetNull) << 1U) | static_cast<uint32_t>(param.biasNull);

    switch (mask) {
        case 29U: {
            auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                                INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, yOffsetDesc,
                                      nullptr, param.transposeX1, param.transposeX2, param.groupSize),
                                OUTPUT(outDesc));
            return ut.TestGetWorkspaceSize(&workspace_size);
        }
        case 30U: {
            auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                                INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
                                      biasDesc, param.transposeX1, param.transposeX2, param.groupSize),
                                OUTPUT(outDesc));
            return ut.TestGetWorkspaceSize(&workspace_size);
        }
        case 31U: {
            auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                                INPUT(x1Desc, x2Desc, x1ScaleDesc, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
                                      nullptr, param.transposeX1, param.transposeX2, param.groupSize),
                                OUTPUT(outDesc));
            return ut.TestGetWorkspaceSize(&workspace_size);
        }
        case 79U: {
            auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                                INPUT(x1Desc, x2Desc, nullptr, x2ScaleDesc, yScaleDesc, nullptr, nullptr, nullptr,
                                      nullptr, param.transposeX1, param.transposeX2, param.groupSize),
                                OUTPUT(outDesc));
            return ut.TestGetWorkspaceSize(&workspace_size);
        }
        case 87U: {
            auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                                INPUT(x1Desc, x2Desc, nullptr, x2ScaleDesc, nullptr, x1OffsetDesc, nullptr, nullptr,
                                      nullptr, param.transposeX1, param.transposeX2, param.groupSize),
                                OUTPUT(outDesc));
            return ut.TestGetWorkspaceSize(&workspace_size);
        }
        case 93U: {
            auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                                INPUT(x1Desc, x2Desc, nullptr, x2ScaleDesc, nullptr, nullptr, nullptr, yOffsetDesc,
                                      nullptr, param.transposeX1, param.transposeX2, param.groupSize),
                                OUTPUT(outDesc));
            return ut.TestGetWorkspaceSize(&workspace_size);
        }
        case 95U: {
            auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                                INPUT(x1Desc, x2Desc, nullptr, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr, nullptr,
                                      param.transposeX1, param.transposeX2, param.groupSize),
                                OUTPUT(outDesc));
            return ut.TestGetWorkspaceSize(&workspace_size);
        }
        case 223U: {
            auto ut = OP_API_UT(aclnnQuantMatmulWeightNz,
                                INPUT(x1Desc, nullptr, nullptr, x2ScaleDesc, nullptr, nullptr, nullptr, nullptr,
                                      nullptr, param.transposeX1, param.transposeX2, param.groupSize),
                                OUTPUT(outDesc));
            return ut.TestGetWorkspaceSize(&workspace_size);
        }
        default:
            ADD_FAILURE() << "Unsupported optional-input mask for case: " << param.caseName << ", mask=" << mask;
            return ACLNN_ERR_PARAM_INVALID;
    }
}

TEST_P(l2_QuantBatchMatmulWeightNz_test, QuantBatchMatmulWeightNzFromCsv)
{
    const auto& param = GetParam();
    std::unique_ptr<op::NpuArchManager> archManager;
    if (param.arch == "DAV_3510") {
        archManager = std::make_unique<op::NpuArchManager>(NpuArch::DAV_3510);
    }

    const aclnnStatus aclRet = RunCaseWithMask(param);
    if (param.checkRet) {
        EXPECT_EQ(aclRet, param.expectRet);
    }
}

INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulWeightNz, l2_QuantBatchMatmulWeightNz_test, testing::ValuesIn(GetParams()));
