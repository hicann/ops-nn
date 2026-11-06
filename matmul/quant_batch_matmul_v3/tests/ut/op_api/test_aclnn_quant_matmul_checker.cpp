/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_matmul_checker.cpp
 * \brief ND checker contracts and group-size inference; WeightNz API regressions use its existing CSV.
 */

#include <array>
#include <fstream>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#pragma push_macro("OP_LOG_LIBOPAPI_ONLY")
#ifndef OP_LOG_LIBOPAPI_ONLY
#define OP_LOG_LIBOPAPI_ONLY
#endif
#include "graph/ge_error_codes.h"
#include "../../../../../common/inc/op_host/op_const_def_nn.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_common/op_host/util/op_const_def.h"
#include "opdev/op_log.h"
#include "log/log.h"

// Internal checker symbols are hidden in the API shared library. Compile the real implementation
// under a test-only class name, as in the repository's other API white-box tests.
// Route opbase diagnostics through the SDK API logger already linked by this UT target.
#pragma push_macro("OP_LOGE_LIBOPAPI_REPORT")
#undef OP_LOGE_LIBOPAPI_REPORT
#define OP_LOGE_LIBOPAPI_REPORT(opName, fmt, ...) OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s: " fmt, opName, ##__VA_ARGS__)
#define QuantMatmulChecker QuantMatmulCheckerForUT
#include "../../../op_api/quant_matmul_checker.cpp"
#undef QuantMatmulChecker
#pragma pop_macro("OP_LOGE_LIBOPAPI_REPORT")
#pragma pop_macro("OP_LOG_LIBOPAPI_ONLY")

namespace {
struct CheckerTensorParam {
    std::string shape;
    std::string dtype;
};

struct CheckerCase {
    std::string caseName;
    NpuArch arch;
    std::array<CheckerTensorParam, 7> tensors; // x1, x2, x1Scale, x2Scale, x2Offset, bias, out.
    bool transposeX1;
    bool transposeX2;
    int64_t groupSize;
    int64_t interfaceType;
    bool inferGroup;
    aclnnStatus expectRet;
    bool expectInfer;
    int64_t expectGroupSize;
    bool expectNonSymmetric;
};

std::vector<CheckerCase> LoadCheckerCases()
{
    const std::string
        path = ut_str::GetExeDirPath() +
               "../../../../matmul/quant_batch_matmul_v3/tests/ut/op_api/test_aclnn_quant_matmul_checker.csv";
    std::ifstream csv(path);
    if (!csv.is_open()) {
        ADD_FAILURE() << "Cannot open checker cases: " << path;
        return {};
    }
    std::vector<CheckerCase> cases;
    std::string line;
    while (std::getline(csv, line)) {
        line = ut_str::Trim(line);
        if (line.empty() || line[0] == '#' || line.rfind("caseName,", 0) == 0) {
            continue;
        }
        std::vector<std::string> columns;
        ut_str::SplitStr2Vec(line, ",", columns);
        constexpr size_t expectedColumns = 25;
        if (columns.size() != expectedColumns) {
            ADD_FAILURE() << "Expected 25 columns: " << line;
            continue;
        }
        static const std::map<std::string, NpuArch> arches{{"DAV_3510", Ops::Base::DAV_3510},
                                                           {"DAV_9201", NpuArch::DAV_9201},
                                                           {"DAV_9202", NpuArch::DAV_9202},
                                                           {"DAV_2201", NpuArch::DAV_2201},
                                                           {"DAV_2002", NpuArch::DAV_2002}};
        size_t index = 0;
        CheckerCase param{};
        param.caseName = columns[index++];
        param.arch = arches.at(columns[index++]);
        for (auto& tensor : param.tensors) {
            tensor.shape = columns[index++];
            tensor.dtype = columns[index++];
        }
        param.transposeX1 = ut_str::ParseBool(columns[index++]);
        param.transposeX2 = ut_str::ParseBool(columns[index++]);
        param.groupSize = std::stoll(columns[index++]);
        param.interfaceType = std::stoll(columns[index++]);
        param.inferGroup = columns[index++] == "infer";
        param.expectRet = ut_str::ParseAclnnStatus(columns[index++]);
        param.expectInfer = ut_str::ParseBool(columns[index++]);
        param.expectGroupSize = std::stoll(columns[index++]);
        param.expectNonSymmetric = ut_str::ParseBool(columns[index++]);
        cases.push_back(param);
    }
    return cases;
}

using TensorPtr = std::unique_ptr<aclTensor, void (*)(aclTensor*)>;

TensorPtr MakeCheckerTensor(const CheckerTensorParam& param)
{
    if (param.shape == "null") {
        return TensorPtr(nullptr, [](aclTensor*) {});
    }
    // The checker consumes metadata only. Construct HIF4 metadata without depending on SDK allocation support.
    const bool isHif4 = param.dtype == "HIFLOAT4" || param.dtype == "HIFLOAT4_SCALE";
    auto tensor = ::TensorDesc(ut_str::ParseInt64Vec(param.shape),
                               isHif4 ? ACL_INT8 : ut_str::ParseAclDataType(param.dtype), ACL_FORMAT_ND)
                      .ToAclType();
    if (isHif4) {
        tensor->SetDataType(param.dtype == "HIFLOAT4" ? Ops::NN::DT_HIFLOAT4 : Ops::NN::DT_HIFLOAT4_SCALE);
    }
    return tensor;
}

class QuantMatmulCheckerTest : public testing::TestWithParam<CheckerCase> {};

// The existing API CSV does not describe scale storage padding, view offsets, or view strides.
aclnnStatus CheckScaleNzMetadata(const std::vector<int64_t>& storage, const std::vector<int64_t>& strides,
                                 int64_t offset, aclFormat weightFormat = ACL_FORMAT_FRACTAL_NZ)
{
    auto x1 = ::TensorDesc({16, 128}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND).ToAclType();
    auto x2 = ::TensorDesc({128, 256}, ACL_FLOAT8_E4M3FN, weightFormat, {}, 0, {8, 8, 16, 32}).ToAclType();
    auto x1Scale = ::TensorDesc({16, 2, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND).ToAclType();
    auto x2Scale = ::TensorDesc({2, 256, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_FRACTAL_NZ, strides, offset, storage)
                       .ToAclType();
    auto out = ::TensorDesc({16, 256}, ACL_BF16, ACL_FORMAT_ND).ToAclType();
    const int64_t groupSize = (1LL << 32) | (1LL << 16) | 32;
    const int64_t interfaceType = 5;
    const TupleQuant quant{x1Scale.get(), x2Scale.get(), nullptr,   nullptr,      nullptr,
                           nullptr,       nullptr,       groupSize, interfaceType};
    op::QuantMatmulCheckerForUT checker({x1.get(), x2.get()}, quant, {false, false}, out.get(), true,
                                        "QuantMatmulCheckerScaleNz");
    checker.Init();
    return checker.CheckParams();
}

TEST(QuantMatmulCheckerScaleNz, CanonicalPackingRequiresSupportedArchitectureAndNzWeight)
{
    {
        op::NpuArchManager archManager(Ops::Base::DAV_3510);
        EXPECT_EQ(CheckScaleNzMetadata({16, 2, 16, 2}, {}, 0), ACLNN_SUCCESS);
        EXPECT_EQ(CheckScaleNzMetadata({16, 2, 16, 2}, {}, 0, ACL_FORMAT_ND), ACLNN_ERR_PARAM_INVALID);
    }
    op::NpuArchManager archManager(NpuArch::DAV_9201);
    EXPECT_EQ(CheckScaleNzMetadata({16, 2, 16, 2}, {}, 0), ACLNN_ERR_PARAM_INVALID);
}

TEST(QuantMatmulCheckerScaleNz, RejectsWrongPaddingOffsetAndNonContiguousScaleView)
{
    op::NpuArchManager archManager(Ops::Base::DAV_3510);
    EXPECT_EQ(CheckScaleNzMetadata({15, 2, 16, 2}, {}, 0), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckScaleNzMetadata({16, 2, 16, 2}, {}, 1), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(CheckScaleNzMetadata({16, 2, 16, 2}, {768, 3, 1}, 0), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(QuantMatmulCheckerTest, ValidatesMetadataAndGroupInference)
{
    const auto& param = GetParam();
    SCOPED_TRACE(param.caseName);
    op::NpuArchManager archManager(param.arch);
    auto x1 = MakeCheckerTensor(param.tensors[0]);
    auto x2 = MakeCheckerTensor(param.tensors[1]);
    auto x1Scale = MakeCheckerTensor(param.tensors[2]);
    auto x2Scale = MakeCheckerTensor(param.tensors[3]);
    auto x2Offset = MakeCheckerTensor(param.tensors[4]);
    auto bias = MakeCheckerTensor(param.tensors[5]);
    auto out = MakeCheckerTensor(param.tensors[6]);
    const TupleInput inputs{x1.get(), x2.get()};
    const TupleQuant quant{x1Scale.get(),   x2Scale.get(),      nullptr, nullptr, x2Offset.get(), nullptr, bias.get(),
                           param.groupSize, param.interfaceType};
    op::QuantMatmulCheckerForUT checker(inputs, quant, {param.transposeX1, param.transposeX2}, out.get(), false,
                                        "QuantMatmulCheckerTest");
    checker.Init();
    if (param.inferGroup) {
        int64_t inferredGroup = param.groupSize;
        EXPECT_EQ(checker.InferGroupSize(inferredGroup), param.expectInfer);
        EXPECT_EQ(inferredGroup, param.expectGroupSize);
    } else {
        EXPECT_EQ(checker.CheckParams(), param.expectRet);
    }
    EXPECT_EQ(checker.IsA4W4PergroupNonSymmetric(), param.expectNonSymmetric);
}

INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulV3, QuantMatmulCheckerTest, testing::ValuesIn(LoadCheckerCases()),
                         [](const testing::TestParamInfo<CheckerCase>& info) { return info.param.caseName; });
} // namespace
