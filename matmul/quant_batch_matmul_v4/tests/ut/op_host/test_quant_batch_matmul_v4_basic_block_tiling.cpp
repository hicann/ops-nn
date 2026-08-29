/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "log/log.h"
#include "../../../op_host/op_tiling/arch35/quant_batch_matmul_v4_basic_block_tiling.h"
#include "ut_string_utils.h"

using namespace ut_str;
using namespace optiling::matmul_v4;

namespace {
PlatformParam GetV100PlatformParam()
{
    PlatformParam param;
    param.blockNum = 32;
    param.aicNum = 32;
    param.ubSize = 196352;
    param.l1Size = 524032;
    param.l0aSize = 65536;
    param.l0bSize = 65536;
    param.l0cSize = 262144;
    param.cacheLine = 256;
    param.minCacheLine = 128;
    param.frequency = 1.6;
    param.hbmBW = 1.6;
    param.l2BW = 5.4;
    return param;
}
} // namespace

struct QuantBatchMatmulV4BasicBlockTilingTestParam {
    std::string caseName;
    bool expectRet;
    bool expectBaseM;
    bool expectBaseN;
    bool expectBaseK;
    bool expectMte2BW;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t groupSize;
    bool transA;
    bool transB;
    bool hasBias;
    bool weightNz;
    bool isMx;
    int64_t l1Size;
    int64_t aBits;
    int64_t bBits;
    int64_t biasBits;
    int64_t yScaleBits;
};

class TestQuantBatchMatmulV4BasicBlockTiling
    : public testing::TestWithParam<QuantBatchMatmulV4BasicBlockTilingTestParam> {};

static std::vector<QuantBatchMatmulV4BasicBlockTilingTestParam> GetParams()
{
    std::vector<QuantBatchMatmulV4BasicBlockTilingTestParam> params;
    const std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    const std::string casePath(
        rootPath + "matmul/quant_batch_matmul_v4/tests/ut/op_host/test_quant_batch_matmul_v4_basic_block_tiling.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    std::string line;
    bool headerSkipped = false;
    while (std::getline(csvData, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }

        std::vector<std::string> testParam;
        SplitStr2Vec(line, ",", testParam);
        if (testParam.size() < 20) {
            continue;
        }

        QuantBatchMatmulV4BasicBlockTilingTestParam param;
        size_t idx = 0;
        param.caseName = Trim(testParam[idx++]);
        param.expectRet = ParseBool(Trim(testParam[idx++]));
        param.expectBaseM = ParseBool(Trim(testParam[idx++]));
        param.expectBaseN = ParseBool(Trim(testParam[idx++]));
        param.expectBaseK = ParseBool(Trim(testParam[idx++]));
        param.expectMte2BW = ParseBool(Trim(testParam[idx++]));
        param.m = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        param.n = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        param.k = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        param.groupSize = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        param.transA = ParseBool(Trim(testParam[idx++]));
        param.transB = ParseBool(Trim(testParam[idx++]));
        param.hasBias = ParseBool(Trim(testParam[idx++]));
        param.weightNz = ParseBool(Trim(testParam[idx++]));
        param.isMx = ParseBool(Trim(testParam[idx++]));
        param.l1Size = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        param.aBits = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        param.bBits = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        param.biasBits = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        param.yScaleBits = ParseInt64OrDefault(Trim(testParam[idx++]), 0);
        params.push_back(param);
    }
    return params;
}

static void TestOneParamCase(const QuantBatchMatmulV4BasicBlockTilingTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;

    PlatformParam platformParam = GetV100PlatformParam();
    platformParam.l1Size = param.l1Size;

    QuantBatchMatmulV4BasicBlockTiling tiling;
    tiling.SetPlatformParam(platformParam);
    tiling.SetShape(param.m, param.n, param.k, param.groupSize);
    tiling.SetAttr("QuantBatchMatmulV4", param.transA, param.transB, param.hasBias, param.weightNz);
    tiling.SetQuantType(param.isMx);
    tiling.SetDtypeBits(param.aBits, param.bBits, param.biasBits, param.yScaleBits);

    bool ret = tiling.GetBasicBlockTiling();
    if (param.expectRet) {
        EXPECT_TRUE(ret);
    } else {
        EXPECT_FALSE(ret);
    }

    const auto& result = tiling.GetTilingResult();
    if (param.expectBaseM) {
        EXPECT_GT(result.basicBlock.baseM, 0);
    }
    if (param.expectBaseN) {
        EXPECT_GT(result.basicBlock.baseN, 0);
    }
    if (param.expectBaseK) {
        EXPECT_GT(result.basicBlock.baseK, 0);
    }
    if (param.expectMte2BW) {
        EXPECT_GT(result.basicBlock.mte2BW, 0.0);
    }
}

TEST_P(TestQuantBatchMatmulV4BasicBlockTiling, generalTest)
{
    QuantBatchMatmulV4BasicBlockTilingTestParam param = GetParam();
    TestOneParamCase(param);
}

static const std::vector<QuantBatchMatmulV4BasicBlockTilingTestParam> kCaseParams = GetParams();

INSTANTIATE_TEST_CASE_P(MM, TestQuantBatchMatmulV4BasicBlockTiling, testing::ValuesIn(kCaseParams));
