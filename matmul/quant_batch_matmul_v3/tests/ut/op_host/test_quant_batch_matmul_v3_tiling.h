/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TEST_QBMMV3_TILINHG_H_
#define TEST_QBMMV3_TILINHG_H_

#include <gtest/gtest.h>
#include <vector>
#include "../../../op_host/op_tiling/quant_batch_matmul_v3_basic_tiling.h"
#include "../../../op_host/op_tiling/quant_batch_matmul_v3_tiling.h"
#include "../../../op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"
#include "../../../op_host/op_tiling/arch35/base_block_calculator.h"
#include "../../../op_host/op_tiling/arch35/qbmm_streamk_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v3_tiling_data.h"

using namespace optiling;

class QuantBatchMatmulV3TilingTestParam {
public:
    void Prepare(QuantBatchMatmulV3CompileInfo& compileInfo) const;
    void InvokeTilingFunc(QuantBatchMatmulV3CompileInfo& compileInfo) const;
    void Test() const;
    std::string socVersion;
    std::string caseName;
    std::string kernelUtDir;
    std::string prefix;
    int64_t aicNum;
    int64_t aivNum;
    int64_t x1Dim;
    int64_t x2Dim;
    int64_t yDim;
    int64_t batchA;
    int64_t batchB;
    int64_t batchC;
    int64_t m;
    int64_t k;
    int64_t n;
    bool offsetFlag;
    bool pertokenFlag;
    bool biasFlag;
    bool transA;
    bool transB;
    size_t quantMode;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType scaleDtype;
    ge::DataType perTokenScaleDtype;
    ge::DataType biasDtype;
    ge::DataType yDtype;
    bool fmapNz;
    bool weightNz;
    int32_t deterministicLevel = 0;

    // output
    bool result; // false means tiling fail
    uint32_t numBlocks;
    uint64_t tilingKey;
    std::string tilingData;
    bool tilingStub; // 是否tililg打桩，只给kernel的用例，此时tiling ut里不校验tiling出参
};

class TestQuantBatchMatmulV3Tiling : public testing::TestWithParam<QuantBatchMatmulV3TilingTestParam> {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}
};

std::vector<QuantBatchMatmulV3TilingTestParam> GetParams(const std::string& socVersion);

void TestMultiThread(const QuantBatchMatmulV3TilingTestParam* params, size_t testcaseNum, size_t threadNum);

#endif
