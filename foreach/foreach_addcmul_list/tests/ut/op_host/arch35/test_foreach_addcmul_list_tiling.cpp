/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../../foreach_utils/op_host/foreach_tiling_common.h"

using namespace std;
using namespace ge;

// ascend950 regbase tiling UT for foreach_addcmul_list (ForeachRegbaseTilingTernaryScalarList).
// scalars dtype 与 x 一致（所有 soc 统一原型）。 TilingKey: FP16=10001 FP32=10002 INT32=10003 BF16=10004.
class ForeachAddcmulListRegbaseTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachAddcmulListRegbaseTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachAddcmulListRegbaseTiling TearDown" << std::endl; }

    static optiling::ForeachCompileInfo MakeCompileInfo()
    {
        optiling::ForeachCompileInfo info{};
        info.coreNum = 64;
        info.aivCoreNum = 64;
        info.aicCoreNum = 32;
        info.ubSize = 262144;
        info.sysWorkspaceSize = 16777216;
        return info;
    }
};

TEST_F(ForeachAddcmulListRegbaseTiling, tiling_fp32)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara tilingContextPara("ForeachAddcmulList",
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {/* attrs */}, &compileInfo, 64, 262144, 4096);
    uint64_t expectTilingKey = 10002; // FP32
    string expectTilingData = EMPTY_EXPECT_TILING_DATA;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ForeachAddcmulListRegbaseTiling, tiling_fp16)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara tilingContextPara("ForeachAddcmulList",
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2048}, {2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {/* attrs */}, &compileInfo, 64, 262144, 4096);
    uint64_t expectTilingKey = 10001; // FP16
    string expectTilingData = EMPTY_EXPECT_TILING_DATA;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ForeachAddcmulListRegbaseTiling, tiling_int32)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara tilingContextPara("ForeachAddcmulList",
                                              {
                                                  {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{512}, {512}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {/* attrs */}, &compileInfo, 64, 262144, 4096);
    uint64_t expectTilingKey = 10003; // INT32
    string expectTilingData = EMPTY_EXPECT_TILING_DATA;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(ForeachAddcmulListRegbaseTiling, tiling_bf16)
{
    auto compileInfo = MakeCompileInfo();
    gert::TilingContextPara tilingContextPara("ForeachAddcmulList",
                                              {
                                                  {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096}, {4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {/* attrs */}, &compileInfo, 64, 262144, 4096);
    uint64_t expectTilingKey = 10004; // BF16
    string expectTilingData = EMPTY_EXPECT_TILING_DATA;
    std::vector<size_t> expectWorkspaces = {32};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
