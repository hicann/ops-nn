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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class AddExamplePyptoTiling : public testing::Test {
public:
    static void SetUpTestCase() { cout << "AddExamplePyptoTiling SetUp" << endl; }
    static void TearDownTestCase() { cout << "AddExamplePyptoTiling TearDown" << endl; }
};

struct AddExamplePyptoCompileInfo {};

TEST_F(AddExamplePyptoTiling, shape_ok)
{
    AddExamplePyptoCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AddExamplePypto",
                                              {
                                                  {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 16, 262144, 4096);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "32 64 ";
    vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(AddExamplePyptoTiling, shape_mismatch)
{
    AddExamplePyptoCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AddExamplePypto",
                                              {
                                                  {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{16, 32}, {16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 16, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

TEST_F(AddExamplePyptoTiling, dim_mismatch)
{
    AddExamplePyptoCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("AddExamplePypto",
                                              {
                                                  {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{64}, {64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{32, 64}, {32, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, 16, 262144, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
