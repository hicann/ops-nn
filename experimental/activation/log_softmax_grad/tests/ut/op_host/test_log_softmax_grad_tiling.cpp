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

#include <vector>
#include <string>
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class LogSoftmaxGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LogSoftmaxGradTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LogSoftmaxGradTiling TearDown" << std::endl; }
};

// opName / inputs / output follow the real operator definition:
//   LogSoftmaxGrad(dy, x) -> z, with attr "axis" (list_int, the reduce dims).
// Tiling key layout (see op_kernel/log_softmax_grad_tiling_key.h):
//   SCH_MOD   : bits [0,8)   (0=NO_NEED_REDUCE, 1=REDUCE_TAIL, 2=REDUCE_MID)
//   IS_SMALL  : bit  8
//   IS_CONT   : bit  9
// so key = schMode | (isSmall << 8) | (isContiguous << 9).

TEST_F(LogSoftmaxGradTiling, no_need_reduce)
{
    struct LogSoftmaxGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("LogSoftmaxGrad",
                                              {
                                                  {{{3, 1, 7}, {3, 1, 7}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
                                                  {{{3, 1, 7}, {3, 1, 7}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x
                                              },
                                              {
                                                  {{{3, 1, 7}, {3, 1, 7}}, ge::DT_FLOAT, ge::FORMAT_ND}, // z
                                              },
                                              {
                                                  {"axis", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-2})},
                                              },
                                              &compileInfo,
                                              64,     // coreNum
                                              262144, // ubSize
                                              4096);  // tilingDataSize
    // mergedDim1 == 1 -> NO_NEED_REDUCE, isSmall=false, isContiguous=false
    uint64_t expectTilingKey = 0;
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(LogSoftmaxGradTiling, reduce_tail)
{
    struct LogSoftmaxGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("LogSoftmaxGrad",
                                              {
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x
                                              },
                                              {
                                                  {{{2, 16}, {2, 16}}, ge::DT_FLOAT, ge::FORMAT_ND}, // z
                                              },
                                              {
                                                  {"axis", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-1})},
                                              },
                                              &compileInfo,
                                              64,     // coreNum
                                              262144, // ubSize
                                              4096);  // tilingDataSize
    // {2,16} reduce last -> mergedDim2==1 -> REDUCE_TAIL, small+contiguous
    uint64_t expectTilingKey = 1 | (1u << 8) | (1u << 9); // 769
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

TEST_F(LogSoftmaxGradTiling, reduce_mid)
{
    struct LogSoftmaxGradCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("LogSoftmaxGrad",
                                              {
                                                  {{{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dy
                                                  {{{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // x
                                              },
                                              {
                                                  {{{2, 16, 8}, {2, 16, 8}}, ge::DT_FLOAT, ge::FORMAT_ND}, // z
                                              },
                                              {
                                                  {"axis", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({-2})},
                                              },
                                              &compileInfo,
                                              64,     // coreNum
                                              262144, // ubSize
                                              4096);  // tilingDataSize
    // {2,16,8} reduce mid -> all merged dims != 1 -> REDUCE_MID, small+contiguous
    uint64_t expectTilingKey = 2 | (1u << 8) | (1u << 9); // 770
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
