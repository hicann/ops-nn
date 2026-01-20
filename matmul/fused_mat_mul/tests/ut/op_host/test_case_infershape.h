/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_BUILT_IN_TESTS_UT_OPS_TEST_FUSED_MATMUL_CASE_PROTO_H_
#define OPS_BUILT_IN_TESTS_UT_OPS_TEST_FUSED_MATMUL_CASE_PROTO_H_

#include <cstdlib>
#include <tuple>
#include <vector>

#include "../../../op_graph/fused_mat_mul_proto.h"
#include "graph/types.h"

#define OP_TUPLE std::tuple<std::vector<int64_t>, ge::DataType, std::vector<std::pair<int64_t, int64_t>>>
#define RES_TUPLE std::tuple<std::vector<int64_t>, std::vector<std::pair<int64_t, int64_t>>, ge::DataType, bool>
#define CASE_TUPLE \
  std::tuple<OP_TUPLE, OP_TUPLE, OP_TUPLE, OP_TUPLE, bool, bool, bool, std::string, RES_TUPLE>

ge::op::FusedMatMul CreateFusedMatmulOp(OP_TUPLE x1, OP_TUPLE x2, OP_TUPLE bias, OP_TUPLE x3,
                                                          const bool &transpose_x1,
                                                          const bool &transpose_x2,
                                                          const bool &enable_hf32,
                                                          const std::string &fused_op_type);

// x1 x2 bias x3 transpose_x1 transpose_x2 enable_hf32 fused_op_type
const static std::vector<CASE_TUPLE> testcase_fusedmatmul_runtime = {
    // static shape
    //   pass cases
    // f16 +bias_f16 relu
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_FLOAT16, {}},
               OP_TUPLE{{2, 3}, ge::DT_FLOAT16, {}},
               OP_TUPLE{{3}, ge::DT_FLOAT16, {}},
               {},
               true,
               false,
               false,
               "relu",
               RES_TUPLE{{3, 3}, {}, ge::DT_FLOAT16, true}},
    // f16 +bias_f32 relu
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_FLOAT16, {}},
               OP_TUPLE{{2, 3}, ge::DT_FLOAT16, {}},
               OP_TUPLE{{3}, ge::DT_FLOAT, {}},
               {},
               true,
               false,
               false,
               "relu",
               RES_TUPLE{{3, 3}, {}, ge::DT_FLOAT16, true}},
    // bf16 +bias_bf16 ""
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{3}, ge::DT_BF16, {}},
               {},
               true,
               false,
               false,
               "",
               RES_TUPLE{{3, 3}, {}, ge::DT_BF16, true}},
    // f16 +bias_f32 ""
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{3}, ge::DT_FLOAT, {}},
               {},
               true,
               false,
               false,
               "",
               RES_TUPLE{{3, 3}, {}, ge::DT_BF16, true}},
    CASE_TUPLE{OP_TUPLE{{4, 5}, ge::DT_FLOAT16, {}},
               OP_TUPLE{{5, 4}, ge::DT_FLOAT16, {}},
               {},
               {},
               false,
               false,
               false,
               "",
               RES_TUPLE{{4, 4}, {}, ge::DT_FLOAT16, true}},
    CASE_TUPLE{OP_TUPLE{{6, 8}, ge::DT_BF16, {}},
               OP_TUPLE{{8, 6}, ge::DT_BF16, {}},
               {},
               {},
               false,
               false,
               false,
               "add",
               RES_TUPLE{{6, 6}, {}, ge::DT_BF16, false}},
    // hf32 +bias relu
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_FLOAT, {}},
               OP_TUPLE{{2, 3}, ge::DT_FLOAT, {}},
               OP_TUPLE{{3}, ge::DT_FLOAT, {}},
               {},
               true,
               false,
               true,
               "relu",
               RES_TUPLE{{3, 3}, {}, ge::DT_FLOAT, true}},
    // not builtin scene for compatibility
    CASE_TUPLE{OP_TUPLE{{288, 320}, ge::DT_BF16, {}},
               OP_TUPLE{{320, 16}, ge::DT_BF16, {}},
               {},
               OP_TUPLE{{288, 16}, ge::DT_BF16, {}},
               false,
               false,
               false,
               "add",
               RES_TUPLE{{288, 16}, {}, ge::DT_BF16, true}},
    // failed case: bias shape is not (n,)
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{5}, ge::DT_BF16, {}},
               {},
               true,
               false,
               false,
               "",
               RES_TUPLE{{3, 3}, {}, ge::DT_BF16, false}},
    // failed case: bias shape is not (1,n)
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               {},
               true,
               false,
               false,
               "",
               RES_TUPLE{{3, 3}, {}, ge::DT_BF16, false}},
    // failed case: bias shape is not (1,n)
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{1, 2}, ge::DT_BF16, {}},
               {},
               true,
               false,
               false,
               "",
               RES_TUPLE{{3, 3}, {}, ge::DT_BF16, false}},
    // failed case: bias shape dims > 2
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{1, 1, 3}, ge::DT_BF16, {}},
               {},
               true,
               false,
               false,
               "",
               RES_TUPLE{{3, 3}, {}, ge::DT_BF16, false}},
    // failed case: not support bias with fused_op_type gelu_tanh
    CASE_TUPLE{OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{2, 3}, ge::DT_BF16, {}},
               OP_TUPLE{{3}, ge::DT_BF16, {}},
               {},
               true,
               false,
               false,
               "gelu_tanh",
               RES_TUPLE{{3, 3}, {}, ge::DT_BF16, false}},
    // failed case 1: wrong fused_op_type no fused_op_type called gelu
    CASE_TUPLE{OP_TUPLE{{4, 5}, ge::DT_FLOAT16, {}},
               OP_TUPLE{{5, 4}, ge::DT_FLOAT16, {}},
               {},
               {},
               false,
               false,
               false,
               "gelu",
               RES_TUPLE{{4, 4}, {}, ge::DT_FLOAT16, false}},
    // failed case 2 : wrong shape of x3
    CASE_TUPLE{OP_TUPLE{{4, 5}, ge::DT_FLOAT16, {}},
               OP_TUPLE{{5, 4}, ge::DT_FLOAT16, {}},
               {},
               OP_TUPLE{{4, 3}, ge::DT_FLOAT16, {}},
               false,
               false,
               false,
               "mul",
               RES_TUPLE{{4, 4}, {}, ge::DT_FLOAT16, false}},
    // failed case 4: wrong shape of x1 or x2
    CASE_TUPLE{OP_TUPLE{{5, 4}, ge::DT_FLOAT16, {}},
               OP_TUPLE{{5, 4}, ge::DT_FLOAT16, {}},
               {},
               OP_TUPLE{{4, 4}, ge::DT_FLOAT16, {}},
               false,
               false,
               false,
               "mul",
               RES_TUPLE{{4, 4}, {}, ge::DT_FLOAT16, false}},

};
#endif