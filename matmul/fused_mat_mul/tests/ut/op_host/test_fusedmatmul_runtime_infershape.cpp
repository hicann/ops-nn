/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdlib>
#include <tuple>

#include "test_case_infershape.h"
#include "error_util.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/storage_format.h"
#include "gtest/gtest.h"
#include "kernel_run_context_facker.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace std;
using namespace ge;
using namespace op;

template <typename T>
std::string DebugString(const std::vector<T>& v) {
  std::ostringstream oss;
  oss << "[";
  if (v.size() > 0) {
    for (size_t i = 0; i < v.size() - 1; ++i) {
      oss << v[i] << ", ";
    }
    oss << v[v.size() - 1];
  }
  oss << "]";
  return oss.str();
}

bool IsUnknownRankShape(const std::vector<int64_t>& shape_vec) {
  if (shape_vec.size() == 1 && shape_vec[0] == -2) {
    return true;
  }
  return false;
}

bool IsUnKnownShape(const std::vector<int64_t>& shape_vec) {
  auto found = find(shape_vec.begin(), shape_vec.end(), -1);
  return found != shape_vec.end();
}

bool IsUnknown(const std::vector<int64_t>& shape_vec) {
  return (IsUnKnownShape(shape_vec) || IsUnknownRankShape(shape_vec));
}

gert::KernelRunContextHolder CreateFusedMatMulHolder(gert::StorageShape& shape_x1, gert::StorageShape& shape_x2,
                                                     gert::StorageShape& shape_bias, gert::StorageShape& shape_x3,
                                                     gert::StorageShape& shape_y, const bool &transpose_x1,
                                                     const bool &transpose_x2, const bool &enable_hf32,
                                                     const std::string &fused_op_type)
{
    gert::KernelRunContextHolder holder;

    holder = gert::InferShapeContextFaker()
                         .NodeIoNum(4, 1)
                         .IrInstanceNum({1, 1, 1, 1})
                         .InputShapes({&shape_x1, &shape_x2, &shape_bias, &shape_x3})
                         .OutputShapes({&shape_y})
                         .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transpose_x1)},
                                     {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(transpose_x2)},
                                     {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enable_hf32)},
                                     {"fused_op_type", Ops::NN::AnyValue::CreateFrom<std::string>(fused_op_type)}})
                         .Build();
    return holder;
}

void OperateFused(FusedMatMul& op, gert::KernelRunContextHolder& holder, bool expected_result)
{
    auto verify_ret = op.VerifyAllAttr(true);
    if (expected_result) {
        EXPECT_EQ(verify_ret, ge::GRAPH_SUCCESS);
    } else if (verify_ret == ge::GRAPH_FAILED) {
        return;
    }

    auto func_infer_shape = gert::OpImplRegistry::GetInstance().GetOpImpl("FusedMatMul")->infer_shape;

    auto ret = func_infer_shape(holder.GetContext<gert::InferShapeContext>());
    if (expected_result) {
        EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    } else {
        EXPECT_EQ(ret, ge::GRAPH_FAILED);
    }
}

void CheckContext(gert::KernelRunContextHolder& holder, const RES_TUPLE& expected)
{
    auto shape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    EXPECT_EQ(Ops::Base::ToString(*shape), DebugString(get<0>(expected)));
}

class FusedMatMulRuntimeProtoTest : public testing::TestWithParam<CASE_TUPLE>
{
public:
    FusedMatMulRuntimeProtoTest()
    {
    }
    static void SetUpTestSuite()
    {
        setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", true);
    }
    static void TearDownTestSuite()
    {
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
    }
};

TEST_P(FusedMatMulRuntimeProtoTest, General)
{
    OP_TUPLE tuple_x1 = get<0>(GetParam());
    OP_TUPLE tuple_x2 = get<1>(GetParam());
    OP_TUPLE tuple_bias = get<2>(GetParam());
    OP_TUPLE tuple_x3 = get<3>(GetParam());

    auto vec_shape_x1 = get<0>(tuple_x1);
    auto vec_shape_x2 = get<0>(tuple_x2);
    auto vec_shape_bias = get<0>(tuple_bias);
    auto vec_shape_x3 = get<0>(tuple_x3);
    if (IsUnknown(vec_shape_x1) || IsUnknown(vec_shape_x2) || IsUnknown(vec_shape_bias) ||
        IsUnknown(vec_shape_x3)) {
        // NOTE not support dynamic yet
        return;
    }

    auto shape_x1 = CreateStorageShape(vec_shape_x1);
    auto shape_x2 = CreateStorageShape(vec_shape_x2);
    auto shape_bias = CreateStorageShape(vec_shape_bias);
    auto shape_x3 = CreateStorageShape(vec_shape_x3);
    auto shape_y = CreateStorageShape({});

    auto holder = CreateFusedMatMulHolder(shape_x1, shape_x2, shape_bias, shape_x3, shape_y, get<4>(GetParam()),
                                          get<5>(GetParam()), get<6>(GetParam()), get<7>(GetParam()));
    auto op = CreateFusedMatmulOp(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()),
                                  std::get<3>(GetParam()), std::get<4>(GetParam()), std::get<5>(GetParam()),
                                  std::get<6>(GetParam()), std::get<7>(GetParam()));

    auto result_tuple = get<8>(GetParam());
    auto expected_result = get<3>(result_tuple);

    OperateFused(op, holder, expected_result);

    if (expected_result) {
        CheckContext(holder, result_tuple);
    }
}

INSTANTIATE_TEST_CASE_P(FusedMatMul, FusedMatMulRuntimeProtoTest, testing::ValuesIn(testcase_fusedmatmul_runtime));