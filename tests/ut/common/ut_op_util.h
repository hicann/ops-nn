/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ut_op_util.h
 */

#ifndef NN_TESTS_UT_COMMON_UT_OP_UTIL_H_
#define NN_TESTS_UT_COMMON_UT_OP_UTIL_H_

#include <graph/utils/type_utils.h>
#include "graph/tensor.h"
#include "infershape_test_util.h"
#include "array_ops.h"

using namespace ge;

#define TENSOR_INPUT_WITH_SHAPE(paras, key, shape, dtype, foramt, range)                        \
  auto tensor_desc_##key = create_desc_shape_range(shape, dtype, foramt, shape, foramt, range); \
  auto data##key = op::Data(#key);                                                              \
  data##key.update_input_desc_x(tensor_desc_##key);                                             \
  data##key.update_output_desc_y(tensor_desc_##key);                                            \
  paras.set_input_##key(data##key);                                                             \
  paras.UpdateInputDesc(#key, tensor_desc_##key)

#define TENSOR_INPUT_WITH_SHAPE_AND_CONST_VALUE(paras, key, shape, dtype, foramt, const_value) \
  auto tensor_desc_##key = create_desc_shape_range(shape, dtype, foramt, shape, foramt, {});   \
  Tensor tensor_const_##key(tensor_desc_##key);                                                \
  tensor_desc_##key.SetName(#key);                                                             \
  SetValueToConstTensor(tensor_const_##key, const_value);                                      \
  auto const##key = op::Constant(#key).set_attr_value(tensor_const_##key);                     \
  paras.set_input_##key(const##key);                                                           \
  paras.UpdateInputDesc(#key, tensor_desc_##key)

#define TENSOR_OUTPUT_WITH_SHAPE(paras, key, shape, dtype, foramt, range)                           \
  auto tensor_out_desc_##key = create_desc_shape_range(shape, dtype, foramt, shape, foramt, range); \
  paras.UpdateOutputDesc(#key, tensor_out_desc_##key)

namespace ut_util {
std::vector<int64_t> ToVector(const gert::Shape& shape);
}  // namespace ut_util

#endif //NN_TESTS_UT_COMMON_UT_OP_UTIL_H_