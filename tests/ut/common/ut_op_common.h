/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ut_op_common.h
 */

#ifndef NN_TESTS_UT_COMMON_UT_OP_COMMON_H_
#define NN_TESTS_UT_COMMON_UT_OP_COMMON_H_

#include <iostream>
#include <vector>
#include "register/op_impl_registry.h"
#include "runtime/storage_shape.h"
#include "runtime/infer_shape_context.h"
#include "runtime/kernel_context.h"
#include "infer_shape_context_faker.h"
#include "infer_shaperange_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "op_impl_registry.h"
#include "graph/operator.h"

using namespace std;
using namespace ge;

struct Runtime2TestParam {
  std::vector<std::string> attrs;
  std::vector<bool> input_const;
  std::vector<uint32_t> irnum;
};

ge::graphStatus InferShapeTest(ge::Operator& op, const Runtime2TestParam& param);

ge::graphStatus InferShapeTest(ge::Operator& op);

ge::graphStatus InferDataTypeTest(ge::Operator& op, const Runtime2TestParam& param);

ge::graphStatus InferDataTypeTest(ge::Operator& op);

#endif //NN_TESTS_UT_COMMON_UT_OP_COMMON_H_