/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef assert_PROTO_H
#define assert_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Asserts that the given condition is true. \n

*@par Inputs:
*If input_condition evaluates to false, print the list of tensors in data.
*Inputs include:
*@li input_condition: The condition to evaluate.
*@li input_data: The tensors to print out when condition is false.
 It's a dynamic input.  \n

*@par Attributes:
*summarize: An optional int. Defaults to 3. Print this many entries of each tensor. \n

*@par Third-party framework compatibility
*Compatible with tensorflow Assert operator. \n

*@par Restrictions:
*Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
*/
REG_OP(Assert)
  .INPUT(input_condition, TensorType{DT_BOOL})
  .DYNAMIC_INPUT(input_data, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8,
      DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
      DT_UINT64, DT_BOOL, DT_DOUBLE, DT_STRING}))
  .ATTR(summarize, Int, 3)
  .OP_END_FACTORY_REG(Assert)
}
#endif