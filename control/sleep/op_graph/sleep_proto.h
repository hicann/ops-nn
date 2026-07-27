/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_OP_PROTO_INC_SLEEP_OPS_H_
#define OPS_OP_PROTO_INC_SLEEP_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
* @brief Inserts a device-side delay into the current stream by busy-spinning
* on the AI Core clock for the specified number of cycles.

* @par Inputs:
 * cycles: A required int64 scalar tensor. Number of AI Core clock cycles
 * to busy-spin. Must be a positive integer (cycles > 0).

* @par Outputs:
* None.

* @par Third-party framework compatibility
* Compatible with pytorch torch.cuda._sleep operator.

* @par Restrictions:
* Warning: This operator uses busy-wait and occupies AI Core resources
* during the sleep period. Actual sleep precision depends on the clock
* frequency of the target platform.
*/
REG_OP(Sleep).INPUT(cycles, TensorType({DT_INT64})).OP_END_FACTORY_REG(Sleep)
} // namespace ge

#endif // OPS_OP_PROTO_INC_SLEEP_OPS_H_
