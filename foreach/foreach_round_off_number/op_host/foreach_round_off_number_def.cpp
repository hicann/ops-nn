/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file foreach_round_off_number_def.cpp
 * \brief
 */

#include "../../foreach_utils/op_host/foreach_proto_utils.h"

namespace ops {
FOREACH_OPDEF_BEGIN(RoundOffNumber)
FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16);
std::vector<ge::DataType> scalar_tensor_dtype_list{ge::DT_INT8, ge::DT_INT8, ge::DT_INT8};
FOREACH_OPDEF_PARAM_TENSORLIST(Input, x)
FOREACH_OPDEF_PARAM_SCALAR_TENSOR(Input, roundMode);
FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)
FOREACH_OPDEF_END_Atlas_A2_AND_910_93(RoundOffNumber);

    OP_ADD(ForeachRoundOffNumber);
} // namespace ops
