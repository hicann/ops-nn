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
 * \file foreach_lerp_list_def.cpp
 * \brief
 */

#include "../../foreach_utils/op_host/foreach_proto_utils.h"

namespace ops {
FOREACH_OPDEF_BEGIN(LerpList)
FOREACH_TENSOR_DTYPE_AND_FORMAT_PREPARE(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16);
FOREACH_OPDEF_PARAM_TENSORLIST(Input, x1)
FOREACH_OPDEF_PARAM_TENSORLIST(Input, x2)
FOREACH_OPDEF_PARAM_TENSORLIST(Input, weight)
FOREACH_OPDEF_PARAM_TENSORLIST(Output, y)
FOREACH_OPDEF_END_Atlas_A2_AND_910_93(LerpList);

    OP_ADD(ForeachLerpList);
} // namespace ops
