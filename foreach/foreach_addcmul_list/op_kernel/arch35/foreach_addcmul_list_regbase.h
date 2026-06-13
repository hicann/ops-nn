/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file foreach_addcmul_list_regbase.h
 * \brief Reg-VF kernel for foreach_addcmul_list: y[t][i] = x1[t][i] + scalars[t] * (x2[t][i] * x3[t][i]).
 *        Thin alias over the shared addc kernel with the multiply combine op.
 */
#ifndef FOREACH_ADDCMUL_LIST_REGBASE_H
#define FOREACH_ADDCMUL_LIST_REGBASE_H

#include "../foreach_utils/arch35/foreach_addc_list_regbase.h"

namespace ForeachAddcmulList {
template <typename T, typename ScalarT, typename Tiling>
using ForeachAddcmulListRegbase =
    ForeachAddcList::ForeachAddcListRegbase<T, ScalarT, Tiling, ForeachAddcList::AddcOp::MUL>;
} // namespace ForeachAddcmulList

#endif // FOREACH_ADDCMUL_LIST_REGBASE_H
