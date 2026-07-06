/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_ftrl.h
 * \brief ApplyFtrl L0 API declaration (namespace l0op).
 *
 * Launches the AI Core ApplyFtrl kernel.  var / accum / linear are updated in
 * place: var through the single declared output port, accum / linear through
 * write-back into their own input GM.  The returned tuple aliases the three Ref
 * tensors so the L2 layer can ViewCopy them back onto non-contiguous user tensors.
 */

#ifndef OP_API_INC_LEVEL0_APPLY_FTRL_H_
#define OP_API_INC_LEVEL0_APPLY_FTRL_H_

#include "opdev/op_executor.h"

namespace l0op {

std::tuple<const aclTensor*, const aclTensor*, const aclTensor*> ApplyFtrl(
    const aclTensor* varRef, const aclTensor* accumRef, const aclTensor* linearRef, const aclTensor* grad,
    const aclTensor* lr, const aclTensor* l1, const aclTensor* l2, const aclTensor* lrPower, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_APPLY_FTRL_H_
