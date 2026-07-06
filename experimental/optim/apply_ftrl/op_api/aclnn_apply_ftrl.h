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
 * \file aclnn_apply_ftrl.h
 * \brief ApplyFtrl L2 (aclnn) API declaration -- DERIVED prototype.
 *
 * This is the hand-written aclnn wrapper for the experimental registry-invoke
 * ApplyFtrl operator (ascend910b / DAV_2201).  ApplyFtrl is natively a GE-graph /
 * TensorFlow op (no CANN-deployed aclnn interface); this two-phase aclnn façade is
 * NEW for this experimental extension to enable ATK pyaclnn testing.
 *
 * Two-phase interface:
 *   1. aclnnApplyFtrlGetWorkspaceSize -- param check + workspace sizing + executor.
 *   2. aclnnApplyFtrl                 -- launch the kernel.
 *
 * In-place (Ref) semantics: var / accum / linear are passed as non-const aclTensor*
 * and are updated in place (var via its declared output port, accum / linear via
 * write-back through their own input GM).  grad / lr / l1 / l2 / lrPower are read-only.
 */

#ifndef ACLNN_APPLY_FTRL_H_
#define ACLNN_APPLY_FTRL_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First phase of aclnnApplyFtrl: validate params, size workspace, build executor.
 *
 * @param varRef    [in/out] parameter tensor; updated in place (formula var).
 * @param accumRef  [in/out] gradient-square accumulator; updated in place (formula accum, >= 0).
 * @param linearRef [in/out] linear correction term; updated in place (formula linear).
 * @param grad      [in]     current-step gradient (same dtype/shape as varRef).
 * @param lr        [in]     learning rate, scalar (0-D or 1-element 1-D), same dtype.
 * @param l1        [in]     L1 regularization coefficient, scalar.
 * @param l2        [in]     L2 regularization coefficient, scalar.
 * @param lrPower   [in]     power of the scaling factor, scalar.
 * @param useLocking [in]    maps to attribute use_locking; only false supported.
 * @param workspaceSize [out] required device workspace size.
 * @param executor  [out]    returned op executor.
 * @return aclnnStatus status code.
 */
ACLNN_API aclnnStatus aclnnApplyFtrlGetWorkspaceSize(aclTensor* varRef, aclTensor* accumRef, aclTensor* linearRef,
                                                     const aclTensor* grad, const aclTensor* lr, const aclTensor* l1,
                                                     const aclTensor* l2, const aclTensor* lrPower, bool useLocking,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief Second phase of aclnnApplyFtrl: execute the computation (var/accum/linear in place).
 *
 * @param workspace     [in] device workspace address.
 * @param workspaceSize [in] device workspace size (from the first phase).
 * @param executor      [in] op executor (from the first phase).
 * @param stream        [in] execution stream.
 * @return aclnnStatus status code.
 */
ACLNN_API aclnnStatus aclnnApplyFtrl(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_APPLY_FTRL_H_
