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
 * \file aclnn_group_norm_swish.h
 * \brief
 */
#ifndef ACLNN_GROUP_NORM_SWISH_H_
#define ACLNN_GROUP_NORM_SWISH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeInitRoutingV2GetWorkspaceSize
 * parameters :
 * x : required
 * gamma : required
 * activeNumOptional : optional
 * expertCapacityOptional : optional
 * expertNumOptional : optional
 * dropPadModeOptional : optional
 * expertTokensCountOrCumsumFlagOptional : optional
 * expertTokensBeforeCapacityFlagOptional : optional
 * expandedXOut : required
 * expandedRowIdxOut : required
 * expertTokensCountOrCumsumOutOptional : optional
 * expertTokensBeforeCapacityOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(
    const aclTensor *x, const aclTensor *gamma,const aclTensor *beta, int64_t numGroups, char *dataFormatOptional,
    double eps, bool activateSwish, double swishScale, const aclTensor *yOut, const aclTensor *meanOut,
    const aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/* funtion: aclnnMoeInitRoutingV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupNormSwish(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
