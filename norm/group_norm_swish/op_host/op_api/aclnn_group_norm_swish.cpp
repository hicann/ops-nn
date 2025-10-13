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
 * \file aclnn_group_norm_swish.cpp
 * \brief
 */
#include <string>
#include "graph/types.h"
#include "aclnn_group_norm_swish.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerGroupNormSwishGetWorkspaceSize(
    const aclTensor *x, const aclTensor *gamma,const aclTensor *beta, int64_t numGroups, char *dataFormatOptional,
    double eps, bool activateSwish, double swishScale, const aclTensor *yOut, const aclTensor *meanOut,
    const aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerGroupNormSwish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);

aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(
    const aclTensor *x, const aclTensor *gamma,const aclTensor *beta, int64_t numGroups, char *dataFormatOptional,
    double eps, bool activateSwish, double swishScale, const aclTensor *yOut, const aclTensor *meanOut,
    const aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerGroupNormSwishGetWorkspaceSize(
        x, gamma, beta, numGroups, dataFormatOptional, eps, activateSwish, swishScale, yOut, meanOut, rstdOut,
        workspaceSize, executor);
}

aclnnStatus aclnnGroupNormSwish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return aclnnInnerGroupNormSwish(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
