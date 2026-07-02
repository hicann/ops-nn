/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_INC_MATMUL_ADD_H_
#define OP_API_INC_MATMUL_ADD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMatmulAddGetWorkspaceSize(
    const aclTensor* a, const aclTensor* b, const aclTensor* bias,
    aclTensor* yOut, uint64_t* workspaceSize, aclOpExecutor** executor);

ACLNN_API aclnnStatus aclnnMatmulAdd(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_MATMUL_ADD_H_
