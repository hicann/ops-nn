/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_rt.h"
#include "opdev/platform.h"

// Phase-one CPU UT has a mock platform and no runtime device context. Keep the SDK's
// resource query consistent with that platform instead of creating an NPU context.
extern "C" aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    if (value == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }
    if (type == ACL_RT_DEV_RES_CUBE_CORE) {
        *value = op::GetCurrentPlatformInfo().GetCubeCoreNum();
    } else if (type == ACL_RT_DEV_RES_VECTOR_CORE) {
        *value = op::GetCurrentPlatformInfo().GetVectorCoreNum();
    } else {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}
