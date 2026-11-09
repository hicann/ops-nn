/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_platform.h"
#include "platform/platform_info.h"
#include <cstdio>

aclError aclplatformGetDeviceInfo(aclplatformDevInfo infoType, char* value, uint32_t maxLen)
{
    if (infoType != ACL_PLATFORM_NPU_ARCH || value == nullptr || maxLen == 0) {
        return static_cast<aclError>(-1);
    }
    value[0] = '\0';
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != 0) {
        return static_cast<aclError>(-1);
    }
    (void)snprintf(value, maxLen, "%d", platformInfo.soc_info.arch_type);
    return ACL_SUCCESS;
}
