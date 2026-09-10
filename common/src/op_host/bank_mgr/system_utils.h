/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_COMMON_UTILS_SYSTEM_UTILS_H_
#define RUNTIME_KB_COMMON_UTILS_SYSTEM_UTILS_H_

#include <string>

#include "kb_status.h"

namespace RuntimeKb {
class SystemUtils {
public:
    static std::string RealPath(const std::string& path);
    static std::string GetParentDir(const std::string& path);
    static bool CheckDirExistenceAndPermission(const std::string& path);
    static bool IsDirExist(const std::string& path);
    static std::string GetEnv(const std::string& k);
    static Status CreateMultiDirectory(const std::string& dir);
    static int32_t GetCpuCoreNum();
    static uint32_t GetPid();
};
} // namespace RuntimeKb
#endif
