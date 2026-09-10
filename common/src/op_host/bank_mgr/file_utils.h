/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_COMMON_UTILS_FILE_UTILS_H_
#define RUNTIME_KB_COMMON_UTILS_FILE_UTILS_H_

#include <fstream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>

#include "kb_status.h"

namespace RuntimeKb {
constexpr mode_t kSafePermission = S_IRUSR | S_IWUSR | S_IRGRP;
class FileUtils {
public:
    template <typename T>
    static Status OpenWithSafePermission(T& filestream, const std::string& filename,
                                         std::ios_base::openmode mode = std::ios_base::in | std::ios_base::out);
    static bool IsFileAccessedWithSafePermission(const std::string& filepath);
    static bool IsFileExist(const std::string& path);

private:
    FileUtils() = default;
    ~FileUtils() = default;
    static int32_t Open(const std::string& path, const int32_t flags, const mode_t mode);
};
} // namespace RuntimeKb
#endif
