/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "file_utils.h"

#include <cerrno>
#include <climits>
#include <fcntl.h>

#include "kb_log.h"
#include "kb_common.h"
#include "system_utils.h"

namespace RuntimeKb {
template <typename T>
Status FileUtils::OpenWithSafePermission(T& filestream, const std::string& filename, std::ios_base::openmode mode)
{
    RTKB_CHECK(filename.empty() || filename.size() >= PATH_MAX, CANNKB_LOGE("Invalid filename"), return FAILED);
    char resolved_path[PATH_MAX] = {0x00};
    std::string open_path = filename;
    if (realpath(filename.c_str(), resolved_path) != nullptr) {
        open_path = resolved_path;
    } else if (errno != ENOENT) {
        CANNKB_LOGE("Real path file failed: %s", filename.c_str());
        return FAILED;
    }
    int fd = open(open_path.c_str(), O_RDWR | O_CREAT, kSafePermission);
    if (fd < 0) {
        CANNKB_LOGE("Open file: %s failed", filename.c_str());
        return FAILED;
    }
    close(fd);

    filestream.open(open_path, mode);

    if (!filestream.good()) {
        CANNKB_LOGW("Open file: %s failed", filename.c_str());
        return FAILED;
    }
    return SUCCESS;
}

bool FileUtils::IsFileAccessedWithSafePermission(const std::string& filepath)
{
    RTKB_CHECK(filepath.empty() || filepath.size() >= PATH_MAX, CANNKB_LOGE("Invalid filepath"), return false);
    int fd = FileUtils::Open(filepath, O_RDWR, kSafePermission);
    if (fd < 0) {
        return false;
    }
    close(fd);
    return true;
}

bool FileUtils::IsFileExist(const std::string& path)
{
    if (access(path.c_str(), F_OK) != 0) {
        return false;
    }
    return true;
}

int32_t FileUtils::Open(const std::string& path, const int32_t flags, const mode_t mode)
{
    RTKB_CHECK(path.empty(), CANNKB_LOGE("Path %s is empty.", path.c_str()), return -1);
    auto fd = open(path.c_str(), flags, mode);
    return fd;
}

template Status FileUtils::OpenWithSafePermission(std::fstream&, const std::string&, std::ios_base::openmode);
template Status FileUtils::OpenWithSafePermission(std::ofstream&, const std::string&, std::ios_base::openmode);
} // namespace RuntimeKb
