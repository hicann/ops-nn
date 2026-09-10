/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "system_utils.h"

#include <list>
#include <sys/stat.h>
#include <fcntl.h>
#include <climits>
#include <unistd.h>
#include <cstdlib>
#include "kb_log.h"
#include "kb_common.h"

namespace RuntimeKb {
const std::string kLinuxFileSeperator = "/";
constexpr mode_t kCreateDirPermission = S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP;

std::string SystemUtils::RealPath(const std::string& path)
{
    std::string res;
    if (path.empty()) {
        CANNKB_LOGW("Path string is nullptr.");
        return res;
    }
    if (path.size() >= PATH_MAX) {
        CANNKB_LOGW("File path %s is too long.", path.c_str());
        return res;
    }

    char resolve_path[PATH_MAX] = {0x00};

    if (realpath(path.c_str(), resolve_path) != nullptr) {
        res = resolve_path;
    } else {
        CANNKB_LOGW("Path %s is not exist.", path.c_str());
    }
    return res;
}

std::string SystemUtils::GetParentDir(const std::string& path)
{
    std::string dir_path;
    auto last_slash_idx = path.rfind(kLinuxFileSeperator);
    if (last_slash_idx != std::string::npos) {
        dir_path = path.substr(0, last_slash_idx);
    } else {
        dir_path = "";
    }
    return dir_path;
}

bool SystemUtils::CheckDirExistenceAndPermission(const std::string& path)
{
    std::string dir_path = GetParentDir(path);
    RTKB_CHECK(!IsDirExist(dir_path), CANNKB_LOGE("Dir %s is not exist.", dir_path.c_str()), return false);
    if (access(dir_path.c_str(), R_OK | W_OK | X_OK) >= 0) {
        return true;
    }
    return false;
}

bool SystemUtils::IsDirExist(const std::string& path)
{
    RTKB_CHECK(path.empty(), CANNKB_LOGE("Path is empty."), return false);
    CANNKB_LOGD("Check dir: %s existence and permission.", path.c_str());
    struct stat status;
    if (stat(path.c_str(), &status) == 0) {
        if ((status.st_mode & S_IFDIR) != 0) {
            return true;
        }
    }
    CANNKB_LOGD("Dir %s is not exist.", path.c_str());
    return false;
}

std::string SystemUtils::GetEnv(const std::string& k)
{
    const char* val = getenv(k.c_str());
    return val == nullptr ? std::string() : std::string(val);
}

Status SystemUtils::CreateMultiDirectory(const std::string& dir)
{
    RTKB_CHECK(dir.empty(), CANNKB_LOGE("Given directory is empty."), return FAILED);
    if (SystemUtils::IsDirExist(dir)) {
        return SUCCESS;
    }
    std::list<std::string> dir_lists;
    dir_lists.push_front(dir);

    std::string cur_dir = dir;
    std::string parent_dir = SystemUtils::GetParentDir(cur_dir);
    while (parent_dir != cur_dir) {
        if (SystemUtils::IsDirExist(parent_dir)) {
            break;
        }
        dir_lists.push_front(parent_dir);
        cur_dir = parent_dir;
        parent_dir = SystemUtils::GetParentDir(cur_dir);
    }

    for (const auto& it : dir_lists) {
        if (mkdir(it.c_str(), kCreateDirPermission) != 0 && errno != EEXIST) {
            CANNKB_LOGE("Create directory %s failed.", it.c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

int32_t SystemUtils::GetCpuCoreNum()
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) {
        CANNKB_LOGE("GetCpuInfo Failed.");
        return 0;
    }
    return static_cast<int32_t>(n);
}

uint32_t SystemUtils::GetPid()
{
    pid_t pid = getpid();
    if (pid >= 0) {
        CANNKB_LOGD("The current pid is %d.", pid);
        return static_cast<uint32_t>(pid);
    }
    return UINT_MAX;
}
} // namespace RuntimeKb
