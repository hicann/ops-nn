/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "configuration.h"
#include <fstream>
#include <sstream>
#include <dlfcn.h>

#include "kb_log.h"
#include "file_utils.h"
#include "system_utils.h"

namespace RuntimeKb {
namespace {
constexpr char kHome[] = "HOME";
constexpr char kTuneBankPath[] = "TUNE_BANK_PATH";
constexpr char kAscendCachePath[] = "ASCEND_CACHE_PATH";
} // namespace

Configuration& Configuration::Instance()
{
    static Configuration config;
    return config;
}

Status Configuration::InitCannKbLibPath()
{
    Dl_info dlinfo;
    Configuration& (*instance_ptr)() = &Configuration::Instance;
    RTKB_CHECK(instance_ptr == nullptr, CANNKB_LOGE("Get configuration instance failed."), return FAILED);
    if (dladdr(reinterpret_cast<void*>(instance_ptr), &dlinfo) == 0) {
        CANNKB_LOGE("Can not find cannkb lib");
        return FAILED;
    }
    std::string so_path = dlinfo.dli_fname;
    CANNKB_LOGD("Cannkb lib so file path is %s.", so_path.c_str());
    if (so_path.empty()) {
        CANNKB_LOGE("Can not find cannkb lib.");
        return FAILED;
    }
    cann_kb_libpath_ = SystemUtils::RealPath(so_path);
    std::string::size_type pos = cann_kb_libpath_.rfind("/");
    if (pos == std::string::npos) {
        CANNKB_LOGE("Can not find cannkb lib.");
        return FAILED;
    }
    cann_kb_libpath_ = cann_kb_libpath_.substr(0, pos + 1);
    CANNKB_LOGD("Real path of cannkb lib is %s", cann_kb_libpath_.c_str());
    return SUCCESS;
}

Status Configuration::InitEnvPath(const char* env, std::string& real_path) const
{
    const std::string path = SystemUtils::GetEnv(env);
    real_path = "";
    RTKB_CHECK(path.empty(), CANNKB_LOGD("The environment variable called %s is empty.", env), return SUCCESS);
    real_path = SystemUtils::RealPath(path);
    RTKB_CHECK(real_path.empty(), CANNKB_LOGE("Realpath is empty"), return FAILED);
    real_path += "/";
    CANNKB_LOGD("%s path by user is %s.", env, real_path.c_str());
    return SUCCESS;
}

Status Configuration::Initialize()
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (is_init_) {
        return SUCCESS;
    }
    Status status = InitCannKbLibPath();
    if (status != SUCCESS) {
        CANNKB_LOGE("Failed to initialize the real path of cannkb lib.");
        return FAILED;
    }
    if (InitEnvPath(kTuneBankPath, tune_bank_path_) != SUCCESS) {
        CANNKB_LOGE("Init tune bank path failed.");
        return FAILED;
    }
    if (tune_bank_path_.empty()) {
        CANNKB_LOGD("Cannot find TUNE_BANK_PATH, check ASCEND_CACHE_PATH instead.");
        if (InitEnvPath(kAscendCachePath, ascend_cache_path_) != SUCCESS) {
            CANNKB_LOGE("Init ascend cache path failed.");
            return FAILED;
        }
        if (!ascend_cache_path_.empty()) {
            ascend_cache_path_ += "aoe_data/";
            if (SystemUtils::CreateMultiDirectory(ascend_cache_path_) != SUCCESS) {
                CANNKB_LOGE("Init ascend cache path failed.");
                return FAILED;
            }
        }
    }
    if (InitEnvPath(kHome, home_path_) != SUCCESS) {
        CANNKB_LOGE("Init home path failed");
        return FAILED;
    }
    is_init_ = true;
    return SUCCESS;
}

std::string Configuration::GetTuneBankPath() const { return tune_bank_path_; }

std::string Configuration::GetHomePath() const { return home_path_; }

std::string Configuration::GetAscendCachePath() const { return ascend_cache_path_; }

std::string Configuration::GetBuiltInBankPath() const
{
    const std::string CONFIG_BANK_RELATIVE_PATH = "../../../../../../../data/op/";
    return cann_kb_libpath_ + CONFIG_BANK_RELATIVE_PATH;
}

void Configuration::Finalize()
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!is_init_) {
        return;
    }
    home_path_.clear();
    tune_bank_path_.clear();
    cann_kb_libpath_.clear();
    is_init_ = false;
    return;
}
} // namespace RuntimeKb
