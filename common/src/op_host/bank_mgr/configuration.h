/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_COMMON_UTILS_CONFIGURATION_UTILS_H_
#define RUNTIME_KB_COMMON_UTILS_CONFIGURATION_UTILS_H_

#include <mutex>
#include <string>

#include "kb_status.h"
#include "kb_common.h"

namespace RuntimeKb {
class Configuration {
public:
    Configuration(const Configuration&) = delete;

    Configuration& operator=(const Configuration&) = delete;

    static Configuration& Instance();

    Status Initialize();

    std::string GetBuiltInBankPath() const;

    std::string GetTuneBankPath() const;

    std::string GetAscendHomePath() const;

    std::string GetHomePath() const;

    std::string GetAscendCachePath() const;

    std::string GetCannKbLibPath() const;

    void Finalize();

private:
    Configuration() : is_init_(false), cann_kb_libpath_() {}
    ~Configuration() {};
    Status InitCannKbLibPath();
    Status InitEnvPath(const char* env, std::string& real_path) const;
    bool is_init_;
    std::string cann_kb_libpath_;
    std::string tune_bank_path_;
    std::string ascend_cache_path_;
    std::string home_path_;
    std::mutex mtx_;
};
} // namespace RuntimeKb
#endif
