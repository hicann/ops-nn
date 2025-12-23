/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "common/inc/legacy_common_manager.h"

#include <dlfcn.h>
#include <dirent.h>
#include <limits.h>
#include "log/log.h"

// opapi、ophost ut场景下无法通过当前所在so的相对位置找到目标so，只能依赖环境变量ASCEND_OPP_PATH去找
namespace {
static const std::string BUILTIN_SO_NAME = "libophost_nn.so";
static const std::string CUSTOM_SO_NAME = "libcust_opmaster_rt2.0.so";
static const std::string LEGACY_SO_NAME = "libophost_comm_legacy.so";
static const Ops::NN::LegacyCommonMgr LEGACY_COMMMON_MGR;
}

namespace Ops {
namespace NN {
const LegacyCommonMgr& LegacyCommonMgr::GetInstance()
{
    // 简单修改，从单例改为返回静态常量，确保本so在被dlopen时就触发对libophost_comm_legacy.so的dlopen
    // GE的atc转om场景下，om在推理时so被释放到临时目录下，加载完删除，因此不能只靠单例被业务侧调用来触发对legacy的dlopen
    return LEGACY_COMMMON_MGR;
}

LegacyCommonMgr::LegacyCommonMgr()
{
    handle_ = nullptr;
    std::string soPath;
    if (GetLegacyCommonSoPath(soPath)) {
        handle_ = dlopen(soPath.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (handle_ == nullptr) {
            OP_LOGW("LegacyCommonMgr", "Fail to dlopen %s, reason: %s.", soPath.c_str(), dlerror());
        }
    }
}

LegacyCommonMgr::~LegacyCommonMgr()
{
    if (handle_ != nullptr) {
        (void)dlclose(handle_);
        handle_ = nullptr;
    }
}

bool LegacyCommonMgr::GetLegacyCommonSoPath(std::string& soPath) const
{
    // 考虑自定义算子包场景，无法根据当前so所在路径取找，只能通过环境变量去找
    char oppParentPath[PATH_MAX] = {0};
    auto ascendOppPath = std::getenv("ASCEND_OPP_PATH");
    if (ascendOppPath == nullptr) {
        OP_LOGW("LegacyCommonMgr", "Fail to get opp path from env.");
        return false;
    }

    if (realpath(ascendOppPath, oppParentPath) == nullptr) {
        OP_LOGW("LegacyCommonMgr", "Fail to get opp realpath.");
        return false;
    }

    // 只执行一次，性能不敏感
    soPath = std::string(oppParentPath) + std::string("/built-in/op_impl/ai_core/tbe/op_host/lib/linux/") +
             GetCpuArch() + std::string("/libophost_comm_legacy.so");
    OP_LOGI("LegacyCommonMgr", "so path spell with env is [%s].", soPath.c_str());

    return true;
}

std::string LegacyCommonMgr::GetCpuArch() const
{
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#else
    return "x86_64";
#endif
}

} // namespace NN
} // namespace Ops
