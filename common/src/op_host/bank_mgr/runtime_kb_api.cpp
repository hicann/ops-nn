/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime_kb_api.h"
#include "runtime_bank_manager.h"
#include "kb_log.h"

namespace Ops {
namespace NN {
uint32_t QueryBank(const void* src, size_t src_len, const std::string& op_type, const std::string& soc_version,
                   uint32_t core_num, tuningtiling::TuningTilingDefPtr& tiling)
{
    if (core_num == 0U || soc_version.empty()) {
        CANNKB_LOGE("Platform Info is invalid: socVersion = %s, coreNum = %u", soc_version.c_str(), core_num);
        return RuntimeKb::FAILED;
    }
    RuntimeKb::PlatformInfo platform(core_num, soc_version);
    auto status = RuntimeKb::RuntimeBankManager::Instance().Query(src, src_len, op_type, platform, tiling);
    return static_cast<uint32_t>(status);
}
} // namespace NN
} // namespace Ops
