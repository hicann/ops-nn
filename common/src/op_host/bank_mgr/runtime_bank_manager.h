/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_RUNTIME_BANK_MANAGER_H_
#define RUNTIME_KB_RUNTIME_BANK_MANAGER_H_
#include <set>
#include <unordered_map>
#include <string>
#include <mutex>
#include "lock.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/tuning_tiling_registry.h"
#include "op_runtime_bank.h"
#include "kb_status.h"

namespace RuntimeKb {
struct PlatformInfo {
    PlatformInfo() = default;
    PlatformInfo(uint32_t in_core_num, const std::string& in_soc_version)
        : core_num(in_core_num), soc_version(in_soc_version)
    {}
    uint32_t core_num = 1U;
    std::string soc_version;
};
class RuntimeBankManager {
public:
    static RuntimeBankManager& Instance();
    Status InitAoeOpBank(const PlatformInfo& plat, const std::set<std::string>& opLists);
    Status Query(const void* src, size_t src_len, const std::string& optype, const PlatformInfo& plat,
                 tuningtiling::TuningTilingDefPtr& tiling);
    Status Query(const gert::TilingContext* op, const std::string& optype, const PlatformInfo& plat,
                 tuningtiling::TuningTilingDefPtr& tiling);
    Status Update(const gert::TilingContext* op, const std::string& optype,
                  const tuningtiling::TuningTilingDefPtr& tiling);
    Status Save();
    Status SetTuningTiling(const gert::TilingContext* op, const std::string& optype, const std::string& tiling);
    Status SetTuningTiling(const uint32_t pid, const std::string& optype, const std::string& tiling);

private:
    Status InitAoeOpBank(const std::string& op);
    void InitAtcOpBank(const PlatformInfo& plat, const std::string& op);
    Status InitOpBank(const std::string& op, const std::string& custom_path, const std::string& builtin_path,
                      RuntimeBankType type);
    std::vector<std::string> GetAllOpBankId(const std::string& optype, const std::string& bank_prefix) const;
    std::string GetAoeOpBankId(const std::string& optype) const;
    RuntimeBankManager() = default;
    ~RuntimeBankManager() = default;
    RuntimeBankManager(const RuntimeBankManager&) = delete;
    RuntimeBankManager& operator=(const RuntimeBankManager&) = delete;
    std::string bank_prefix_;
    Ops::NN::HostTiling::RWLock rwlock_;
    std::mutex mtx_;
    std::unordered_map<std::string, OpRuntimeBankPtr> bank_cache_;
    std::unordered_map<std::string, std::vector<std::string>> bank_id_;
    std::unordered_map<std::string, std::string> aoe_bank_id_;
};
} // namespace RuntimeKb
#endif
