/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "runtime_bank_manager.h"

#include <set>
#include <sstream>
#include <climits>
#include "nlohmann/json.hpp"
#include "register/tuning_bank_key_registry.h"
#include "register/op_binary_resource_manager.h"

#include "op_hash.h"
#include "kb_log.h"
#include "kb_common.h"
#include "file_utils.h"
#include "system_utils.h"
#include "configuration.h"

namespace RuntimeKb {
static uint32_t core_num_ = 1U;
static std::string soc_version_ = "Ascend910B1";
constexpr char kRepoSuffix[] = "_runtime_kb.json";
const std::set<std::string> kOpWhiteLists{"DynamicRNN",
                                          "MatMulV2",
                                          "Conv2DBackpropInput",
                                          "Conv2DBackpropFilter",
                                          "Conv2DTranspose",
                                          "Conv2D",
                                          "Conv3DBackpropInput",
                                          "Conv3DTranspose",
                                          "Conv3D",
                                          "BatchMatmulFixpipe",
                                          "FFN",
                                          "MatMulV3",
                                          "QuantBatchMatmulV3",
                                          "Conv2DV2",
                                          "Conv3DV2",
                                          "Mc2MatMulV3",
                                          "Mc2QuantBatchMatmulV3",
                                          "Conv3DBackpropInputV2",
                                          "Conv3DBackpropFilterV2"};
const std::map<std::string, std::string> OP_TYPE_MAP{
    {"MatMul", "MatMulV2"}, {"MatMulV2", "MatMulV2"}, {"BatchMatMul", "MatMulV2"}, {"BatchMatMulV2", "MatMulV2"}};
static std::string GetBankFileName(const std::string& soc_version, const std::string& optype, uint32_t core_num)
{
    std::stringstream bank_file_name;
    bank_file_name << soc_version << "_" << core_num << "_AiCore_" << optype << kRepoSuffix;
    return bank_file_name.str();
}

static Status CheckGivenCustomPath(std::string& custom_dir, const std::string& bank_file)
{
    std::string custom_file_path = custom_dir + bank_file;
    if (!FileUtils::IsFileExist(custom_file_path)) {
        custom_dir = "";
        return SUCCESS;
    }
    if (!FileUtils::IsFileAccessedWithSafePermission(custom_file_path)) {
        CANNKB_LOGE("Custom repo file %s have not permission.", custom_file_path.c_str());
        return FAILED;
    }
    CANNKB_LOGI("Get custom repo path: %s.", custom_dir.c_str());
    return SUCCESS;
}

static void GetUserCustomRepoPath(std::string& home_path, const std::string& plat)
{
    std::string path = Configuration::Instance().GetHomePath();
    if (path.empty()) {
        return;
    }
    home_path = path + "/Ascend/cann/data/aoe/custom/op/" + plat + "/unified_bank/";
    return;
}

static Status GetAtcCustomPath(const std::string& plat, const std::string& bank_file, std::string& custom_dir)
{
    std::string tune_bank_path = Configuration::Instance().GetTuneBankPath();
    if (!tune_bank_path.empty()) {
        custom_dir = tune_bank_path + "/" + plat + "/unified_bank/";
        return CheckGivenCustomPath(custom_dir, bank_file);
    }
    std::string ascend_cache_path = Configuration::Instance().GetAscendCachePath();
    if (!ascend_cache_path.empty()) {
        custom_dir = ascend_cache_path + "/" + plat + "/unified_bank/";
        return CheckGivenCustomPath(custom_dir, bank_file);
    }
    GetUserCustomRepoPath(custom_dir, plat);
    std::string custom_bank_file = custom_dir + bank_file;
    if (custom_dir.empty() || !FileUtils::IsFileAccessedWithSafePermission(custom_bank_file)) {
        CANNKB_LOGW("Get one custom repo file %s failed.", custom_dir.c_str());
        custom_dir = "";
        return SUCCESS;
    }
    CANNKB_LOGD("Get custom repo path: %s.", custom_dir.c_str());
    return SUCCESS;
}

static Status GetAoeCustomPath(const std::string& plat, std::string& custom_dir)
{
    custom_dir = Configuration::Instance().GetTuneBankPath();
    if (!custom_dir.empty()) {
        custom_dir = custom_dir + "/" + plat + "/unified_bank/";
        return SUCCESS;
    }
    custom_dir = Configuration::Instance().GetAscendCachePath();
    if (!custom_dir.empty()) {
        custom_dir = custom_dir + "/" + plat + "/unified_bank/";
        return SUCCESS;
    }
    GetUserCustomRepoPath(custom_dir, plat);
    if (custom_dir.empty()) {
        CANNKB_LOGE("TUNE_BANK_PATH or ASCEND_CACHE_PATH or HOME must be assigned.");
        return FAILED;
    }
    return SUCCESS;
}

RuntimeBankManager& RuntimeBankManager::Instance()
{
    static RuntimeBankManager inst;
    return inst;
}

Status RuntimeBankManager::InitAoeOpBank(const PlatformInfo& plat, const std::set<std::string>& opLists)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (Configuration::Instance().Initialize() != SUCCESS) {
        return FAILED;
    }
    soc_version_ = plat.soc_version;
    core_num_ = plat.core_num;
    bank_prefix_ = soc_version_ + std::to_string(core_num_);

    std::set<std::string> opWhiteLists;
    std::set_intersection(opLists.cbegin(), opLists.cend(), kOpWhiteLists.cbegin(), kOpWhiteLists.cend(),
                          std::inserter(opWhiteLists, opWhiteLists.begin()));
    for (const auto& op : opWhiteLists) {
        if (InitAoeOpBank(op) != SUCCESS) {
            CANNKB_LOGE("Init aoe op %s bank %s failed.", op.c_str(), bank_prefix_.c_str());
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RuntimeBankManager::InitAoeOpBank(const std::string& op)
{
    std::string custom_path;
    Status ret = GetAoeCustomPath(soc_version_, custom_path);
    if (ret != SUCCESS) {
        CANNKB_LOGE("Get aoe custom path %s failed.", custom_path.c_str());
        return FAILED;
    }
    const std::string& bank_file = GetBankFileName(soc_version_, op, core_num_);
    if (!custom_path.empty()) {
        custom_path += bank_file;
    }
    const std::string& default_bank_dir = Configuration::Instance().GetBuiltInBankPath();
    const std::string& builtin_path = default_bank_dir + "/" + soc_version_ + "/unified_bank/" + bank_file;
    const std::string& data_bank_dir = default_bank_dir + "../../../../ops_nn/built-in/data/op/";
    const std::string& data_builtin_path = data_bank_dir + "/" + soc_version_ + "/unified_bank/" + bank_file;
    if (FileUtils::IsFileExist(data_builtin_path)) {
        return InitOpBank(op, custom_path, data_builtin_path, RuntimeBankType::AOE);
    }
    const std::string& transformer_bank_dir = default_bank_dir + "../../../../ops_transformer/built-in/data/op/";
    const std::string& transformer_builtin_path = transformer_bank_dir + "/" + soc_version_ + "/unified_bank/" +
                                                  bank_file;
    if (FileUtils::IsFileExist(transformer_builtin_path)) {
        return InitOpBank(op, custom_path, transformer_builtin_path, RuntimeBankType::AOE);
    }
    return InitOpBank(op, custom_path, builtin_path, RuntimeBankType::AOE);
}

void RuntimeBankManager::InitAtcOpBank(const PlatformInfo& plat, const std::string& op)
{
    rwlock_.wrlock();
    (void)Configuration::Instance().Initialize();
    soc_version_ = plat.soc_version;
    core_num_ = plat.core_num;
    bank_prefix_ = soc_version_ + std::to_string(core_num_);
    std::string custom_path;
    const std::string& file = GetBankFileName(soc_version_, op, core_num_);
    Status ret = GetAtcCustomPath(soc_version_, file, custom_path);
    if (ret != SUCCESS) {
        CANNKB_LOGE("Get atc custom path %s failed.", custom_path.c_str());
        rwlock_.unlock();
        return;
    }
    if (!custom_path.empty()) {
        custom_path += file;
    }
    std::string default_bank_dir = Configuration::Instance().GetBuiltInBankPath();
    std::string data_bank_dir = default_bank_dir + "../../../../ops_nn/built-in/data/op/";
    std::string builtin_path = default_bank_dir + "/" + soc_version_ + "/unified_bank/" + file;
    std::string data_builtin_path = data_bank_dir + "/" + soc_version_ + "/unified_bank/" + file;
    std::string transformer_bank_dir = default_bank_dir + "../../../../ops_transformer/built-in/data/op/";
    std::string transformer_builtin_path = transformer_bank_dir + "/" + soc_version_ + "/unified_bank/" + file;
    if (FileUtils::IsFileExist(data_builtin_path)) {
        (void)InitOpBank(op, custom_path, data_builtin_path, RuntimeBankType::ATC);
    } else if (FileUtils::IsFileExist(transformer_builtin_path)) {
        (void)InitOpBank(op, custom_path, transformer_builtin_path, RuntimeBankType::ATC);
    } else {
        (void)InitOpBank(op, custom_path, builtin_path, RuntimeBankType::ATC);
    }
    rwlock_.unlock();
    return;
}

inline std::string GenerateUniqueId(const std::string& bank_prefix, const std::string& optype)
{
    return bank_prefix + optype;
}

static void JsonStrParser(const std::string& soc_version, const std::string& core_num,
                          const std::vector<ge::AscendString>& json_vec, std::vector<std::string>& kb_str)
{
    std::string plat_info = soc_version + "_" + core_num;
    for (const ge::AscendString& json : json_vec) {
        std::stringstream temp(json.GetString());
        std::string line;
        int32_t idx = 0;
        while (getline(temp, line, '\n')) {
            if (idx == 0) {
                if (line.find(plat_info) == std::string::npos) {
                    CANNKB_LOGD("Repo %s is not ok for paltform: %s.", line.c_str(), plat_info.c_str());
                    break;
                } else {
                    idx += 1;
                    continue;
                }
            }
            static_cast<void>(kb_str.emplace_back(line));
        }
    }
}

Status RuntimeBankManager::InitOpBank(const std::string& op, const std::string& custom_path,
                                      const std::string& builtin_path, RuntimeBankType type)
{
    std::stringstream index;
    index << custom_path << soc_version_ << core_num_ << op << static_cast<uint32_t>(type);
    const auto& it = bank_cache_.find(index.str());
    if (it != bank_cache_.cend()) {
        return SUCCESS;
    }

    std::vector<ge::AscendString> json_vec;
    (void)nnopbase::OpBinaryResourceManager::GetInstance().GetOpRuntimeKB(ge::AscendString(op.c_str()), json_vec);
    std::vector<std::string> kb_str;
    if (!json_vec.empty()) {
        JsonStrParser(soc_version_, std::to_string(core_num_), json_vec, kb_str);
    }
    CANNKB_LOGD("GetOpRuntimeKB by static lib: op[%s], soc[%s], core_num[%u], json_vec_size = %zu, kb_str_size = %zu",
                op.c_str(), soc_version_.c_str(), core_num_, json_vec.size(), kb_str.size());
    OpRuntimeBankPtr op_bank = nullptr;
    if (kb_str.empty()) {
        op_bank = MakeShared<OpRuntimeBank>(custom_path, builtin_path, type);
    } else {
        op_bank = MakeShared<OpRuntimeBank>(custom_path, builtin_path, type, kb_str);
    }
    if (op_bank == nullptr || op_bank->Initialize(type == RuntimeBankType::AOE) != SUCCESS) {
        CANNKB_LOGE("Init op bank failed.");
        return FAILED;
    }
    const auto& id = GenerateUniqueId(bank_prefix_, op);
    bank_id_[id].push_back(index.str());
    if (type == RuntimeBankType::AOE) {
        aoe_bank_id_[id] = index.str();
    }
    bank_cache_[index.str()] = op_bank;
    return SUCCESS;
}

std::vector<std::string> RuntimeBankManager::GetAllOpBankId(const std::string& optype,
                                                            const std::string& bank_prefix) const
{
    const auto& id = GenerateUniqueId(bank_prefix, optype);
    const auto& it = bank_id_.find(id);
    if (it == bank_id_.cend()) {
        return std::vector<std::string>{};
    }
    return it->second;
}

std::string RuntimeBankManager::GetAoeOpBankId(const std::string& optype) const
{
    const auto& id = GenerateUniqueId(bank_prefix_, optype);
    const auto& it = aoe_bank_id_.find(id);
    if (it == aoe_bank_id_.cend()) {
        return "";
    }
    return it->second;
}

Status RuntimeBankManager::Query(const void* src, size_t src_len, const std::string& optype, const PlatformInfo& plat,
                                 tuningtiling::TuningTilingDefPtr& tiling)
{
    RTKB_CHECK(src == nullptr, CANNKB_LOGE("Src is nullptr!"), return FAILED);
    rwlock_.rdlock();
    const auto bank_prefix = plat.soc_version + std::to_string(plat.core_num);
    auto bank_ids = GetAllOpBankId(optype, bank_prefix);
    if (bank_ids.empty()) {
        rwlock_.unlock();
        InitAtcOpBank(plat, optype);
        rwlock_.rdlock();
        bank_ids = GetAllOpBankId(optype, bank_prefix);
    }
    RTKB_CHECK(src_len > UINT32_MAX, CANNKB_LOGE("src_len is invaild!"), return FAILED);
    const uint32_t ky = CommonHash(src, static_cast<uint32_t>(src_len));
    for (const auto& id : bank_ids) {
        const uint32_t pid = SystemUtils::GetPid();
        if (bank_cache_[id]->Query(ky, pid, src, src_len, tiling) == SUCCESS) {
            rwlock_.unlock();
            return SUCCESS;
        }
    }
    CANNKB_LOGD("Not found in %s repo. K: %u", optype.c_str(), ky);
    rwlock_.unlock();
    return KEY_NOT_EXIST;
}

Status RuntimeBankManager::Query(const gert::TilingContext* op, const std::string& optype, const PlatformInfo& plat,
                                 tuningtiling::TuningTilingDefPtr& tiling)
{
    RTKB_CHECK(op == nullptr, CANNKB_LOGE("TilingContext is nullptr!"), return FAILED);
    auto& func = tuningtiling::OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
    auto iter = func.find(optype.c_str());
    if (iter == func.cend()) {
        return KEY_NOT_EXIST;
    }
    std::shared_ptr<void> in_args = nullptr;
    std::size_t in_args_size = 0;
    const auto& convert_func = iter->second.GetBankKeyConvertFuncV2();
    if (!convert_func(op, in_args, in_args_size)) {
        return KEY_NOT_EXIST;
    }
    return Query(in_args.get(), in_args_size, optype, plat, tiling);
}

Status RuntimeBankManager::Update(const gert::TilingContext* op, const std::string& optype,
                                  const tuningtiling::TuningTilingDefPtr& tiling)
{
    RTKB_CHECK(op == nullptr, CANNKB_LOGE("TilingContext is nullptr!"), return FAILED);
    RTKB_CHECK(tiling == nullptr, CANNKB_LOGE("Tiling is nullptr!"), return FAILED);
    const auto& aoe_bank_id = GetAoeOpBankId(optype);
    if (aoe_bank_id.empty()) {
        CANNKB_LOGE("Can not update %s bank!", optype.c_str());
        return FAILED;
    }
    auto& func = tuningtiling::OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
    auto iter = func.find(optype.c_str());
    if (iter == func.cend()) {
        CANNKB_LOGE("Can not find %s convert func.", optype.c_str());
        return FAILED;
    }
    std::shared_ptr<void> in_args = nullptr;
    std::size_t in_args_size = 0;
    const auto& convert_func = iter->second.GetBankKeyConvertFuncV2();
    if (!convert_func(op, in_args, in_args_size)) {
        return FAILED;
    }
    CANNKB_LOGD("Registry %s convert func successful.", optype.c_str());
    uint32_t ky = CommonHash(in_args.get(), in_args_size);
    return bank_cache_[aoe_bank_id]->Update(ky, in_args, in_args_size, optype, tiling);
}

Status RuntimeBankManager::Save()
{
    for (const auto& it : aoe_bank_id_) {
        if (bank_cache_[it.second]->Save() != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

Status RuntimeBankManager::SetTuningTiling(const gert::TilingContext* op, const std::string& optype,
                                           const std::string& tiling)
{
    std::lock_guard<std::mutex> lock(mtx_);
    RTKB_CHECK(op == nullptr, CANNKB_LOGE("TilingContext is nullptr!"), return FAILED);
    RTKB_CHECK(tiling.empty(), CANNKB_LOGW("Tiling is empty!"), return SUCCESS);
    const auto& aoe_bank_id = GetAoeOpBankId(optype);
    if (aoe_bank_id.empty()) {
        CANNKB_LOGE("Can not set tuning tiling %s bank.", optype.c_str());
        return FAILED;
    }
    auto& func = tuningtiling::OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
    auto iter = func.find(optype.c_str());
    if (iter == func.cend()) {
        CANNKB_LOGE("Can not find %s convert func.", optype.c_str());
        return FAILED;
    }
    std::shared_ptr<void> in_args = nullptr;
    std::size_t in_args_size = 0;
    const auto& convert_func = iter->second.GetBankKeyConvertFuncV2();
    if (!convert_func(op, in_args, in_args_size)) {
        return FAILED;
    }
    uint32_t ky = CommonHash(in_args.get(), in_args_size);
    nlohmann::json tiling_json;
    try {
        tiling_json = nlohmann::json::parse(tiling);
    } catch (nlohmann::json::exception& e) {
        CANNKB_LOGE("Set tuning tiling failed, err msg: %s", e.what());
        return FAILED;
    }
    auto tilingdef = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance(ge::AscendString(optype.c_str()));
    RTKB_CHECK(tilingdef == nullptr, CANNKB_LOGE("Can not find tuning tiling, op:%s!", optype.c_str()), return FAILED);
    tilingdef->FromJson(tiling_json);
    bank_cache_[aoe_bank_id]->SetTuningTiling(ky, tilingdef);
    return SUCCESS;
}

Status RuntimeBankManager::SetTuningTiling(const uint32_t pid, const std::string& optype, const std::string& tiling)
{
    RTKB_CHECK(tiling.empty(), CANNKB_LOGE("Tiling is empty!"), return FAILED);
    const auto bank_prefix = soc_version_ + std::to_string(core_num_);
    auto bank_ids = GetAllOpBankId(optype, bank_prefix);
    if (bank_ids.empty()) {
        PlatformInfo plat(core_num_, soc_version_);
        InitAtcOpBank(plat, optype);
        bank_ids = GetAllOpBankId(optype, bank_prefix);
        if (bank_ids.empty()) {
            CANNKB_LOGE("Can not set tuning tiling %s bank, because bankIds is empty.", optype.c_str());
            return FAILED;
        }
    }
    const auto& aoe_bank_id = bank_ids.front();
    if (aoe_bank_id.empty()) {
        CANNKB_LOGE("Can not set tuning tiling %s bank.", optype.c_str());
        return FAILED;
    }
    nlohmann::json tiling_json;
    try {
        tiling_json = nlohmann::json::parse(tiling);
    } catch (nlohmann::json::exception& e) {
        CANNKB_LOGE("Set tuning tiling failed, err msg: %s", e.what());
        return FAILED;
    }
    auto tilingdef = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance(ge::AscendString(optype.c_str()));
    RTKB_CHECK(tilingdef == nullptr, CANNKB_LOGE("Can not find tuning tiling, op:%s!", optype.c_str()), return FAILED);
    tilingdef->FromJson(tiling_json);
    bank_cache_[aoe_bank_id]->SetTuningTiling(pid, tilingdef);
    CANNKB_LOGD("Set tuning tiling successfully, pid = %u optype = %s tiling = %s.", pid, optype.c_str(),
                tiling.c_str());
    return SUCCESS;
}
} // namespace RuntimeKb
