/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "op_runtime_bank.h"

#include <sys/file.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <climits>
#include <map>
#include "error_util.h"
#include "kb_log.h"
#include "kb_common.h"
#include "file_utils.h"
#include "system_utils.h"
#include "thread_pool.h"
#include "op_hash.h"

namespace RuntimeKb {
namespace {
constexpr char kOpId[] = "id";
constexpr char kOpType[] = "op";
constexpr char kVersion[] = "version";
constexpr char kFeature[] = "info_dict";
constexpr char kTiling[] = "knowledge";
} // namespace

std::string Record::Serialize()
{
    try {
        nlohmann::json json_record;
        json_record[kOpId] = ky;
        json_record[kOpType] = optype;
        json_record[kVersion] = version;
        if (tiling == nullptr || ori_ky == nullptr) {
            return "";
        }
        auto& func = tuningtiling::OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
        auto iter = func.find(optype.c_str());
        if (iter == func.cend()) {
            return "";
        }
        ge::AscendString kfeatureAs;
        const auto& parseFunc = iter->second.GetBankKeyParseFuncV2();
        if (!parseFunc(ori_ky, bank_key_size, kfeatureAs)) {
            return "";
        }
        json_record[kFeature] = nlohmann::json::parse(kfeatureAs.GetString());
        if (json_record[kFeature].empty()) {
            CANNKB_LOGE("Failed to get kFeature!");
            return "";
        }
        tiling->ToJson(json_record[kTiling]);
        return json_record.dump();
    } catch (nlohmann::json::exception& e) {
        CANNKB_LOGE("Serialize to json failed! Error message: %s.", e.what());
        return "";
    }
}

bool Record::Deserialize(const std::string& record)
{
    try {
        if (!nlohmann::json::accept(record)) {
            CANNKB_LOGE("Invalid json! File: %s.", record.c_str());
            return false;
        }
        nlohmann::json json_val = nlohmann::json::parse(record);
        ky = json_val.at(kOpId).get<uint32_t>();
        optype = json_val.at(kOpType).get<std::string>();
        version = json_val.at(kVersion).get<uint32_t>();
        nlohmann::json oriKyJson = json_val.at(kFeature).get<nlohmann::json>();
        ge::AscendString oriKyJsonStr = ge::AscendString(oriKyJson.dump().c_str());
        nlohmann::json tiling_json = json_val.at(kTiling).get<nlohmann::json>();
        tiling = tuningtiling::TuningTilingClassFactory::CreateTilingDataInstance(optype.c_str());
        auto& func = tuningtiling::OpBankKeyFuncRegistryV2::RegisteredOpFuncInfoV2();
        auto iter = func.find(optype.c_str());
        if (iter == func.cend()) {
            return false;
        }
        const auto& loadFunc = iter->second.GetBankKeyLoadFuncV2();
        if (!loadFunc(ori_ky, bank_key_size, oriKyJsonStr)) {
            return false;
        }
        if (tiling == nullptr || ori_ky == nullptr) {
            CANNKB_LOGE("Deserialize file failed, can not find optype: %s", optype.c_str());
            return false;
        }
        tiling->FromJson(tiling_json);
        return true;
    } catch (nlohmann::json::exception& e) {
        CANNKB_LOGE("Parse from json failed! Error message: %s.", e.what());
        return false;
    }
}

OpRuntimeBank::OpRuntimeBank(const std::string& custom_path, const std::string& builtin_path,
                             const RuntimeBankType type)
    : custom_path_(custom_path), builtin_path_(builtin_path), type_(type)
{}

OpRuntimeBank::OpRuntimeBank(const std::string& custom_path, const std::string& builtin_path,
                             const RuntimeBankType type, const std::vector<std::string>& kb_str)
    : custom_path_(custom_path), builtin_path_(builtin_path), type_(type), kb_str_(kb_str)
{}

Status OpRuntimeBank::UnpackageStaticLib()
{
    builtin_->clear();
    for (const std::string& line : kb_str_) {
        Record record;
        if (!record.Deserialize(line)) {
            CANNKB_LOGE("Failed to deserialize repo line %s.", line.c_str());
            return FAILED;
        }
        uint32_t op_hash = CommonHash(record.ori_ky.get(), record.bank_key_size);
        record.ky = op_hash;
        (void)builtin_->insert({op_hash, record});
    }
    CANNKB_LOGD("Get runtime kb form static lib successfully, builtin_kb_size = %zu.", builtin_->size());
    return SUCCESS;
}

Status OpRuntimeBank::Initialize(bool check_custom_path)
{
    std::lock_guard<std::mutex> lock(mtx_);
    if (!kb_str_.empty() && UnpackageStaticLib() != SUCCESS) {
        CANNKB_LOGE("Failed to load from static lib!");
        return FAILED;
    }
    if (builtin_->empty() && LoadFromFile(builtin_path_, true) != SUCCESS) {
        CANNKB_LOGE("Failed to load from file: %s!", builtin_path_.c_str());
        return FAILED;
    }
    if (type_ == RuntimeBankType::ATC) {
        if (custom_path_.empty()) {
            CANNKB_LOGD("Custom repo file %s not found, return. Init op runtime bank succeeded.", custom_path_.c_str());
            return SUCCESS;
        }
        std::fstream fs;
        Status ret = FileUtils::OpenWithSafePermission(fs, custom_path_, std::ios::in | std::ios::out | std::ios::app);
        if (ret != SUCCESS) {
            CANNKB_LOGW("Open custom repo file failed. file: %s.", custom_path_.c_str());
            return SUCCESS;
        }
    }
    bool check_path = true;
    if (type_ == RuntimeBankType::AOE) {
        std::string custom_dir = SystemUtils::GetParentDir(custom_path_);
        if (!SystemUtils::IsDirExist(custom_dir)) {
            check_path = false;
        } else {
            if (SystemUtils::CreateMultiDirectory(custom_dir) != SUCCESS) {
                CANNKB_LOGE("Failed to create custom repo directory! No permission for %s.", custom_dir.c_str());
                return FAILED;
            }
        }
    }
    if (check_custom_path && check_path && !SystemUtils::CheckDirExistenceAndPermission(custom_path_)) {
        CANNKB_LOGE("No permission for %s.", custom_path_.c_str());
        ge::ReportPredefinedErrMsg("EC0011", {"custom_file_dir"}, {custom_path_.c_str()});
        return FAILED;
    }

    CustomRepoGuard guard(*this, false);

    if (check_path && LoadFromFile(custom_path_) != SUCCESS) {
        CANNKB_LOGE("Failed to load custom repo from file: %s.", custom_path_.c_str());
        return FAILED;
    }
    CANNKB_LOGD("Init model bank succeeded.");
    return SUCCESS;
}

static Status UnpackageFstream(std::shared_ptr<std::fstream> fs,
                               std::shared_ptr<std::unordered_map<uint32_t, Record>> repo, std::shared_ptr<bool> ready,
                               std::shared_ptr<std::mutex> control, std::shared_ptr<int8_t> async_actives)
{
    {
        std::lock_guard<std::mutex> lk(*control);
        if (*ready) {
            return SUCCESS;
        }
        (*async_actives)++;
    }
    while (true) {
        std::string line;
        {
            std::lock_guard<std::mutex> lk(*control);
            if ((*ready) || !getline(*fs, line)) {
                (*ready) = true;
                (*async_actives)--;
                break;
            }
        }
        Record record;
        if (!record.Deserialize(line)) {
            CANNKB_LOGE("Failed to deserialize repo line %s.", line.c_str());
            std::lock_guard<std::mutex> lk(*control);
            (*async_actives)--;
            return FAILED;
        }
        uint32_t op_hash = CommonHash(record.ori_ky.get(), record.bank_key_size);
        record.ky = op_hash;
        std::lock_guard<std::mutex> lk(*control);
        (void)repo->insert({op_hash, record});
    }
    return SUCCESS;
}

Status OpRuntimeBank::CheckObj()
{
    if (async_mtx_ == nullptr || async_ready_ == nullptr || async_ready_custom_ == nullptr ||
        async_actives_ == nullptr || async_actives_custom_ == nullptr || builtin_ == nullptr || custom_ == nullptr) {
        return FAILED;
    }
    return SUCCESS;
}

Status OpRuntimeBank::LoadFromFile(const std::string& file, bool builtin_repo)
{
    CANNKB_LOGD("Repo file %s.", file.c_str());
    if (!FileUtils::IsFileExist(file)) {
        CANNKB_LOGW("Repo file %s does not exist.", file.c_str());
        return SUCCESS;
    }
    std::shared_ptr<std::fstream> fs = MakeShared<std::fstream>();
    if (fs == nullptr || CheckObj() != SUCCESS) {
        CANNKB_LOGE("MakeShared Object failed.");
        return FAILED;
    }
    if (builtin_repo) {
        builtin_->clear();
        fs->open(file, std::ios::in);
        if (!fs->good()) {
            CANNKB_LOGE("Failed to open file: %s.", file.c_str());
            return FAILED;
        }
    } else {
        custom_->clear();
        Status ret = FileUtils::OpenWithSafePermission(*fs, file, std::ios::in | std::ios::out | std::ios::app);
        if (ret != SUCCESS) {
            CANNKB_LOGE("Failed to open repo file: %s.", file.c_str());
            return FAILED;
        }
    }
    rwlock_.wrlock();

    for (int32_t cnt = 0; cnt < std::min(ThreadPool::GetInstance()->GetSize(), DEFAULT_THREAD_POOL_NUM); cnt++) {
        (void)ThreadPool::GetInstance()->Submit(UnpackageFstream, fs, builtin_repo ? builtin_ : custom_,
                                                builtin_repo ? async_ready_ : async_ready_custom_, async_mtx_,
                                                builtin_repo ? async_actives_ : async_actives_custom_);
    }

    while (true) {
        std::lock_guard<std::mutex> lk(*async_mtx_);
        if ((builtin_repo) && (*async_ready_) && ((*async_actives_) == 0)) {
            break;
        }
        if ((!builtin_repo) && (*async_ready_custom_) && ((*async_actives_custom_) == 0)) {
            break;
        }
    }
    rwlock_.unlock();
    CANNKB_LOGD("Open repo file %s succeeded, record %zu.", file.c_str(), (builtin_repo ? builtin_ : custom_)->size());
    return SUCCESS;
}

Status OpRuntimeBank::Save()
{
    RTKB_CHECK(custom_ == nullptr, CANNKB_LOGE("custom_ is nullptr!"), return FAILED);
    if (custom_->empty()) {
        CANNKB_LOGD("no need to save %s.", custom_path_.c_str());
        return SUCCESS;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    CANNKB_LOGD("Saving tiling to file.");
    if (type_ == RuntimeBankType::AOE) {
        std::string custom_dir = SystemUtils::GetParentDir(custom_path_);
        if (SystemUtils::CreateMultiDirectory(custom_dir) != SUCCESS) {
            CANNKB_LOGE("Failed to create custom repo directory! No permission for %s.", custom_dir.c_str());
            return FAILED;
        }
        if (!SystemUtils::CheckDirExistenceAndPermission(custom_path_)) {
            CANNKB_LOGE("No permission for %s.", custom_path_.c_str());
            ge::ReportPredefinedErrMsg("EC0011", {"custom_file_dir"}, {custom_path_.c_str()});
            return FAILED;
        }
    }
    CustomRepoGuard guard(*this);
    std::ofstream ofs;
    Status ret = FileUtils::OpenWithSafePermission(ofs, custom_path_, std::ios::out);
    if (ret != SUCCESS) {
        CANNKB_LOGE("Failed to open file: %s.", custom_path_.c_str());
        return FAILED;
    }
    for (auto& index : *custom_) {
        std::string out_str = index.second.Serialize();
        if (out_str.empty()) {
            CANNKB_LOGE("Serialize tiling failed.");
            return FAILED;
        }
        ofs << out_str << std::endl;
    }
    CANNKB_LOGD("Flush repo file %s succeeded, recording.", custom_path_.c_str());
    return SUCCESS;
}
inline bool QuickCmp(const void* src, size_t src_size, const std::shared_ptr<void>& dst, size_t dst_size)
{
    if (src_size != dst_size) {
        return false;
    }
    return memcmp(src, dst.get(), dst_size) == 0;
}

RuntimeBankType OpRuntimeBank::GetType() { return type_; }

Status OpRuntimeBank::Query(uint32_t k, uint32_t pid, const void* src, size_t src_size,
                            tuningtiling::TuningTilingDefPtr& tiling)
{
    RTKB_CHECK(custom_ == nullptr, CANNKB_LOGE("custom_ is nullptr."), return FAILED);
    rwlock_.rdlock();
    const auto& itKey = tuning_cache_.find(k);
    if (itKey != tuning_cache_.cend()) {
        CANNKB_LOGD("Found in tuning cache repo. K: %u", k);
        tiling = itKey->second;
        tuning_cache_.erase(k);
        rwlock_.unlock();
        return SUCCESS;
    }
    const auto& itPid = tuning_cache_.find(pid);
    if (itPid != tuning_cache_.cend()) {
        CANNKB_LOGD("Found in tuning cache repo. pid: %u", pid);
        tiling = itPid->second;
        tuning_cache_.erase(pid);
        rwlock_.unlock();
        return SUCCESS;
    }
    auto iter = custom_->find(k);
    do {
        if (iter != custom_->cend() && QuickCmp(src, src_size, iter->second.ori_ky, iter->second.bank_key_size)) {
            CANNKB_LOGD("Found in custom repo. K: %u", iter->second.ky);
            break;
        }
        if (builtin_ == nullptr) {
            break;
        }
        iter = builtin_->find(k);
        if (iter != builtin_->cend() && QuickCmp(src, src_size, iter->second.ori_ky, iter->second.bank_key_size)) {
            CANNKB_LOGD("Found in builtin repo, K: %u.", iter->second.ky);
            break;
        }
        CANNKB_LOGD("Not found in custom or built-in repo. K: %u", k);
        rwlock_.unlock();
        return KEY_NOT_EXIST;
    } while (0);
    tiling = iter->second.tiling;
    rwlock_.unlock();
    return SUCCESS;
}

Status OpRuntimeBank::Update(uint32_t k, const std::shared_ptr<void>& ori_ky, size_t ori_ky_size,
                             const std::string& optype, const tuningtiling::TuningTilingDefPtr& tiling)
{
    RTKB_CHECK(tiling == nullptr, CANNKB_LOGE("Update repo failed. op: %s", optype.c_str()), return FAILED);
    RTKB_CHECK(custom_ == nullptr, CANNKB_LOGE("Update repo failed. op: %s", optype.c_str()), return FAILED);
    Record record;
    record.ky = k;
    record.optype = optype;
    record.ori_ky = ori_ky;
    record.bank_key_size = ori_ky_size;
    record.tiling = tiling;
    rwlock_.wrlock();
    custom_->erase(k);
    custom_->emplace(k, record);
    rwlock_.unlock();
    CANNKB_LOGD("Update op: %s hash: %u, tiling succeeded.", optype.c_str(), k);
    return SUCCESS;
}

void OpRuntimeBank::SetTuningTiling(uint32_t ky, const tuningtiling::TuningTilingDefPtr& tiling)
{
    rwlock_.wrlock();
    tuning_cache_.clear();
    tuning_cache_[ky] = tiling;
    rwlock_.unlock();
    return;
}

CustomRepoGuard::CustomRepoGuard(OpRuntimeBank& bank, bool auto_create) { Lock(bank.custom_path_, auto_create); }

CustomRepoGuard::~CustomRepoGuard() { UnLock(); }

void CustomRepoGuard::Lock(const std::string& custom_path, bool auto_create)
{
    if (custom_path.empty() || custom_path.size() >= PATH_MAX) {
        CANNKB_LOGE("Custom path %s is invalid.", custom_path.c_str());
        return;
    }
    char custom_real_path[PATH_MAX] = {0x00};
    if (realpath(custom_path.c_str(), custom_real_path) == nullptr) {
        CANNKB_LOGD("Will create lock file: %s", custom_path.c_str());
    }
    if (auto_create) {
        handle_lock_ = open(custom_real_path, O_RDWR | O_CREAT, kSafePermission);
    } else {
        handle_lock_ = open(custom_real_path, O_RDWR);
    }

    if (handle_lock_ < 0) {
        CANNKB_LOGW("No repo lib exist, no need to lock it.");
        return;
    }
    flock(handle_lock_, LOCK_EX);
    return;
}

void CustomRepoGuard::UnLock() const
{
    if (handle_lock_ < 0) {
        return;
    }
    flock(handle_lock_, LOCK_UN);
    (void)close(handle_lock_);
    return;
}
} // namespace RuntimeKb
