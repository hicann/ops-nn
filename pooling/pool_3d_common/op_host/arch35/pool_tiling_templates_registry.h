/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tiling_templates_registry.h
 * \brief
 */
#ifndef POOL_TILING_TEMPLATES_REGISTRY
#define POOL_TILING_TEMPLATES_REGISTRY

#include <map>
#include <string>
#include <memory>
#include <exe_graph/runtime/tiling_context.h>
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "error_util.h"

namespace optiling 
{
using Ops::NN::Optiling::TilingBaseClass;
template <typename T> std::unique_ptr<TilingBaseClass> TILING_CLASS(gert::TilingContext *context)
{
    return std::unique_ptr<T>(new (std::nothrow) T(context));
}

using TilingClassCase = std::unique_ptr<TilingBaseClass> (*)(gert::TilingContext *);

class TilingCases {
public:
    explicit TilingCases(std::string op_type) : op_type_(std::move(op_type))
    {
    }

    template <typename T> void AddTiling(int32_t priority)
    {
        OPS_ERR_IF(cases_.find(priority) != cases_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(op_type_, "There are duplicate registrations."), return);
        cases_[priority] = TILING_CLASS<T>;
        OPS_ERR_IF(
            cases_[priority] == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(op_type_, "PoolRegister op tiling func failed, please check the class name."),
            return);
    }

    const std::map<int32_t, TilingClassCase> &GetTilingCases()
    {
        return cases_;
    }

private:
    std::map<int32_t, TilingClassCase> cases_;
    const std::string op_type_;
};

// --------------------------------Interfacce with soc version --------------------------------
class PoolTilingRegistryNew {
public:
    PoolTilingRegistryNew() = default;

#ifdef ASCENDC_OP_TEST
    static PoolTilingRegistryNew &GetInstance();
#else
    static PoolTilingRegistryNew &GetInstance()
    {
        static PoolTilingRegistryNew registry_impl_;
        return registry_impl_;
    }
#endif

    std::shared_ptr<TilingCases> RegisterOp(const std::string &op_type,
                                            int32_t soc_version)
    {
        auto soc_iter = registry_map_.find(soc_version);
        if (soc_iter == registry_map_.end()) {
            std::map<std::string, std::shared_ptr<TilingCases>> op_type_map;
            op_type_map[op_type] = std::shared_ptr<TilingCases>(new (std::nothrow) TilingCases(op_type));
            registry_map_[soc_version] = op_type_map;
        } else {
            if (soc_iter->second.find(op_type) == soc_iter->second.end()) {
                soc_iter->second[op_type] =
                    std::shared_ptr<TilingCases>(new (std::nothrow) TilingCases(op_type));
            }
        }

        OPS_ERR_IF(registry_map_[soc_version][op_type] == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(op_type, "PoolRegister tiling func failed, please check the class name."),
                    return nullptr);
        return registry_map_[soc_version][op_type];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context)
    {
        int32_t soc_version = (int32_t)platform_ascendc::SocVersion::RESERVED_VERSION;
        const char *op_type = context->GetNodeType();
        fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compileInfoPtr = reinterpret_cast<const Ops::NN::Optiling::CompileInfoCommon *>(context->GetCompileInfo());
            OPS_ERR_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(op_type, "compileInfoPtr is null."),
                        return ge::GRAPH_FAILED);
            soc_version = compileInfoPtr->socVersion;
            OPS_LOG_D(context, "soc version in compileInfo is %d", soc_version);
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            soc_version = static_cast<int32_t>(ascendcPlatform.GetSocVersion());
            OPS_LOG_D(context, "soc version is %d", soc_version);
            if (soc_version == (int32_t)platform_ascendc::SocVersion::RESERVED_VERSION) {
                OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, cannot find soc version.");
                return ge::GRAPH_FAILED;
            }
        }
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type, soc_version);
        for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
            auto tilingTemplate = it->second(context);
            if (tilingTemplate != nullptr) {
                ge::graphStatus status = tilingTemplate->DoTiling();
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OPS_LOG_D(context, "Do general op tiling success priority=%d", it->first);
                    return status;
                }
                OPS_LOG_D(context, "Ignore general op tiling priority=%d", it->first);
            }
        }
        OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context, const std::vector<int32_t> &priorities)
    {
        int32_t soc_version;
        const char *op_type = context->GetNodeType();
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compileInfoPtr = reinterpret_cast<const Ops::NN::Optiling::CompileInfoCommon *>(context->GetCompileInfo());
            OPS_ERR_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(op_type, "compileInfoPtr is null."),
                        return ge::GRAPH_FAILED);
            soc_version = compileInfoPtr->socVersion;
            OPS_LOG_D(context, "soc version in compileInfo is %d", soc_version);
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            soc_version = static_cast<int32_t>(ascendcPlatform.GetSocVersion());
            OPS_LOG_D(context, "soc version is %d", soc_version);
        }

        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type, soc_version);
        for (auto priority_id : priorities) {
            auto tilingCaseIter = tilingTemplateRegistryMap.find(priority_id);
            if (tilingCaseIter != tilingTemplateRegistryMap.end()) {
                auto templateFunc = tilingCaseIter->second(context);
                if (templateFunc != nullptr) {
                    ge::graphStatus status = templateFunc->DoTiling();
                    if (status == ge::GRAPH_SUCCESS) {
                        OPS_LOG_D(context, "Do general op tiling success priority=%d", priority_id);
                        return status;
                    }
                    OPS_LOG_D(context, "Ignore general op tiling priority=%d", priority_id);
                }
            }
        }
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, TilingClassCase> &GetTilingTemplates(const std::string &op_type,
                                                                    int32_t soc_version)
    {
        auto soc_iter = registry_map_.find(soc_version);
        OPS_ERR_IF(soc_iter == registry_map_.end(),
                    OPS_REPORT_VECTOR_INNER_ERR(op_type, "Get op tiling func failed, please check the soc version %d",
                                                soc_version),
                    return empty_tiling_case_);
        auto op_iter = soc_iter->second.find(op_type);
        OPS_ERR_IF(op_iter == soc_iter->second.end(),
                    OPS_REPORT_VECTOR_INNER_ERR(op_type, "Get op tiling func failed, please check the op name."),
                    return empty_tiling_case_);
        return op_iter->second->GetTilingCases();
    }

private:
    std::map<int32_t, std::map<std::string, std::shared_ptr<TilingCases>>> registry_map_; // key is socversion
    const std::map<int32_t, TilingClassCase> empty_tiling_case_{};
};

class PoolRegisterNew {
public:
    explicit PoolRegisterNew(std::string op_type) : op_type_(std::move(op_type))
    {
    }

    template <typename T> PoolRegisterNew &tiling(int32_t priority, int32_t soc_version)
    {
        auto tilingCases = PoolTilingRegistryNew::GetInstance().RegisterOp(op_type_, soc_version);
        OPS_ERR_IF(tilingCases == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(op_type_, "PoolRegister op tiling failed, please the op name."),
                    return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

    template <typename T> PoolRegisterNew &tiling(int32_t priority, const std::vector<int32_t>& soc_versions)
    {
        for (int32_t soc_version : soc_versions) {
            auto tilingCases = PoolTilingRegistryNew::GetInstance().RegisterOp(op_type_, soc_version);
            OPS_ERR_IF(tilingCases == nullptr,
                        OPS_REPORT_VECTOR_INNER_ERR(op_type_, "PoolRegister op tiling failed, please the op name."),
                        return *this);
            tilingCases->AddTiling<T>(priority);
        }
        return *this;
    }

private:
    const std::string op_type_;
};

// --------------------------------Interfacce without soc version --------------------------------
class PoolTilingRegistry {
public:
    PoolTilingRegistry() = default;

#ifdef ASCENDC_OP_TEST
    static PoolTilingRegistry &GetInstance();
#else
    static PoolTilingRegistry &GetInstance()
    {
        static PoolTilingRegistry registry_impl_;
        return registry_impl_;
    }
#endif

    std::shared_ptr<TilingCases> RegisterOp(const std::string &op_type)
    {
        if (registry_map_.find(op_type) == registry_map_.end()) {
            registry_map_[op_type] = std::shared_ptr<TilingCases>(new (std::nothrow) TilingCases(op_type));
        }
        OPS_ERR_IF(registry_map_[op_type] == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(op_type, "PoolRegister tiling func failed, please check the class name."),
                   return nullptr);
        return registry_map_[op_type];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context)
    {
        const char *op_type = context->GetNodeType();
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
        for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
            auto tilingTemplate = it->second(context);
            if (tilingTemplate != nullptr) {
                ge::graphStatus status = tilingTemplate->DoTiling();
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OPS_LOG_D(context, "Do general op tiling success priority=%d", it->first);
                    return status;
                }
                OPS_LOG_D(context, "Ignore general op tiling priority=%d", it->first);
            }
        }
        OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context, const std::vector<int32_t> &priorities)
    {
        const char *op_type = context->GetNodeType();
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
        for (auto priorityId : priorities) {
            auto templateFunc = tilingTemplateRegistryMap[priorityId](context);
            if (templateFunc != nullptr) {
                ge::graphStatus status = templateFunc->DoTiling();
                if (status == ge::GRAPH_SUCCESS) {
                    OPS_LOG_D(context, "Do general op tiling success priority=%d", priorityId);
                    return status;
                }
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OPS_LOG_D(context, "Do op tiling failed");
                    return status;
                }
                OPS_LOG_D(context, "Ignore general op tiling priority=%d", priorityId);
            }
        }
        OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, TilingClassCase> &GetTilingTemplates(const std::string &op_type)
    {
        OPS_ERR_IF(registry_map_.find(op_type) == registry_map_.end(),
                   OPS_REPORT_VECTOR_INNER_ERR(op_type, "Get op tiling func failed, please check the op name."),
                   return empty_tiling_case_);
        return registry_map_[op_type]->GetTilingCases();
    }

private:
    std::map<std::string, std::shared_ptr<TilingCases>> registry_map_;
    const std::map<int32_t, TilingClassCase> empty_tiling_case_;
};

class PoolRegister {
public:
    explicit PoolRegister(std::string op_type) : op_type_(std::move(op_type))
    {
    }

    template <typename T> PoolRegister &tiling(int32_t priority)
    {
        auto tilingCases = PoolTilingRegistry::GetInstance().RegisterOp(op_type_);
        OPS_ERR_IF(tilingCases == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(op_type_, "PoolRegister op tiling failed, please the op name."),
                   return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

private:
    const std::string op_type_;
};

// op_type: 算子名称， class_name: 注册的 tiling 类, soc_version：芯片版本号
// priority: tiling 类的优先级, 越小表示优先级越高, 即会优先选择这个tiling类
#define REGISTER_POOL_TILING_TEMPLATE_WITH_SOCVERSION(op_type, class_name, soc_versions, priority) \
    static PoolRegisterNew VAR_UNUSED##op_type##class_name##priority_register = \
        PoolRegisterNew(#op_type).tiling<class_name>(priority, soc_versions)

// op_type: 算子名称， class_name: 注册的 tiling 类,
// priority: tiling 类的优先级, 越小表示优先级越高, 即被选中的概率越大
#define REGISTER_POOL_TILING_TEMPLATE(op_type, class_name, priority)                                                        \
    static PoolRegister VAR_UNUSED##op_type_##class_name##priority_register = PoolRegister(op_type).tiling<class_name>(priority)

// op_type: 算子名称， class_name: 注册的 tiling 类,
// soc_version: soc版本，用于区分不同的soc
// priority: tiling 类的优先级, 越小表示优先级越高, 即会优先选择这个tiling类
#define REGISTER_POOL_TILING_TEMPLATE_NEW(op_type, class_name, soc_version, priority)            \
    static PoolRegisterNew VAR_UNUSED##op_type##class_name##priority_register =        \
        PoolRegisterNew(#op_type).tiling<class_name>(priority, soc_version)

// op_type: 算子名称， class_name: 注册的 tiling 类,
// priority: tiling 类的优先级, 越小表示优先级越高, 即被选中的概率越大
// 取代 REGISTER_POOL_TILING_TEMPLATE , 传入的op_type如果是字符串常量，需要去掉引号
#define REGISTER_OPS_POOL_TILING_TEMPLATE(op_type, class_name, priority) \
    static PoolRegister __attribute__((unused)) tiling_##op_type##_##class_name##_##priority##_register = \
    PoolRegister(#op_type).tiling<class_name>(priority)

} // namespace optiling

#endif  // CONV_TILING_TEMPLATES_REGISTRY