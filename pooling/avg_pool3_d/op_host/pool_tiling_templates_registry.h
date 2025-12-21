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
 * \file pool_tiling_templates_registry.h
 * \brief
 */

#ifndef POOL_TILING_TEMPLATES_REGISTRY
#define POOL_TILING_TEMPLATES_REGISTRY

#include <map>
#include <string>
#include <memory>
#include "exe_graph/runtime/tiling_context.h"
#include "tiling_base.h"
#include "log/log.h"

namespace optiling {
template <typename T>
std::unique_ptr<TilingBaseClass> TILING_CLASS(gert::TilingContext* context)
{
    return std::make_unique<T>(T(context));
}

using TilingClassCase = std::unique_ptr<TilingBaseClass> (*)(gert::TilingContext*);

class PoolTilingCases
{
public:
    explicit PoolTilingCases(std::string op_type) : op_type_(std::move(op_type))
    {}

    template <typename T>
    void AddTiling(int32_t priority)
    {
        OP_CHECK_IF(
            cases_.find(priority) != cases_.end(), OP_LOGE(op_type_, "There are duplicate registrations."), return);
        cases_[priority] = TILING_CLASS<T>;
        OP_CHECK_IF(
            cases_[priority] == nullptr,
            OP_LOGE(op_type_, "PoolRegister op tiling func failed, please check the class name."), return);
    }

    const std::map<int32_t, TilingClassCase>& GetPoolTilingCases()
    {
        return cases_;
    }

private:
    std::map<int32_t, TilingClassCase> cases_;
    const std::string op_type_;
};

// --------------------------------Interfacce without soc version --------------------------------
class PoolTilingRegistry
{
public:
    PoolTilingRegistry() = default;

#ifdef ASCENDC_OP_TEST
    static PoolTilingRegistry& GetInstance();
#else
    static PoolTilingRegistry& GetInstance()
    {
        static PoolTilingRegistry registry_impl_;
        return registry_impl_;
    }
#endif

    std::shared_ptr<PoolTilingCases> RegisterOp(const std::string& op_type)
    {
        if (registry_map_.find(op_type) == registry_map_.end()) {
            registry_map_[op_type] = std::make_shared<PoolTilingCases>(PoolTilingCases(op_type));
        }
        OP_CHECK_IF(
            registry_map_[op_type] == nullptr,
            OP_LOGE(op_type, "PoolRegister tiling func failed, please check the class name."), return nullptr);
        return registry_map_[op_type];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext* context)
    {
        const char* op_type = context->GetNodeType();
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
        for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
            auto tilingTemplate = it->second(context);
            if (tilingTemplate != nullptr) {
                ge::graphStatus status = tilingTemplate->DoTiling();
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OP_LOGD(context, "Do general op tiling success priority=%d", it->first);
                    return status;
                }
                OP_LOGD(context, "Ignore general op tiling priority=%d", it->first);
            }
        }
        OP_LOGE(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext* context, const std::vector<int32_t>& priorities)
    {
        const char* op_type = context->GetNodeType();
        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type);
        for (auto priorityId : priorities) {
            auto templateFunc = tilingTemplateRegistryMap[priorityId](context);
            if (templateFunc != nullptr) {
                ge::graphStatus status = templateFunc->DoTiling();
                if (status == ge::GRAPH_SUCCESS) {
                    OP_LOGD(context, "Do general op tiling success priority=%d", priorityId);
                    return status;
                }
                if (status != ge::GRAPH_PARAM_INVALID) {
                    OP_LOGD(context, "Do op tiling failed");
                    return status;
                }
                OP_LOGD(context, "Ignore general op tiling priority=%d", priorityId);
            }
        }
        OP_LOGE(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, TilingClassCase>& GetTilingTemplates(const std::string& op_type)
    {
        OP_CHECK_IF(
            registry_map_.find(op_type) == registry_map_.end(),
            OP_LOGE(op_type, "Get op tiling func failed, please check the op name."), return empty_tiling_case_);
        return registry_map_[op_type]->GetPoolTilingCases();
    }

private:
    std::map<std::string, std::shared_ptr<PoolTilingCases>> registry_map_;
    const std::map<int32_t, TilingClassCase> empty_tiling_case_;
};

class PoolRegister
{
public:
    explicit PoolRegister(std::string op_type) : op_type_(std::move(op_type))
    {}

    template <typename T>
    PoolRegister& tiling(int32_t priority)
    {
        auto PoolTilingCases = PoolTilingRegistry::GetInstance().RegisterOp(op_type_);
        OP_CHECK_IF(
            PoolTilingCases == nullptr, OP_LOGE(op_type_, "PoolRegister op tiling failed, please the op name."), return *this);
        PoolTilingCases->AddTiling<T>(priority);
        return *this;
    }

private:
    const std::string op_type_;
};

// op_type: 算子名称， class_name: 注册的 tiling 类,
// priority: tiling 类的优先级, 越小表示优先级越高, 即被选中的概率越大
#define REGISTER_POOL_TILING_TEMPLATE(op_type, class_name, priority)                                                        \
    static PoolRegister VAR_UNUSED##op_type_##class_name##priority_register = \
    PoolRegister(op_type).tiling<class_name>(priority)

// op_type: 算子名称， class_name: 注册的 tiling 类,
// priority: tiling 类的优先级, 越小表示优先级越高, 即被选中的概率越大
// 取代 REGISTER_TILING_TEMPLATE , 传入的op_type如果是字符串常量，需要去掉引号
#define REGISTER_OPS_POOL_TILING_TEMPLATE(op_type, class_name, priority) \
    static PoolRegister __attribute__((unused)) tiling_##op_type##_##class_name##_##priority##_register = \
    PoolRegister(#op_type).tiling<class_name>(priority)
}
#endif  // CONV_TILING_TEMPLATES_REGISTRY