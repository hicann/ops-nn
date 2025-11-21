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
 * \file conv_tiling_templates_registry.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_TILING_TEMPLATES_REGISTRY_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_TILING_TEMPLATES_REGISTRY_H
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
namespace conv_ops_tiling {
class ConvTilingRegistry {
public:
    ConvTilingRegistry() = default;

#ifdef ASCENDC_OP_TEST
    static ConvTilingRegistry &GetInstance();
#else
    static ConvTilingRegistry &GetInstance()
    {
        static ConvTilingRegistry registry_impl_;
        return registry_impl_;
    }
#endif

    std::shared_ptr<Ops::NN::Optiling::TilingCases> RegisterOp(const std::string &op_type,
                                                               int32_t soc_version)
    {
        auto soc_iter = registry_map_.find(soc_version);
        if (soc_iter == registry_map_.end()) {
            std::map<std::string, std::shared_ptr<Ops::NN::Optiling::TilingCases>> op_type_map;
            op_type_map[op_type] = std::make_shared<Ops::NN::Optiling::TilingCases>(op_type);
            registry_map_[soc_version] = op_type_map;
        } else {
            if (soc_iter->second.find(op_type) == soc_iter->second.end()) {
                soc_iter->second[op_type] = std::make_shared<Ops::NN::Optiling::TilingCases>(op_type);
            }
        }

        OPS_ERR_IF(registry_map_[soc_version][op_type] == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(op_type, "Register tiling func failed, please check the class name."),
                   return nullptr);
        return registry_map_[soc_version][op_type];
    }

    ge::graphStatus DoTilingImpl(gert::TilingContext *context)
    {
        int32_t soc_version = static_cast<int32_t>(platform_ascendc::SocVersion::RESERVED_VERSION);
        const char *op_type = context->GetNodeType();
        fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        soc_version = static_cast<int32_t>(ascendcPlatform.GetSocVersion());
        OPS_LOG_D(context, "soc version is %d", soc_version);
        if (soc_version == static_cast<int32_t>(platform_ascendc::SocVersion::RESERVED_VERSION)) {
            OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, cannot find soc version.");
            return ge::GRAPH_FAILED;
        }

        auto tilingTemplateRegistryMap = GetTilingTemplates(op_type, soc_version);
        for (auto it = tilingTemplateRegistryMap.begin(); it != tilingTemplateRegistryMap.end(); ++it) {
            auto tilingTemplate = it->second(context);
            if (tilingTemplate != nullptr) {
                ge::graphStatus status = tilingTemplate->DoTiling();
                if (status == ge::GRAPH_SUCCESS) {
                    OPS_LOG_D(context, "Do general op tiling success priority=%d", it->first);
                    return status;
                } else if (status != ge::GRAPH_PARAM_INVALID) {
                    OPS_LOG_E(context, "Do general op tiling not success priority=%d", it->first);
                    return status;
                }
                OPS_LOG_D(context, "Ignore general op tiling priority=%d", it->first);
            }
        }
        OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, no valid template is found.");
        return ge::GRAPH_FAILED;
    }

    const std::map<int32_t, Ops::NN::Optiling::TilingClassCase> &GetTilingTemplates(const std::string &op_type,
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
    std::map<int32_t, std::map<std::string, std::shared_ptr<Ops::NN::Optiling::TilingCases>>> registry_map_; // key is socversion
    const std::map<int32_t, Ops::NN::Optiling::TilingClassCase> empty_tiling_case_{};
};

class ConvRegisterNew {
public:
    explicit ConvRegisterNew(std::string op_type) : op_type_(std::move(op_type))
    {
    }

    template <typename T> ConvRegisterNew &tiling(int32_t priority, int32_t soc_version)
    {
        auto tilingCases = ConvTilingRegistry::GetInstance().RegisterOp(op_type_, soc_version);
        OPS_ERR_IF(tilingCases == nullptr,
                   OPS_REPORT_VECTOR_INNER_ERR(op_type_, "Register op tiling failed, please the op name."),
                   return *this);
        tilingCases->AddTiling<T>(priority);
        return *this;
    }

private:
    const std::string op_type_;
};

#define CONV_REGISTER_TILING_TEMPLATE(op_type, class_name, soc_version, priority)            \
    static ConvRegisterNew VAR_UNUSED##op_type##class_name##priority_register =        \
        ConvRegisterNew(#op_type).tiling<class_name>(priority, soc_version)

}
}
#endif