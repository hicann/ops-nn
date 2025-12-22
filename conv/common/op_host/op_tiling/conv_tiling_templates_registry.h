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
#include "tiling_base/static_register_symbol.h"
#include "platform/platform_infos_def.h"
#include "arch35/conv_base.h"
#include "arch35/conv_base_utils.h"

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
        if (platformInfoPtr == nullptr) {
            if (strcmp(op_type, "QuantConv2D") == 0) {
                auto compileInfoPtr =
                    static_cast<const Conv2DTilingParseInfo *>(context->GetCompileInfo());
                OPS_ERR_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(op_type, "compileInfoPtr is null."),
                        return ge::GRAPH_FAILED);
                soc_version = static_cast<int32_t>(conv_ops_tiling::socConvertMap.at(compileInfoPtr->socVersion));            
            } else {
                auto compileInfoPtr =
                    static_cast<const conv_ops_tiling::ConvTilingParseInfo *>(context->GetCompileInfo());
                OPS_ERR_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(op_type, "compileInfoPtr is null."),
                        return ge::GRAPH_FAILED);
                soc_version = static_cast<int32_t>(conv_ops_tiling::socConvertMap.at(compileInfoPtr->socVersion));                
            }
            OPS_LOG_D(context, "soc version in compileInfo is %d", soc_version);
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            soc_version = static_cast<int32_t>(ascendcPlatform.GetSocVersion());
            OPS_LOG_D(context, "soc version is %d", soc_version);
            if (soc_version == static_cast<int32_t>(platform_ascendc::SocVersion::RESERVED_VERSION)) {
                OPS_REPORT_VECTOR_INNER_ERR(op_type, "Do op tiling failed, cannot find soc version.");
                return ge::GRAPH_FAILED;
            }
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

#define CONV_REGISTER_TILING_TEMPLATE(op_type, class_name, soc_version, priority)        \
    GLOBAL_REGISTER_SYMBOL(op_type, class_name, priority, __COUNTER__, __LINE__);                \
    static ConvRegisterNew VAR_UNUSED##op_type##class_name##priority##_register =        \
        ConvRegisterNew(#op_type).tiling<class_name>(priority, soc_version)
        
inline ge::graphStatus ConvTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "context is null"),
                    return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeType(), "Begin process ConvTilingFunc");
    return ConvTilingRegistry::GetInstance().DoTilingImpl(context);
}

inline ge::graphStatus TilingPrepareForConv(gert::TilingParseContext *context) {
    OP_TILING_CHECK(context == nullptr,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "context is null"),
    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
    return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<ConvTilingParseInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersion);
    compileInfoPtr->aicoreNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemBw(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Rate);

    OP_LOGD(context->GetNodeName(),
            "compileInfoPtr->socVersion %s"
            " l1Size:%lu, l2Size:%lu, coreNum:%u"
            "%lu, %lu, %lu, %lu",
            compileInfoPtr->socVersion.c_str(),
            compileInfoPtr->l1Size,
            compileInfoPtr->l2Size,
            compileInfoPtr->aicoreNum,
            compileInfoPtr->l0aSize,
            compileInfoPtr->l0bSize,
            compileInfoPtr->l0cSize,
            compileInfoPtr->l2Rate);
    return ge::GRAPH_SUCCESS;
}
}
}
#endif