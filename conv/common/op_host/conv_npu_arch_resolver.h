/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_NN_CONV_COMMON_OP_HOST_CONV_NPU_ARCH_RESOLVER_H
#define OPS_NN_CONV_COMMON_OP_HOST_CONV_NPU_ARCH_RESOLVER_H

#include <string>
#include "platform/platform_infos_def.h"

namespace conv_arch {

inline bool IsCubeVectorFuseSoc(fe::PlatFormInfos& platformInfo)
{
    std::string cubeVecState;
    platformInfo.GetPlatformRes("SoCInfo", "cube_vector_combine", cubeVecState);
    return cubeVecState == "fuse";
}

class NpuArchResolver {
public:
    virtual ~NpuArchResolver() = default;
    virtual const std::string& GetArchKey() const = 0;
    static const NpuArchResolver& GetInstance(fe::PlatFormInfos& platformInfo);
};

class FuseNpuArchResolver : public NpuArchResolver {
public:
    const std::string& GetArchKey() const override
    {
        static const std::string key = "FUSE";
        return key;
    }
};

class DefaultNpuArchResolver : public NpuArchResolver {
public:
    const std::string& GetArchKey() const override
    {
        static const std::string key = "3510";
        return key;
    }
};

inline const NpuArchResolver& NpuArchResolver::GetInstance(fe::PlatFormInfos& platformInfo)
{
    static const FuseNpuArchResolver fuseResolver;
    static const DefaultNpuArchResolver defaultResolver;
    return IsCubeVectorFuseSoc(platformInfo) ? static_cast<const NpuArchResolver&>(fuseResolver) :
                                               static_cast<const NpuArchResolver&>(defaultResolver);
}

inline const std::string& GetNpuArchKey(fe::PlatFormInfos& platformInfo)
{
    return NpuArchResolver::GetInstance(platformInfo).GetArchKey();
}

} // namespace conv_arch

#endif // OPS_NN_CONV_COMMON_OP_HOST_CONV_NPU_ARCH_RESOLVER_H
