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
 * \file foreach_tiling_common.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_TILING_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_TILING_COMMON_H_

#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {

struct ForeachCompileInfo {
    uint64_t coreNum;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t sysWorkspaceSize;
};

struct ForeachSoloCompileInfo {
    int64_t numBlocks;
    uint64_t ubSize;
    platform_ascendc::SocVersion socVersion;
};

class ForeachBaseClass : public Ops::NN::Optiling::TilingBaseClass
{
public:
    explicit ForeachBaseClass(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoLibApiTiling() override;

    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

} // namespace optiling

#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_FOREACH_TILING_COMMON_H_
