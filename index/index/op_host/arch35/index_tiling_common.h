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
 * \file index_tiling_common.h
 * \brief ac index tiling common
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_COMMON_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_COMMON_H_
#pragma once
#include "tiling_base/tiling_base.h"
#include "tiling/tiling_api.h"
#include "index_tiling.h"

namespace optiling {

class IndexTilingCommon : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit IndexTilingCommon(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetPlatformInfo() override
    {
        auto platformPtr = context_->GetPlatformInfo();
        if (platformPtr == nullptr) {
            auto compileInfoPtr = reinterpret_cast<const IndexCompileInfo*>(context_->GetCompileInfo());
            OP_CHECK_IF(
                compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
            coreNum_ = compileInfoPtr->core_num;
            ubSize_ = compileInfoPtr->ubSize;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
            coreNum_ = ascendcPlatform.GetCoreNumAiv();
            uint64_t ubSizePlatform;
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
            ubSize_ = ubSizePlatform;
        }
        OP_CHECK_IF(
            coreNum_ == static_cast<uint64_t>(0), OP_LOGE(context_->GetNodeName(), "coreNum is 0"),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

public:
    uint64_t coreNum_{0};
    uint64_t ubSize_{0};
};
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_INDEX_TILING_COMMON_H_
