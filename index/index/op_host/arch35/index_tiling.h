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
 * \file index_tiling.h
 * \brief Common compile info and base class for Index tiling
 */
#ifndef OPS_NN_PRIVATE_INDEX_INDEX_OP_HOST_ARCH35_INDEX_TILING_H_
#define OPS_NN_PRIVATE_INDEX_INDEX_OP_HOST_ARCH35_INDEX_TILING_H_

#include <cstddef>
#include <cstdint>
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct IndexCompileInfo {
    int32_t task_align;
    int32_t core_num;
    int32_t available_size;
    int32_t indices_num;
    int32_t align_num;
    int32_t after_v200;
    uint64_t ubSize;
};

class IndexTilingCommon : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit IndexTilingCommon(gert::TilingContext* context) : TilingBaseClass(context)
    {}

protected:
    ge::graphStatus GetPlatformInfo() override
    {
        auto platformPtr = context_->GetPlatformInfo();
        if (platformPtr == nullptr) {
            auto compileInfoPtr = static_cast<const IndexCompileInfo*>(context_->GetCompileInfo());
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

    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        size_t* workspace = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
        workspace[0] = ASCENDC_TOOLS_WORKSPACE;
        return ge::GRAPH_SUCCESS;
    }

    uint64_t GenXDtype(uint64_t factor = 1) const
    {
        auto firstInput = context_->GetInputDesc(0);
        if (firstInput == nullptr) {
            OP_LOGE("IndexTilingCommon", "input desc is null");
            return 0;
        }
        auto paramsDtype = firstInput->GetDataType();
        uint64_t xDType = 1;
        if (paramsDtype == ge::DT_INT64 || paramsDtype == ge::DT_COMPLEX64) {
            xDType = DTYPE_SIZE_B64;
        } else if (paramsDtype == ge::DT_INT32 || paramsDtype == ge::DT_FLOAT) {
            xDType = DTYPE_SIZE_B32;
        } else if (paramsDtype == ge::DT_FLOAT16 || paramsDtype == ge::DT_BF16) {
            xDType = DTYPE_SIZE_B16;
        } else if (paramsDtype == ge::DT_INT8 || paramsDtype == ge::DT_BOOL || paramsDtype == ge::DT_UINT8) {
            xDType = DTYPE_SIZE_B8;
        } else {
            OP_LOGE("IndexTilingCommon", "input x dtype error!");
            return 0;
        }
        return xDType * factor;
    }

protected:
    static constexpr uint64_t DTYPE_SIZE_B8 = 1;
    static constexpr uint64_t DTYPE_SIZE_B16 = 2;
    static constexpr uint64_t DTYPE_SIZE_B32 = 4;
    static constexpr uint64_t DTYPE_SIZE_B64 = 8;
    static constexpr uint64_t DTYPE_SIZE_B128 = 16;
    static constexpr size_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;
    uint64_t coreNum_{0};
    uint64_t ubSize_{0};
};

} // namespace optiling
#endif // OPS_NN_PRIVATE_INDEX_INDEX_OP_HOST_ARCH35_INDEX_TILING_H_
