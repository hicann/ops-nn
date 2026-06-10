/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "../op_kernel/relu_v3_tiling_data.h"
#include "../op_kernel/relu_v3_tiling_key.h"

namespace optiling
{
    struct ReluV3CompileInfo {};

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto &x = context->GetInputShape(0)->GetStorageShape();
        OP_CHECK_IF(x.GetDimNum() > 0 && x[x.GetDimNum() - 1] % 8 != 0, OP_LOGE(context->GetNodeName(), "the last dim of `x` must be multiple of 8."), return ge::GRAPH_FAILED);
        long size = 1;
        for (auto i = 0; i < x.GetDimNum(); i++)
            size *= x[i];
        OP_CHECK_IF(size > std::numeric_limits<int>::max() / 512 * 512, OP_LOGE(context->GetNodeName(), "shape size is too big."), return ge::GRAPH_FAILED);

        auto &tiling = *context->GetTilingData<ReluV3TilingData>();
        tiling.size = size;

        auto platform_info = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platform_info);
        auto plaform = platform_ascendc::PlatformAscendC(platform_info);
        auto dtype = context->GetInputTensor(0)->GetDataType();
        auto data_size = ge::GetSizeInBytes(size, dtype);
        auto block_dim = plaform.GetCoreNumAiv();
        if (data_size < 1 << 15)
        {
            block_dim = 1;
        }
        else if (data_size < 1 << 22)
        {
            block_dim /= 2;
        }
        context->SetBlockDim(block_dim);
        auto tiling_key = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tiling_key);

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingParse([[maybe_unused]] gert::TilingParseContext *context)
    {
        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP_OPTILING(ReluV3).Tiling(TilingFunc).TilingParse<ReluV3CompileInfo>(TilingParse);
}
