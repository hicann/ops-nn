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
 * \file adaptive_avg_pool3d_simt_tiling.h
 * \brief simt imply for adaptive_avg_pool3d_simt_tiling
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPATIVE_AVG_POOL3D_SIMT_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPATIVE_AVG_POOL3D_SIMT_TILING_H

#include "../op_kernel/arch35/adaptive_pool3d_tiling_struct.h"
#include "../op_host/arch35/adaptive_pool3d_tiling.h"
namespace optiling
{
class AdaptiveAvgPool3DTilingSimt : public AdaptivePool3dBaseTiling
{
public:
    explicit AdaptiveAvgPool3DTilingSimt(gert::TilingContext* context) : AdaptivePool3dBaseTiling(context)
    {}
    ~AdaptiveAvgPool3DTilingSimt() override
    {}
    
protected:
    void SetTilingData();
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    ge::graphStatus GetDataFormatAttrInfo();

    AdaptivePool3DTiling::AdaptivePool3DSimtTilingData* tilingData_ = 
        context_->GetTilingData<AdaptivePool3DTiling::AdaptivePool3DSimtTilingData>();
    int64_t maxDivUseNum_ = 0;
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPATIVE_AVG_POOL3D_SIMT_TILING_H