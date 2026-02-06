/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pool3d_with_argmax_v2_simt_tiling.h
 * \brief simt imply for max_pool3d_with_argmax
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_WITH_ARGMAX_V2_SIMT_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAX_POOL3D_WITH_ARGMAX_V2_SIMT_TILING_H

#include "max_pool3d_with_argmax_v2_tiling_base.h"
#include "../op_kernel/arch35/max_pool3d_with_argmax_v2_tiling_struct.h"

namespace optiling
{
const int64_t NCDHW_DIMS = 5;   
const int64_t KERNEL_POS = 0;
const int64_t STRIDE_POS = 1;
const int64_t PADDING_POS = 2;
const int64_t DILATION_POS = 3;
const int64_t CEIL_POS = 4;
const int64_t FORMAT_POS = 5;
const int64_t N_DIM_ = 0;
const int64_t C_DIM_ = 1;
const int64_t D_DIM_ = 2;
const int64_t H_DIM_ = 3;
const int64_t W_DIM_ = 4;
const int64_t D_IDX_ = 0;
const int64_t H_IDX_ = 1;
const int64_t W_IDX_ = 2;
const int64_t DOUB = 2;
const int64_t FIRPOS = 0;
const int64_t SECPOS = 1;
constexpr int64_t MAX_INT32 = 2147483647;
constexpr uint64_t SIMT_NCDHW_TILING_KEY_INT32 = 600001;
constexpr uint64_t SIMT_NDHWC_TILING_KEY_INT32 = 600002;
constexpr uint64_t SIMT_NCDHW_TILING_KEY_INT64 = 600011;
constexpr uint64_t SIMT_NDHWC_TILING_KEY_INT64 = 600012;
constexpr int64_t MAX_THREAD_NUM = 256;
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

struct InputSIMTInfo {
    array<uint64_t, NCDHW_DIMS> inputShape;
    array<uint64_t, NCDHW_DIMS> outShape;
    array<uint64_t, DHW_DIMS> kernelSize;
    array<uint64_t, DHW_DIMS> stride;
    array<uint64_t, DHW_DIMS> pad;
    array<uint64_t, DHW_DIMS> dilation;
    bool ceilMode;
    std::string data_format;
};

class MaxPool3DWithArgmaxV2TilingSIMT : public MaxPool3DWithArgmaxV2BaseTiling
{
public:
    explicit MaxPool3DWithArgmaxV2TilingSIMT(gert::TilingContext* context) : MaxPool3DWithArgmaxV2BaseTiling(context)
    {
    }

    ~MaxPool3DWithArgmaxV2TilingSIMT() override
    {
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    // 获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 分配workspace
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;
    // tiling信息打屏
    void DumpTilingInfo() override;

private:
    uint64_t GenerateTilingKey(uint64_t innerKey);
    MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData* tilingData_ = 
        context_->GetTilingData<MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2SimtTilingData>();
    InputSIMTInfo inputData;
    int nDimPos = 0;
    int cDimPos = 1;
    int dDimPos = 2;
    int hDimPos = 3;
    int wDimPos = 4;
    int64_t outputDataCount = 0;
};

}  // namespace optiling
#endif  // MAX_POOL3D_WITH_ARGMAX_V2_SIMT_TILING_H