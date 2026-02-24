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
 * \file adaptive_max_pool2d_simt_tiling.h
 * \brief simt imply for adaptive_max_pool2d
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_MAX_POOL2D_SIMT_TILING_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_MAX_POOL2D_SIMT_TILING_H

#include "adaptive_max_pool2d_tiling_base.h"

namespace optiling
{
const int64_t NCHW_DIMS = 4;   
const int64_t N_DIM_ = 0;
const int64_t C_DIM_ = 1;
const int64_t H_DIM_ = 2;
const int64_t W_DIM_ = 3;
const int64_t H_IDX_ = 0;
const int64_t W_IDX_ = 1;
const int64_t DOUB = 2;
const int64_t FIRPOS = 0;
const int64_t SECPOS = 1;
constexpr int64_t MAX_INT32 = 2147483647;
constexpr int64_t MAX_THREAD_NUM = 256;

struct InputSIMTInfo {
    array<uint64_t, NCHW_DIMS> inputShape;
    array<uint64_t, NCHW_DIMS> outShape;
    int64_t kernelHMax;
    int64_t kernelWMax;
};

class AdaMaxPool2dTilingSIMT : public AdaMaxPool2dBaseTiling
{
public:
    explicit AdaMaxPool2dTilingSIMT(gert::TilingContext* context) : AdaMaxPool2dBaseTiling(context)
    {
    }

    ~AdaMaxPool2dTilingSIMT() override
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
    uint64_t GetTilingKey() const;
    // 分配workspace
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;
    // tiling信息打屏
    void DumpTilingInfo() override;

private:
    uint64_t GenerateTilingKey(uint64_t innerKey);
    ge::graphStatus CheckPlatformAndGetShapes();
    ge::graphStatus CheckDataTypeAndAttrs();
    AdaptiveMaxPool2dTilingData tiling;
    InputSIMTInfo inputData;
    uint64_t coreNum_ = 1;
    uint64_t ubSize_ = 0;
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_MAX_POOL2D_SIMT_TILING_H
