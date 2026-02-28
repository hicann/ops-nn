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
 * \file avg_pool_v2_grad_simt_tiling.h
 * \brief simt imply foravg_pool_v2_grad
 */

#ifndef CANN_AVG_POOL_V2_GRAD_SIMT_TILING_H
#define CANN_AVG_POOL_V2_GRAD_SIMT_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "avg_pool_v2_grad_tiling_common.h"
#include "avg_pool_v2_grad_tiling_base.h"
#include "../op_kernel/arch35/avg_pool_v2_grad_tiling_data.h"
#include "../op_kernel/arch35/avg_pool_v2_grad_tiling_key.h"

namespace optiling {

constexpr int64_t PAD_HL_DIM = 0;
constexpr int64_t PAD_HR_DIM = 1;
constexpr int64_t PAD_WL_DIM = 2;
constexpr int64_t PAD_WR_DIM = 3;

constexpr int64_t MAX_THREAD_NUM = 1024;

constexpr int64_t DCACHE_SIZE = 128 * 1024;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;

class AvgPoolV2GradTilingSIMT : public Ops::NN::Optiling::TilingBaseClass {
public:
    explicit AvgPoolV2GradTilingSIMT(gert::TilingContext* context) : TilingBaseClass(context)
    {}

    ~AvgPoolV2GradTilingSIMT() override
    {}

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;

    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

public:
    AvgPoolV2GradInputInfo inputData;
    uint64_t coreNum;
    uint64_t ubSize;
};

} // namespace optiling
#endif // CANN_AVG_POOL_V2_GRAD_SIMT_TILING_H