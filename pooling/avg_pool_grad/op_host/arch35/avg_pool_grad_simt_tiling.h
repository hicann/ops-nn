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
 * \file avg_pool_grad_simt_tiling.h
 * \brief simt imply foravg_pool_grad
 */

#ifndef CANN_AVG_POOL_GRAD_SIMT_TILING_H
#define CANN_AVG_POOL_GRAD_SIMT_TILING_H

#include "avg_pool_grad_tiling_base.h"
#include "../../../avg_pool_v2_grad/op_host/arch35/avg_pool_v2_grad_simt_tiling.h"

namespace optiling {

class AvgPoolGradTilingSIMT : public AvgPoolV2GradTilingSIMT {
public:
    explicit AvgPoolGradTilingSIMT(gert::TilingContext* context) : AvgPoolV2GradTilingSIMT(context)
    {}
    ~AvgPoolGradTilingSIMT() override
    {}

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
};

} // namespace optiling
#endif // CANN_AVG_POOL_GRAD_SIMT_TILING_H
