/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file lp_loss_tiling_arch35.h
 * \brief
 */
#ifndef OPS_OP_TILING_LP_LOSS_TILING_H
#define OPS_OP_TILING_LP_LOSS_TILING_H

#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/lp_loss_tiling_struct.h"
#include "atvoss/reduce/reduce_tiling.h"
namespace optiling {

struct LpLossTilingKey
{
    Ops::Base::ReduceTilingKey ReduceTiling;
    uint32_t Reduction = 2;
    uint32_t Dtype = 20;
};
} // namespace optiling
#endif // OPS_OP_TILING_LP_LOSS_TILING_H