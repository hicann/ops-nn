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
 * \file cross_entropy_sum_exp_and_index_logit_tiling_arch35.h
 * \brief A5 (ascend950) tiling data struct
 */
#ifndef OPS_BUILT_IN_OP_TILING_ARCH35_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_H
#define OPS_BUILT_IN_OP_TILING_ARCH35_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_H

#include "register/tilingdata_base.h"
#include "loss/cross_entropy_sum_exp_and_index_logit/op_kernel/arch35/cross_entropy_sum_exp_and_index_logit_struct.h"

namespace optiling {

struct CrossEntropySumExpAndIndexLogitCompileInfo {
    int32_t totalCoreNum = 40;
    uint64_t ubSizePlatForm = 0;
};

} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_ARCH35_CROSS_ENTROPY_SUM_EXP_AND_INDEX_LOGIT_H
