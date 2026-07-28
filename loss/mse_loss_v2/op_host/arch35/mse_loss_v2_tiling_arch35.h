/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mse_loss_v2_tiling_arch35.h
 * \brief MSELossV2 arch35 (Ascend950) tiling compile-info declaration
 */

#ifndef MSE_LOSS_V2_TILING_ARCH35_H
#define MSE_LOSS_V2_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

namespace optiling {

struct MSELossV2CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // MSE_LOSS_V2_TILING_ARCH35_H
