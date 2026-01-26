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
 * \file apply_adam_d_tiling.h
 * \brief apply_adam_d_tiling head file
 */
#ifndef RUNTIME_V2_OP_IMPL_APPLY_ADAM_D_TILING_H_
#define RUNTIME_V2_OP_IMPL_APPLY_ADAM_D_TILING_H_

#include "register/tilingdata_base.h"
#include "atvoss/elewise/elewise_tiling.h"

namespace optiling {
    struct ApplyAdamDCompileInfo {
        uint64_t coreNum;
        uint64_t ubSize;
    };
}

#endif // RUNTIME_V2_OP_IMPL_APPLY_ADAM_D_TILING_H_