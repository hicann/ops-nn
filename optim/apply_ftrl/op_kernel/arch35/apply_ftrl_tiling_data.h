/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APPLY_FTRL_TILING_DATA_H
#define APPLY_FTRL_TILING_DATA_H

/*!
 * \file apply_ftrl_tiling_data.h
 * \brief
 */

#include "atvoss/elewise/elewise_base_struct.h"
namespace ApplyFtrlOpTiling {
    using namespace Ops::Base;
    struct ApplyFtrlRegbaseTilingData {
        EleBaseTilingDataV2 elewiseTiling;
    };
}
#endif // APPLY_FTRL_TILING_DATA_H
