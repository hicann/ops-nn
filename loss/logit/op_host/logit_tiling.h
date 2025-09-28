/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file logit_tiling.h
 * \brief
 * 
 * 
 * 
 * 
 * 
 */

#ifndef LOGIT_TILING_DEF_H
#define LOGIT_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct LogitCompileInfo {
};

BEGIN_TILING_DATA_DEF(LogitTilingData)

TILING_DATA_FIELD_DEF(int64_t, elementNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint64_t, needCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Logit, LogitTilingData)
} // namespace optiling

#endif
