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
 * \file dynamic_quant_update_scatter_compile_info.h
 * \brief
 */
#ifndef __DYNAMIC_QUANT_UPDATE_SCATTER_COMPILE_INFO_H__
#define __DYNAMIC_QUANT_UPDATE_SCATTER_COMPILE_INFO_H__

#include "register/tilingdata_base.h"

namespace optiling {

struct DynamicQuantUpdateScatterCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};
} // namespace optiling
#endif // DYNAMIC_QUANT_UPDATE_SCATTER
