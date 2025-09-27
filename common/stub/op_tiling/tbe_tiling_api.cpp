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
 * \file tbe_tiling_api.cpp
 * \brief
 */
#include "tbe_tiling_api.h"
#include <log/log.h>

namespace optiling {

bool GetTbeTiling(gert::TilingContext* context, [[maybe_unused]] Conv3dBackpropV2TBETilingData& tbeTilingForV2, [[maybe_unused]] const optiling::OpTypeV2 opType)
{
    OP_LOGW(context, "GetTbeTiling stub function not implemented");
    return false;
}
}