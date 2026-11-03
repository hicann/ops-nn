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
 * \file modulate_grad_tiling_arch35.h
 * \brief modulate_grad regbase tiling entry for arch35 (ascend950)
 */

#ifndef MODULATE_GRAD_TILING_ARCH35_H
#define MODULATE_GRAD_TILING_ARCH35_H

#include "register/op_impl_registry.h"

namespace optiling {
ge::graphStatus ModulateGradTilingArch35(gert::TilingContext* context, bool isScale, bool isShift);
} // namespace optiling

#endif
