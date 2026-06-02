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
 * \file fake_quant_affine_cachemask_tiling_arch35.h
 * \brief arch35 tiling function declaration for FakeQuantAffineCachemask
 */
#ifndef FAKE_QUANT_AFFINE_CACHEMASK_TILING_ARCH35_H
#define FAKE_QUANT_AFFINE_CACHEMASK_TILING_ARCH35_H

#include "register/op_impl_registry.h"

namespace optiling {
ge::graphStatus Tiling4FakeQuantAffineCachemaskArch35(gert::TilingContext* context);
} // namespace optiling

#endif // FAKE_QUANT_AFFINE_CACHEMASK_TILING_ARCH35_H
