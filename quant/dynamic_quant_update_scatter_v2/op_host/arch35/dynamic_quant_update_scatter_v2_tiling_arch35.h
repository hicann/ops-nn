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
 * \file dynamic_quant_update_scatter_v2_tiling_arch35.h
 * \brief DynamicQuantUpdateScatterV2 regbase (arch35 / Ascend950) tiling entry declaration.
 *
 * CMake tiling routing compiles this registration for ascend950.
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_V2_TILING_ARCH35_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_V2_TILING_ARCH35_H

#include "register/op_impl_registry.h"
#include "quant/dynamic_quant_update_scatter_v2/op_kernel/arch35/dynamic_quant_update_scatter_v2_tiling_data.h"

namespace optiling {
ge::graphStatus Tiling4DynamicQuantUpdateScatterV2Regbase(gert::TilingContext* context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_QUANT_UPDATE_SCATTER_V2_TILING_ARCH35_H
