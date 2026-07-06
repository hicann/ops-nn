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
 * \file ascend_anti_quant_tiling_arch35.h
 * \brief External declarations for AscendAntiQuant pure-regbase tiling.
 *        CompileInfo is a locally-defined AscendAntiQuantCompileInfo
 *        (vectorCoreNum + ubSize); no dependency on atvoss.
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ASCEND_ANTI_QUANT_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ASCEND_ANTI_QUANT_TILING_H

#include "register/op_impl_registry.h"

namespace optiling {

extern ge::graphStatus Tiling4AscendAntiQuant(gert::TilingContext* context);
extern ge::graphStatus TilingPrepareForAscendAntiQuant(gert::TilingParseContext* context);

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ASCEND_ANTI_QUANT_TILING_H
