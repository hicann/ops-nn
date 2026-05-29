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
 * \file fake_quant_with_min_max_args_gradient_struct.h
 * \brief tiling-key template arg declaration
 */
#ifndef FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_STRUCT_H_
#define FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

#ifndef FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_TPL_DEFINED
#define FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_TPL_DEFINED

#define FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_TPL_DEFAULT 0

#endif

namespace FakeQuantWithMinMaxArgsGradientOp {
ASCENDC_TPL_ARGS_DECL(
    FakeQuantWithMinMaxArgsGradient,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_TPL_DEFAULT));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_TPL_DEFAULT)));

} // namespace FakeQuantWithMinMaxArgsGradientOp

#endif // FAKE_QUANT_WITH_MIN_MAX_ARGS_GRADIENT_STRUCT_H_
