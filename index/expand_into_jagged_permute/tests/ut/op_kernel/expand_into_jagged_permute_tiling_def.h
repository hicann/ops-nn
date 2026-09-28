/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXPAND_INTO_JAGGED_PERMUTE_UT_TILING_DEF_H_
#define EXPAND_INTO_JAGGED_PERMUTE_UT_TILING_DEF_H_

#include "test_expand_into_jagged_permute_tiling_def.h"

// AddOpTestCase force-includes <op_name>_tiling_def.h when present. Keep the
// original UT data layout while providing the production type name expected by
// the Ascend910 kernel source.
using ExpandIntoJaggedPermuteTilingData = ExpandIntoJaggedPermuteTilingDataDef;

#endif // EXPAND_INTO_JAGGED_PERMUTE_UT_TILING_DEF_H_
