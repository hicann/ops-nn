/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file binary_cross_entropy_struct.h
 * \brief binary_cross_entropy tiling struct
 */
#ifndef OPS_BINARY_CORSS_ENTROPY_TILING_STRUCT_H_
#define OPS_BINARY_CORSS_ENTROPY_TILING_STRUCT_H_

#include "atvoss/elewise/elewise_base_struct.h"
#include "atvoss/reduce/reduce_tiling_data.h"

using namespace Ops::Base;

namespace optiling
{
struct BinaryCrossEntropyTilingData {
    ReduceOpTilingData reduceTiling;
    EleBaseTilingData eleBaseTiling;
};
} // namespace optiling

#endif // OPS_BINARY_CORSS_ENTROPY_TILING_STRUCT_H_