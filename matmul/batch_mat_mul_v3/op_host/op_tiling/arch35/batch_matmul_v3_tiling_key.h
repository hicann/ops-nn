/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_tiling_key.h
 * \brief
 */

#ifndef __OP_HOST_BATCH_MATMUL_V3_TILING_KEY_H__
#define __OP_HOST_BATCH_MATMUL_V3_TILING_KEY_H__

#include <sstream>
#include "tiling_base/tiling_key.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_key.h"

namespace optiling {
namespace matmul_v3_advanced {
class BatchMatMulV3TilingKey : public MatMulV3TilingKey {
public:
    uint64_t GetTilingKey() const override;
};
} // namespace matmul_v3_advanced
} // namespace optiling

#endif // __OP_HOST_BATCH_MATMUL_V3_TILING_KEY_H__
