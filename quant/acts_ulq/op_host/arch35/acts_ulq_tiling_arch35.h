/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef ACTS_ULQ_TILING_ARCH35_H_
#define ACTS_ULQ_TILING_ARCH35_H_

#include "exe_graph/runtime/tiling_context.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../../op_kernel/arch35/acts_ulq_tiling_struct.h"
#include <vector>
#include <string>
#include <sstream>

namespace optiling {

struct ActsUlqCompileInfo {
    int64_t coreNum{0};
    uint64_t ubSize{0};
};

class ActsUlqTiling {
public:
    explicit ActsUlqTiling(gert::TilingContext* ctx);
    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetShapeInfo();
    template <int64_t R>
    ge::graphStatus DoTilingAndSet();

    gert::TilingContext* ctx_;
    std::vector<std::vector<int64_t>> raw_input_shapes_;
    std::vector<std::vector<int64_t>> raw_output_shapes_;
    std::vector<int64_t> max_bro_shape_;
    std::vector<std::vector<int64_t>> normal_input_shapes_;
    std::vector<std::vector<int64_t>> normal_output_shapes_;
    int64_t rank_{0};
};

} // namespace optiling

#endif
