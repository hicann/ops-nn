/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DEQUANTIZE_TILING_ARCH35_H_
#define DEQUANTIZE_TILING_ARCH35_H_

#include <cstdint>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <exe_graph/runtime/tiling_context.h>
#include "../../op_kernel/arch35/dequantize_tiling_struct.h"
#include "../../op_kernel/arch35/dequantize_struct.h"

namespace dequantize_ns {

bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& padded_in,
                         const std::vector<std::vector<int64_t>>& padded_out, int64_t max_rank);

bool PadAndSqueeze(const std::vector<std::vector<int64_t>>& input_shapes,
                   const std::vector<std::vector<int64_t>>& output_shapes, std::vector<int64_t>& maximum_bro_shape,
                   std::vector<std::vector<int64_t>>& normal_input_shapes,
                   std::vector<std::vector<int64_t>>& normal_output_shapes);

bool FindSplitAxis(const std::vector<int64_t>& max_bro_shape, int64_t dtype_size, int64_t ub_per_core,
                   int64_t phys_nodes, SplitResult& out);

bool MultiCoreSplit(const std::vector<int64_t>& max_bro_shape, const SplitResult& ub_split, int64_t max_cores,
                    MultiCoreResult& out);

bool PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides);

} // namespace dequantize_ns

namespace optiling {

struct DequantizeCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class DequantizeTiling {
public:
    explicit DequantizeTiling(gert::TilingContext* ctx);
    ge::graphStatus RunTiling();

private:
    ge::graphStatus GetShapeInfo();
    template <int64_t R>
    ge::graphStatus DoTilingAndSet();
    template <int64_t R>
    void FillInputTilingData(DequantizeTilingData<R>* tiling, int64_t num_in, int64_t delta,
                             const std::vector<std::vector<int64_t>>& in_strides);
    template <int64_t R>
    void FillOutputTilingData(DequantizeTilingData<R>* tiling, int64_t num_out, int64_t delta,
                              const std::vector<std::vector<int64_t>>& out_strides);
    ge::graphStatus ResolveDtype();
    void PrecomputeConstants();
    ge::graphStatus ReadTensorShapes(bool is_input, std::vector<std::vector<int64_t>>& raw_shapes);

    gert::TilingContext* ctx_;
    std::vector<std::vector<int64_t>> raw_input_shapes_;
    std::vector<std::vector<int64_t>> raw_output_shapes_;
    std::vector<int64_t> max_bro_shape_;
    std::vector<std::vector<int64_t>> normal_input_shapes_;
    std::vector<std::vector<int64_t>> normal_output_shapes_;
    int64_t dtype_size_ = 0;
    int64_t rank_ = 0;
    std::string mode_;
    int64_t dtype_x_ = 0;
    float bias_ = 0.0f;
    float inv_range_ = 0.0f;
};

} // namespace optiling

#endif // DEQUANTIZE_TILING_ARCH35_H_
