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

enum class XDtype : int64_t { INT8 = 0, UINT8 = 1, INT32 = 2 };

struct PerBufferLayout {
    int64_t bytes = 0;
    int64_t elems = 0;
};

bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& in_shapes,
                         const std::vector<std::vector<int64_t>>& out_shapes, int64_t max_rank, const char* op_name);

void PadAndSqueeze(const std::vector<std::vector<int64_t>>& input_shapes,
                   const std::vector<std::vector<int64_t>>& output_shapes, std::vector<int64_t>& maximum_bro_shape,
                   std::vector<std::vector<int64_t>>& normal_input_shapes,
                   std::vector<std::vector<int64_t>>& normal_output_shapes);

bool CalcPerBufferLayout(int64_t ub_per_core, int64_t phys_nodes, int64_t dtype_size, PerBufferLayout& out);

bool FindSplitAxis(const std::vector<int64_t>& max_bro_shape, const PerBufferLayout& layout, SplitResult& out);

bool MultiCoreSplit(const std::vector<int64_t>& max_bro_shape, const SplitResult& ub_split, int64_t max_cores,
                    MultiCoreResult& out);

void PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides);

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
    ge::graphStatus ValidateNodeInfo();
    ge::graphStatus ValidateCompileInfo();
    ge::graphStatus ReadAndValidateShapes();
    ge::graphStatus ReadOneTensorShape(bool is_input, size_t index, std::vector<int64_t>& dims);
    ge::graphStatus ResolveDtypeAndMode();
    ge::graphStatus ResolveInputDtype();
    ge::graphStatus ValidateRangeDtypes();
    ge::graphStatus ValidateOutputDtype();
    ge::graphStatus ParseMode();
    ge::graphStatus BuildAndValidateBroadcast();
    void LogShapeInfo();

    int64_t GetModeNum() const;
    uint64_t GetTilingKey(int64_t rank) const;
    template <int64_t R>
    ge::graphStatus RunWithRank();
    template <int64_t R>
    ge::graphStatus DoTilingAndSet();
    template <int64_t R>
    bool PrepareSplit(DequantizeTilingData<R>* tiling, int64_t& per_buf_bytes);
    template <int64_t R>
    void FillShapeAndStrideTiling(DequantizeTilingData<R>* tiling);
    template <int64_t R>
    void FillScalarTiling(DequantizeTilingData<R>* tiling, int64_t per_buf_bytes);
    template <int64_t R>
    void LogTilingData(const DequantizeTilingData<R>* tiling) const;

    template <int64_t R>
    void FillInputTilingData(DequantizeTilingData<R>* tiling, int64_t num_in, int64_t delta,
                             const std::vector<std::vector<int64_t>>& in_strides);
    template <int64_t R>
    void FillOutputTilingData(DequantizeTilingData<R>* tiling, int64_t num_out, int64_t delta,
                              const std::vector<std::vector<int64_t>>& out_strides);

    void PrecomputeConstants();

    gert::TilingContext* ctx_;
    std::vector<std::vector<int64_t>> raw_input_shapes_;
    std::vector<std::vector<int64_t>> raw_output_shapes_;
    std::vector<int64_t> max_bro_shape_;
    std::vector<std::vector<int64_t>> normal_input_shapes_;
    std::vector<std::vector<int64_t>> normal_output_shapes_;
    int64_t rank_ = 0;
    int64_t x_numel_ = 0;
    std::string mode_;
    dequantize_ns::XDtype dtype_x_ = dequantize_ns::XDtype::INT8;
    float bias_ = 0.0f;
    float denom_ = 0.0f;
};

} // namespace optiling

#endif // DEQUANTIZE_TILING_ARCH35_H_
