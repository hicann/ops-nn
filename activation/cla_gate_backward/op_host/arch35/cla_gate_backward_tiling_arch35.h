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
 * \file cla_gate_backward_tiling_arch35.h
 * \brief
 */

#ifndef CLA_GATE_BACKWARD_TILING_ARCH35_H
#define CLA_GATE_BACKWARD_TILING_ARCH35_H

#include <cstdint>
#include <set>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "activation/cla_gate_backward/op_kernel/arch35/cla_gate_backward_tiling_data.h"

namespace optiling {

struct ClaGateBackwardCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct ClaGateBackwardTilingParam {
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t ubSize{0};
    int64_t headNum{0};
    int64_t headDim{0};
    int64_t totalHeads{0};
    int64_t baseCoreHeads{0};
    int64_t extraCoreCount{0};
    int64_t coreHeadsMax{0};
    int64_t batch{0};
    int64_t headCoreLoopCount{0};
    int64_t headCoreHeadsPerLoop{0};
    int64_t tailCoreLoopCount{0};
    int64_t tailCoreHeadsPerLoop{0};
    int64_t reduceTmpSize{0};
};

class ClaGateBackwardTiling {
public:
    explicit ClaGateBackwardTiling(gert::TilingContext* context) : context_(context) {}
    ge::graphStatus Run();

private:
    ge::graphStatus GetPlatformInfo();

    ge::graphStatus CheckDtype();
    ge::graphStatus CheckLayout();
    ge::graphStatus ValidateShapeAndGetHeads();

    ge::graphStatus SplitInterCore();
    ge::graphStatus SolveBatch();

    void FillTilingData();
    ge::graphStatus FillAndSetBlockDim();

    static int64_t AlignUp(int64_t value, int64_t align);
    static int64_t MaxI(int64_t a, int64_t b);
    static int64_t CeilDiv(int64_t a, int64_t b);
    static int64_t ReduceTmpBytes(int64_t rows, int64_t headDim);
    static int64_t ReduceTmpAllocBytes(int64_t rows, int64_t headDim);

private:
    static constexpr int64_t INPUT_GRAD_MERGED = 0;
    static constexpr int64_t INPUT_GLOBAL_ATTN = 1;
    static constexpr int64_t INPUT_LOCAL_ATTN = 2;
    static constexpr int64_t INPUT_GLOBAL_GATE_LOGITS = 3;
    static constexpr int64_t INPUT_LOCAL_GATE_LOGITS = 4;
    static constexpr int64_t ATTR_INPUT_ATTN_LAYOUT = 0;
    static constexpr const char* LAYOUT_TND = "TND";

    static constexpr int64_t MIN_COPY_BYTES = 4 * 1024;
    static constexpr int64_t MIN_REDUCE_TMP_BYTES = 32;
    static constexpr int64_t VEC_LANES_FP32 = 64;
    static constexpr int64_t SYSTEM_WORKSPACE = 0;

    static constexpr int64_t DTYPE_BYTES = 2;
    static constexpr int64_t HEAD_DIM_128 = 128;
    static constexpr int64_t HEAD_DIM_256 = 256;
    static constexpr int64_t HEAD_NUM_MAX = 128;
    static constexpr size_t TND_DIM_NUM = 3;
    static constexpr size_t GATE_LOGITS_DIM_NUM = 2;
    static constexpr size_t DIM_T = 0;
    static constexpr size_t DIM_N = 1;
    static constexpr size_t DIM_D = 2;

    // UB 预算系数：
    //   TND 区：10 个 b16 buffer（inQueGradMerge 2 + inQueO 4 + outQueGradO 4）
    //           + 2 个 fp32 buffer（gradMergeFp32 + outFp32）；
    //   TN 区：4 个 b16 buffer（inQueZ/outQueGradZ 各 2 半段）
    //           + 4 个 fp32 buffer（sigmoidG/L、reduceG/L）。
    static constexpr int64_t TND_B16_BUF_NUM = 10;
    static constexpr int64_t TND_FP32_BUF_NUM = 2;
    static constexpr int64_t SCALAR_B16_BUF_NUM = 4;
    static constexpr int64_t SCALAR_FP32_BUF_NUM = 4;
    static constexpr int64_t BATCH_INIT_TND_COEF = TND_B16_BUF_NUM * DTYPE_BYTES +
                                                   TND_FP32_BUF_NUM * static_cast<int64_t>(sizeof(float));
    static constexpr int64_t BATCH_INIT_SCALAR_COEF = SCALAR_B16_BUF_NUM * DTYPE_BYTES +
                                                      SCALAR_FP32_BUF_NUM * static_cast<int64_t>(sizeof(float));

    static const std::set<ge::DataType> INPUT_SUPPORT_DTYPE_SET;

    gert::TilingContext* context_ = nullptr;
    ClaGateBackwardTilingData* data_ = nullptr;
    ClaGateBackwardTilingParam params_;
};

ge::graphStatus TilingForClaGateBackward(gert::TilingContext* context);
ge::graphStatus TilingPrepareForClaGateBackward(gert::TilingParseContext* context);

} // namespace optiling

#endif // CLA_GATE_BACKWARD_TILING_ARCH35_H
