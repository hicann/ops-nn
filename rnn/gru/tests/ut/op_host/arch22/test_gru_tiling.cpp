/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_host/gru_tiling.h"

namespace {
const std::vector<uint32_t> kInputIrInstanceAll{1, 1, 1, 1, 1, 1, 1};
const std::vector<uint32_t> kInputIrInstanceNoOpt{1, 1, 1, 0, 0, 0, 0};
const std::vector<uint32_t> kInputIrInstanceBiasInitH{1, 1, 1, 1, 1, 0, 1};
const std::vector<uint32_t> kInputIrInstanceBiasBatchSz{1, 1, 1, 1, 1, 1, 0};
const std::vector<uint32_t> kInputIrInstanceBatchSzOnly{1, 1, 1, 0, 0, 1, 0};
const std::vector<uint32_t> kOutputIrInstance{1, 1, 1, 1, 1, 1};

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;
} // namespace

class GruTiling : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ==================== 正例 ====================

// 正例1: 定长FP16，有偏置，有init_h，3D输入
TEST_F(GruTiling, case_fixed_length_fp16_with_bias_init_h) {
    int64_t T = 4;
    int64_t B = 2;
    int64_t I = 128;
    int64_t H = 256;
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{T, B, I}, {T, B, I}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wh
        {{{gateNum * H}, {gateNum * H}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bi
        {{{gateNum * H}, {gateNum * H}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bh
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // batch_sizes (absent)
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // init_h
    };
    std::vector<TensorDesc> outputs = {
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // y
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // output_h
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // r
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // z
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // n
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceBiasInitH, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    uint64_t expectTilingKey = 0; // GRU_TPL_MM_FP16_SPLIT
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 正例2: 定长FP32，无偏置，无init_h，3D输入
TEST_F(GruTiling, case_fixed_length_fp32_no_bias_no_init_h) {
    int64_t T = 3;
    int64_t B = 4;
    int64_t I = 64;
    int64_t H = 128;
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{T, B, I}, {T, B, I}}, ge::DT_FLOAT, ge::FORMAT_ND},               // x
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // wh
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // bi (absent)
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // bh (absent)
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                              // batch_sizes (absent)
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // init_h (absent)
    };
    std::vector<TensorDesc> outputs = {
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},               // y
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},               // output_h
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},               // r
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},               // z
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},               // n
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},               // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceNoOpt, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    uint64_t expectTilingKey = 1; // GRU_TPL_MM_FP32_SPLIT
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 正例3: 不定长(PackedSequence) FP16，有偏置，有batch_sizes，2D输入
TEST_F(GruTiling, case_variable_length_fp16_with_bias) {
    int64_t totalSteps = 10;
    int64_t B = 3;
    int64_t I = 64;
    int64_t H = 128;
    int64_t T = 5; // batch_sizes length
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{totalSteps, I}, {totalSteps, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x (2D)
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wh
        {{{gateNum * H}, {gateNum * H}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bi
        {{{gateNum * H}, {gateNum * H}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bh
        {{{T}, {T}}, ge::DT_INT64, ge::FORMAT_ND},                             // batch_sizes
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // init_h (absent)
    };
    std::vector<TensorDesc> outputs = {
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // y
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // output_h
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // r
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // z
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // n
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceBiasBatchSz, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    uint64_t expectTilingKey = 0; // GRU_TPL_MM_FP16_SPLIT
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 正例4: 不定长(PackedSequence) FP32，无偏置，无init_h，2D输入
TEST_F(GruTiling, case_variable_length_fp32_no_bias_no_init_h) {
    int64_t totalSteps = 8;
    int64_t B = 4;
    int64_t I = 32;
    int64_t H = 64;
    int64_t T = 4; // batch_sizes length
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{totalSteps, I}, {totalSteps, I}}, ge::DT_FLOAT, ge::FORMAT_ND},   // x (2D)
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT, ge::FORMAT_ND}, // wh
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // bi (absent)
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // bh (absent)
        {{{T}, {T}}, ge::DT_INT64, ge::FORMAT_ND},                            // batch_sizes
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},                              // init_h (absent)
    };
    std::vector<TensorDesc> outputs = {
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT, ge::FORMAT_ND},   // y
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT, ge::FORMAT_ND},               // output_h
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT, ge::FORMAT_ND},   // r
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT, ge::FORMAT_ND},   // z
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT, ge::FORMAT_ND},   // n
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT, ge::FORMAT_ND},   // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceBatchSzOnly, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    uint64_t expectTilingKey = 1; // GRU_TPL_MM_FP32_SPLIT
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 正例5: 定长FP16，无偏置，有init_h，3D输入
TEST_F(GruTiling, case_fixed_length_fp16_no_bias_with_init_h) {
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 32;
    int64_t H = 64;
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{T, B, I}, {T, B, I}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wh
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // bi (absent)
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // bh (absent)
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // batch_sizes (absent)
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // init_h
    };
    std::vector<TensorDesc> outputs = {
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // y
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // output_h
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // r
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // z
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // n
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceNoOpt, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    uint64_t expectTilingKey = 0; // GRU_TPL_MM_FP16_SPLIT
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey);
}

// 正例6: 定长FP16，双向(BIDIRECTIONAL)，有偏置，有init_h，3D输入
TEST_F(GruTiling, case_fixed_length_bidirectional_fp16) {
    int64_t T = 3;
    int64_t B = 2;
    int64_t I = 64;
    int64_t H = 128;
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{T, B, I}, {T, B, I}}, ge::DT_FLOAT16, ge::FORMAT_ND},           // x
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wh
        {{{gateNum * H}, {gateNum * H}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bi
        {{{gateNum * H}, {gateNum * H}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // bh
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // batch_sizes (absent)
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // init_h
    };
    std::vector<TensorDesc> outputs = {
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // y
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // output_h
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // r
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // z
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // n
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("BIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceBiasInitH, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    uint64_t expectTilingKey = 0; // GRU_TPL_MM_FP16_SPLIT
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ==================== 负例 ====================

// 负例1: 输入x维度为1D（非法），期望GRAPH_FAILED
TEST_F(GruTiling, case_invalid_input_dim_1d) {
    int64_t I = 64;
    int64_t H = 128;
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{I}, {I}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // x (1D - invalid)
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wh
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // bi (absent)
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // bh (absent)
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // batch_sizes (absent)
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // init_h (absent)
    };
    std::vector<TensorDesc> outputs = {
        {{{I, H}, {I, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // y
        {{{1, 1, H}, {1, 1, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},             // output_h
        {{{I, H}, {I, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // r
        {{{I, H}, {I, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // z
        {{{I, H}, {I, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // n
        {{{I, H}, {I, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},                    // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceNoOpt, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 负例2: 输入x维度为4D（非法），期望GRAPH_FAILED
TEST_F(GruTiling, case_invalid_input_dim_4d) {
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 32;
    int64_t H = 64;
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{T, B, I, 1}, {T, B, I, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},       // x (4D - invalid)
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wh
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // bi (absent)
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // bh (absent)
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // batch_sizes (absent)
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // init_h (absent)
    };
    std::vector<TensorDesc> outputs = {
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // y
        {{{1, B, H}, {1, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // output_h
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // r
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // z
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // n
        {{{T, B, H}, {T, B, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceNoOpt, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 负例3: 2D输入但没有batch_sizes（不定长模式必须传batch_sizes），期望GRAPH_FAILED
TEST_F(GruTiling, case_variable_length_no_batch_sizes) {
    int64_t totalSteps = 10;
    int64_t I = 64;
    int64_t H = 128;
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{totalSteps, I}, {totalSteps, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // x (2D)
        {{{gateNum * H, I}, {gateNum * H, I}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wi
        {{{gateNum * H, H}, {gateNum * H, H}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // wh
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // bi (absent)
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // bh (absent)
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},                               // batch_sizes (absent)
        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},                             // init_h (absent)
    };
    std::vector<TensorDesc> outputs = {
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // y
        {{{1, 1, H}, {1, 1, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},              // output_h
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // r
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // z
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // n
        {{{totalSteps, H}, {totalSteps, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // n_h
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        kInputIrInstanceNoOpt, kOutputIrInstance,
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 负例4: 输入个数不足（少于2个），期望GRAPH_FAILED
TEST_F(GruTiling, case_insufficient_inputs) {
    int64_t I = 64;
    int64_t H = 128;
    int64_t gateNum = 3;

    std::vector<TensorDesc> inputs = {
        {{{I}, {I}}, ge::DT_FLOAT16, ge::FORMAT_ND},                          // x only
    };
    std::vector<TensorDesc> outputs = {
        {{{I, H}, {I, H}}, ge::DT_FLOAT16, ge::FORMAT_ND},
    };
    std::vector<OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom(std::string("UNIDIRECTIONAL"))},
        {"is_training", Ops::NN::AnyValue::CreateFrom(true)},
    };

    optiling::GruCompileInfo compileInfo{0};
    gert::TilingContextPara para(
        "Gru", inputs, outputs, attrs,
        {1}, {1},
        &compileInfo, "Ascend910B", 1U, 262144ULL, 8192);

    ExecuteTestCase(para, ge::GRAPH_FAILED);
}