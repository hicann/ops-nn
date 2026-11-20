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
#include <type_traits>
#include <iostream>
#include <string>
#include <functional>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "op_impl_registry.h"
#include "../../../op_host/op_tiling/rms_norm_hi_f4_quant_tiling.h"
#include "op_host/op_const_def_nn.h"
using namespace std;
using namespace ge;
namespace {
using ErrSnapshot = std::decay_t<decltype(ErrorManager::GetInstance().GetRawErrorMessages())>;

ErrSnapshot TakeSnapshot() { return ErrorManager::GetInstance().GetRawErrorMessages(); }

string Report(const ErrSnapshot& errors)
{
    string text;
    for (const auto& error : errors) {
        text += error.error_id + "{";
        for (const auto& field : error.args_map) {
            text += field.first + "=" + field.second + ";";
        }
        text += "} ";
    }
    return text;
}

void ClearErrors()
{
    EXPECT_EQ(ErrorManager::GetInstance().Init(), 0);
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
}

struct ExposedRmsHi4Tiling : optiling::RmsNormHiF4QuantRFullLoadTiling {
    using RmsNormHiF4QuantRFullLoadTiling::RmsNormHiF4QuantRFullLoadTiling;
    void SetBetaFlag(uint32_t flag) { betaFlag_ = flag; }
};

gert::KernelRunContextHolder MakeRmsHi4Context(optiling::RmsNormHiF4QuantCompileInfo* compileInfo,
                                               gert::StorageShape& xShape, gert::StorageShape& gammaShape,
                                               ge::DataType xDtype, ge::DataType gammaDtype, ge::DataType betaDtype,
                                               ge::DataType yDtype, ge::DataType scaleDtype, ge::DataType rstdDtype,
                                               bool outputRstd)
{
    static auto tilingData = gert::TilingData::CreateCap(4096);
    static auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    static fe::PlatFormInfos platform;
    platform.Init();
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    static gert::StorageShape betaShape{{64}, {64}};
    static gert::StorageShape yShape{{8, 64}, {8, 64}};
    static gert::StorageShape scaleShape{{8, 1, 1}, {8, 1, 1}};
    static gert::StorageShape rstdShape{{8, 1}, {8, 1}};
    auto epsilon = Ops::NN::AnyValue::CreateFrom<float>(1e-6f);
    auto outputRstdAttr = Ops::NN::AnyValue::CreateFrom<bool>(outputRstd);
    auto scaleAlg = Ops::NN::AnyValue::CreateFrom<int64_t>(0);
    return gert::TilingContextFaker()
        .SetOpType("RmsNormHiF4Quant")
        .NodeIoNum(3, 3)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&xShape, &gammaShape, &betaShape})
        .OutputShapes({&yShape, &scaleShape, &rstdShape})
        .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, gammaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, betaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, rstdDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"epsilon", epsilon}, {"output_rstd", outputRstdAttr}, {"scale_alg", scaleAlg}})
        .CompileInfo(compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();
}

void RunCase(const char* caseId, gert::StorageShape& xShape, gert::StorageShape& gammaShape, ge::DataType xDtype,
             ge::DataType gammaDtype, ge::DataType betaDtype, ge::DataType yDtype, ge::DataType scaleDtype,
             ge::DataType rstdDtype, bool outputRstd, bool injectBetaFlag,
             ge::graphStatus (*entry)(optiling::RmsNormHiF4QuantRFullLoadTiling&),
             const std::function<void(const ErrSnapshot&)>& verify)
{
    SCOPED_TRACE(caseId);
    ClearErrors();
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto holder = MakeRmsHi4Context(&compileInfo, xShape, gammaShape, xDtype, gammaDtype, betaDtype, yDtype, scaleDtype,
                                    rstdDtype, outputRstd);
    ExposedRmsHi4Tiling tiler(holder.GetContext<gert::TilingContext>());
    if (injectBetaFlag) {
        tiler.SetBetaFlag(1U);
    }
    ge::graphStatus status = entry(tiler);
    ErrSnapshot errors = TakeSnapshot();
    cout << "[" << caseId << "] actual status=" << static_cast<int>(status) << " count=" << errors.size()
         << " errors=" << Report(errors) << endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    verify(errors);
}

string Field(const ErrSnapshot& errors, const string& key)
{
    auto it = errors[0].args_map.find(key);
    return it == errors[0].args_map.end() ? string("<missing>") : it->second;
}
} // namespace

TEST(RmsNormHiF4QuantInputDtypeErrorReporting, Case13539)
{
    static gert::StorageShape xShape{{1, 1, 1, 1, 1, 1, 1, 64}, {1, 1, 1, 1, 1, 1, 1, 64}};
    static gert::StorageShape gammaShape{{64}, {64}};
    RunCase(
        "13539", xShape, gammaShape, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4,
        Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT, false, false,
        [](optiling::RmsNormHiF4QuantRFullLoadTiling& t) { return t.CheckInputShapeDim(); },
        [](const ErrSnapshot& errors) {
            EXPECT_EQ(errors[0].error_id, "EZ0012");
            EXPECT_EQ(Field(errors, "param_name"), "x");
            EXPECT_EQ(Field(errors, "incorrect_dim"), "8");
            EXPECT_EQ(Field(errors, "reason"), "Rank must be in [1, 7]");
        });
}

TEST(RmsNormHiF4QuantInputDtypeErrorReporting, Case13543)
{
    static gert::StorageShape xShape{{8, 64}, {8, 64}};
    static gert::StorageShape gammaShape{{8, 64}, {8, 64}};
    RunCase(
        "13543", xShape, gammaShape, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4,
        Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT, false, false,
        [](optiling::RmsNormHiF4QuantRFullLoadTiling& t) { return t.CheckInputShapeValue(); },
        [](const ErrSnapshot& errors) {
            EXPECT_EQ(errors[0].error_id, "EZ0012");
            EXPECT_EQ(Field(errors, "param_name"), "gamma");
            EXPECT_EQ(Field(errors, "incorrect_dim"), "2");
            EXPECT_EQ(Field(errors, "reason"), "Rank must be 1");
        });
}

TEST(RmsNormHiF4QuantInputDtypeErrorReporting, Case13549)
{
    static gert::StorageShape xShape{{8, 64}, {8, 64}};
    static gert::StorageShape gammaShape{{64}, {64}};
    RunCase(
        "13549", xShape, gammaShape, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4,
        Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT, false, false,
        [](optiling::RmsNormHiF4QuantRFullLoadTiling& t) { return t.CheckInputDtype(); },
        [](const ErrSnapshot& errors) {
            EXPECT_EQ(errors[0].error_id, "EZ0020");
            EXPECT_EQ(Field(errors, "param_name"), "x");
            EXPECT_EQ(Field(errors, "reason"), "Supported dtypes are float16 and bfloat16");
            EXPECT_NE(Field(errors, "incorrect_dtype"), "<missing>");
        });
}

TEST(RmsNormHiF4QuantInputDtypeErrorReporting, Case13550)
{
    static gert::StorageShape xShape{{8, 64}, {8, 64}};
    static gert::StorageShape gammaShape{{64}, {64}};
    RunCase(
        "13550", xShape, gammaShape, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4,
        Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT, false, false,
        [](optiling::RmsNormHiF4QuantRFullLoadTiling& t) { return t.CheckInputDtype(); },
        [](const ErrSnapshot& errors) {
            EXPECT_EQ(errors[0].error_id, "EZ0020");
            EXPECT_EQ(Field(errors, "param_name"), "gamma");
            EXPECT_EQ(Field(errors, "reason"), "Supported dtypes are float16, bfloat16 and float32");
            EXPECT_NE(Field(errors, "incorrect_dtype"), "<missing>");
        });
}

TEST(RmsNormHiF4QuantInputDtypeErrorReporting, Case13551)
{
    static gert::StorageShape xShape{{8, 64}, {8, 64}};
    static gert::StorageShape gammaShape{{64}, {64}};
    RunCase(
        "13551", xShape, gammaShape, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4,
        Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT, false, true,
        [](optiling::RmsNormHiF4QuantRFullLoadTiling& t) { return t.CheckInputDtype(); },
        [](const ErrSnapshot& errors) {
            EXPECT_EQ(errors[0].error_id, "EZ0021");
            EXPECT_EQ(Field(errors, "param_names"), "gamma, beta");
            EXPECT_EQ(Field(errors, "reason"), "Input gamma and beta dtypes must be equal");
            EXPECT_NE(Field(errors, "incorrect_dtypes"), "<missing>");
        });
}

TEST(RmsNormHiF4QuantInputDtypeErrorReporting, Case13553)
{
    static gert::StorageShape xShape{{8, 64}, {8, 64}};
    static gert::StorageShape gammaShape{{64}, {64}};
    RunCase(
        "13553", xShape, gammaShape, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
        Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT, false, false,
        [](optiling::RmsNormHiF4QuantRFullLoadTiling& t) { return t.CheckOutputDtype(); },
        [](const ErrSnapshot& errors) {
            EXPECT_EQ(errors[0].error_id, "EZ0020");
            EXPECT_EQ(Field(errors, "param_name"), "y");
            EXPECT_EQ(Field(errors, "reason"), "Output y dtype must be HIFLOAT4");
            EXPECT_NE(Field(errors, "incorrect_dtype"), "<missing>");
        });
}

TEST(RmsNormHiF4QuantInputDtypeErrorReporting, Case13555)
{
    static gert::StorageShape xShape{{8, 64}, {8, 64}};
    static gert::StorageShape gammaShape{{64}, {64}};
    RunCase(
        "13555", xShape, gammaShape, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4, ge::DT_FLOAT,
        ge::DT_FLOAT, false, false, [](optiling::RmsNormHiF4QuantRFullLoadTiling& t) { return t.CheckOutputDtype(); },
        [](const ErrSnapshot& errors) {
            EXPECT_EQ(errors[0].error_id, "EZ0020");
            EXPECT_EQ(Field(errors, "param_name"), "scale");
            EXPECT_EQ(Field(errors, "reason"), "Output scale dtype must be HIFLOAT4_SCALE");
            EXPECT_NE(Field(errors, "incorrect_dtype"), "<missing>");
        });
}

TEST(RmsNormHiF4QuantInputDtypeErrorReporting, Case13558)
{
    static gert::StorageShape xShape{{8, 64}, {8, 64}};
    static gert::StorageShape gammaShape{{64}, {64}};
    RunCase(
        "13558", xShape, gammaShape, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4,
        Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT16, true, false,
        [](optiling::RmsNormHiF4QuantRFullLoadTiling& t) { return t.CheckOutputDtype(); },
        [](const ErrSnapshot& errors) {
            EXPECT_EQ(errors[0].error_id, "EZ0020");
            EXPECT_EQ(Field(errors, "param_name"), "rstd");
            EXPECT_EQ(Field(errors, "reason"), "Output rstd dtype must be float32");
            EXPECT_NE(Field(errors, "incorrect_dtype"), "<missing>");
        });
}
