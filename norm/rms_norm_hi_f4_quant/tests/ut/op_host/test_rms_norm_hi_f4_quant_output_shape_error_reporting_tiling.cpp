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
#include <type_traits>
#include <iostream>
#include <string>
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

void CheckOneReport(const ErrSnapshot& errors, const string& expectedCode, const vector<string>& expectedParams,
                    const string& expectedReason, const string& expectedIncorrectDims = "")
{
    ASSERT_EQ(errors.size(), 1U);
    if (errors.size() != 1U) {
        return;
    }
    const auto& error = errors[0];
    EXPECT_EQ(error.error_id, expectedCode);
    if (expectedCode == "EZ0013") {
        ASSERT_NE(error.args_map.find("param_names"), error.args_map.end()) << "args_map key [param_names] missing";
        EXPECT_EQ(error.args_map.at("param_names"), expectedParams.at(0));
        ASSERT_NE(error.args_map.find("incorrect_dims"), error.args_map.end())
            << "args_map key [incorrect_dims] missing";
        EXPECT_EQ(error.args_map.at("incorrect_dims"), expectedIncorrectDims);
    } else {
        ASSERT_NE(error.args_map.find("param_name"), error.args_map.end()) << "args_map key [param_name] missing";
        EXPECT_EQ(error.args_map.at("param_name"), expectedParams.at(0));
    }
    ASSERT_NE(error.args_map.find("reason"), error.args_map.end()) << "args_map key [reason] missing";
    EXPECT_EQ(error.args_map.at("reason"), expectedReason);
}

struct ExposedRmsHi4ShapeTiling : optiling::RmsNormHiF4QuantRFullLoadTiling {
    using RmsNormHiF4QuantRFullLoadTiling::RmsNormHiF4QuantRFullLoadTiling;

    void InjectState(uint64_t numCol, uint32_t rstdFlag)
    {
        numCol_ = numCol;
        rstdFlag_ = rstdFlag;
    }
    ge::graphStatus RunOriginalCheckOutputShapeValue() { return CheckOutputShapeValue(); }
};

gert::StorageShape MakeRmsHi4Shape(std::initializer_list<int64_t> dims)
{
    gert::StorageShape shape;
    for (auto d : dims) {
        shape.MutableOriginShape().AppendDim(d);
        shape.MutableStorageShape().AppendDim(d);
    }
    return shape;
}

gert::KernelRunContextHolder MakeRmsHi4ShapeContext(optiling::RmsNormHiF4QuantCompileInfo* compileInfo,
                                                    gert::StorageShape& xShape, gert::StorageShape& gammaShape,
                                                    gert::StorageShape& betaShape, gert::StorageShape& yShape,
                                                    gert::StorageShape& scaleShape, gert::StorageShape& rstdShape)
{
    static auto tilingData = gert::TilingData::CreateCap(4096);
    static auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    static fe::PlatFormInfos platform;
    platform.Init();
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    return gert::TilingContextFaker()
        .SetOpType("RmsNormHiF4Quant")
        .NodeIoNum(3, 3)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&xShape, &gammaShape, &betaShape})
        .OutputShapes({&yShape, &scaleShape, &rstdShape})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, Ops::NN::DT_HIFLOAT4, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({})
        .CompileInfo(compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();
}
} // namespace

TEST(RmsNormHiF4QuantOutputShapeErrorReporting, Case13560ScaleRankNotXRankPlusOne)
{
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto xShape = MakeRmsHi4Shape({8, 64});
    auto gammaShape = MakeRmsHi4Shape({64});
    auto betaShape = MakeRmsHi4Shape({64});
    auto yShape = MakeRmsHi4Shape({8, 64});
    auto scaleShape = MakeRmsHi4Shape({8, 1});
    auto rstdShape = MakeRmsHi4Shape({8, 1});
    auto holder = MakeRmsHi4ShapeContext(&compileInfo, xShape, gammaShape, betaShape, yShape, scaleShape, rstdShape);
    ExposedRmsHi4ShapeTiling tiling(holder.GetContext<gert::TilingContext>());
    tiling.InjectState(64, 0);

    ClearErrors();
    auto ret = tiling.RunOriginalCheckOutputShapeValue();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto errors = TakeSnapshot();
    cout << "[13560 actual] ret=" << static_cast<int>(ret) << " count=" << errors.size() << " report=" << Report(errors)
         << endl;
    CheckOneReport(errors, "EZ0013", {"scale, x"}, "Output scale rank must equal input rank plus 1", "2, 2");
}

TEST(RmsNormHiF4QuantOutputShapeErrorReporting, Case13561ScaleDim0MismatchX)
{
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto xShape = MakeRmsHi4Shape({8, 64});
    auto gammaShape = MakeRmsHi4Shape({64});
    auto betaShape = MakeRmsHi4Shape({64});
    auto yShape = MakeRmsHi4Shape({8, 64});
    auto scaleShape = MakeRmsHi4Shape({7, 1, 1});
    auto rstdShape = MakeRmsHi4Shape({8, 1});
    auto holder = MakeRmsHi4ShapeContext(&compileInfo, xShape, gammaShape, betaShape, yShape, scaleShape, rstdShape);
    ExposedRmsHi4ShapeTiling tiling(holder.GetContext<gert::TilingContext>());
    tiling.InjectState(64, 0);

    ClearErrors();
    auto ret = tiling.RunOriginalCheckOutputShapeValue();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto errors = TakeSnapshot();
    cout << "[13561 actual] ret=" << static_cast<int>(ret) << " count=" << errors.size() << " report=" << Report(errors)
         << endl;
    CheckOneReport(errors, "EZ0009", {"scale"}, "Dimension 0 must equal x dimension 8");
}

TEST(RmsNormHiF4QuantOutputShapeErrorReporting, Case13562ScalePenultimateNotCeilDiv)
{
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto xShape = MakeRmsHi4Shape({8, 64});
    auto gammaShape = MakeRmsHi4Shape({64});
    auto betaShape = MakeRmsHi4Shape({64});
    auto yShape = MakeRmsHi4Shape({8, 64});
    auto scaleShape = MakeRmsHi4Shape({8, 2, 1});
    auto rstdShape = MakeRmsHi4Shape({8, 1});
    auto holder = MakeRmsHi4ShapeContext(&compileInfo, xShape, gammaShape, betaShape, yShape, scaleShape, rstdShape);
    ExposedRmsHi4ShapeTiling tiling(holder.GetContext<gert::TilingContext>());
    tiling.InjectState(64, 0);

    ClearErrors();
    auto ret = tiling.RunOriginalCheckOutputShapeValue();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto errors = TakeSnapshot();
    cout << "[13562 actual] ret=" << static_cast<int>(ret) << " count=" << errors.size() << " report=" << Report(errors)
         << endl;
    CheckOneReport(errors, "EZ0009", {"scale"}, "Penultimate dimension must equal ceil(input last dimension / 64) = 1");
}

TEST(RmsNormHiF4QuantOutputShapeErrorReporting, Case13563ScaleLastDimNotOne)
{
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto xShape = MakeRmsHi4Shape({8, 64});
    auto gammaShape = MakeRmsHi4Shape({64});
    auto betaShape = MakeRmsHi4Shape({64});
    auto yShape = MakeRmsHi4Shape({8, 64});
    auto scaleShape = MakeRmsHi4Shape({8, 1, 2});
    auto rstdShape = MakeRmsHi4Shape({8, 1});
    auto holder = MakeRmsHi4ShapeContext(&compileInfo, xShape, gammaShape, betaShape, yShape, scaleShape, rstdShape);
    ExposedRmsHi4ShapeTiling tiling(holder.GetContext<gert::TilingContext>());
    tiling.InjectState(64, 0);

    ClearErrors();
    auto ret = tiling.RunOriginalCheckOutputShapeValue();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto errors = TakeSnapshot();
    cout << "[13563 actual] ret=" << static_cast<int>(ret) << " count=" << errors.size() << " report=" << Report(errors)
         << endl;
    CheckOneReport(errors, "EZ0009", {"scale"}, "Last dimension must be 1");
}

TEST(RmsNormHiF4QuantOutputShapeErrorReporting, Case13565RstdRankNotXRank)
{
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto xShape = MakeRmsHi4Shape({8, 64});
    auto gammaShape = MakeRmsHi4Shape({64});
    auto betaShape = MakeRmsHi4Shape({64});
    auto yShape = MakeRmsHi4Shape({8, 64});
    auto scaleShape = MakeRmsHi4Shape({8, 1, 1});
    auto rstdShape = MakeRmsHi4Shape({8});
    auto holder = MakeRmsHi4ShapeContext(&compileInfo, xShape, gammaShape, betaShape, yShape, scaleShape, rstdShape);
    ExposedRmsHi4ShapeTiling tiling(holder.GetContext<gert::TilingContext>());
    tiling.InjectState(64, 1);

    ClearErrors();
    auto ret = tiling.RunOriginalCheckOutputShapeValue();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto errors = TakeSnapshot();
    cout << "[13565 actual] ret=" << static_cast<int>(ret) << " count=" << errors.size() << " report=" << Report(errors)
         << endl;
    CheckOneReport(errors, "EZ0013", {"rstd, x"}, "Output rstd rank must equal input rank", "1, 2");
}

TEST(RmsNormHiF4QuantOutputShapeErrorReporting, Case13566RstdDim1NotOne)
{
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto xShape = MakeRmsHi4Shape({8, 64});
    auto gammaShape = MakeRmsHi4Shape({64});
    auto betaShape = MakeRmsHi4Shape({64});
    auto yShape = MakeRmsHi4Shape({8, 64});
    auto scaleShape = MakeRmsHi4Shape({8, 1, 1});
    auto rstdShape = MakeRmsHi4Shape({8, 2});
    auto holder = MakeRmsHi4ShapeContext(&compileInfo, xShape, gammaShape, betaShape, yShape, scaleShape, rstdShape);
    ExposedRmsHi4ShapeTiling tiling(holder.GetContext<gert::TilingContext>());
    tiling.InjectState(64, 1);

    ClearErrors();
    auto ret = tiling.RunOriginalCheckOutputShapeValue();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto errors = TakeSnapshot();
    cout << "[13566 actual] ret=" << static_cast<int>(ret) << " count=" << errors.size() << " report=" << Report(errors)
         << endl;
    CheckOneReport(errors, "EZ0009", {"rstd"}, "Dimension 1 must be 1");
}

TEST(RmsNormHiF4QuantOutputShapeErrorReporting, Case13567RstdDim0MismatchX)
{
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto xShape = MakeRmsHi4Shape({8, 64});
    auto gammaShape = MakeRmsHi4Shape({64});
    auto betaShape = MakeRmsHi4Shape({64});
    auto yShape = MakeRmsHi4Shape({8, 64});
    auto scaleShape = MakeRmsHi4Shape({8, 1, 1});
    auto rstdShape = MakeRmsHi4Shape({7, 1});
    auto holder = MakeRmsHi4ShapeContext(&compileInfo, xShape, gammaShape, betaShape, yShape, scaleShape, rstdShape);
    ExposedRmsHi4ShapeTiling tiling(holder.GetContext<gert::TilingContext>());
    tiling.InjectState(64, 1);

    ClearErrors();
    auto ret = tiling.RunOriginalCheckOutputShapeValue();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto errors = TakeSnapshot();
    cout << "[13567 actual] ret=" << static_cast<int>(ret) << " count=" << errors.size() << " report=" << Report(errors)
         << endl;
    CheckOneReport(errors, "EZ0009", {"rstd"}, "Dimension 0 must equal x dimension 8");
}

TEST(RmsNormHiF4QuantOutputShapeErrorReporting, Case13570OddLastDimPackedHiF4)
{
    optiling::RmsNormHiF4QuantCompileInfo compileInfo{};
    auto xShape = MakeRmsHi4Shape({8, 63});
    auto gammaShape = MakeRmsHi4Shape({63});
    auto betaShape = MakeRmsHi4Shape({63});
    auto yShape = MakeRmsHi4Shape({8, 63});
    auto scaleShape = MakeRmsHi4Shape({8, 1, 1});
    auto rstdShape = MakeRmsHi4Shape({8, 1});
    auto holder = MakeRmsHi4ShapeContext(&compileInfo, xShape, gammaShape, betaShape, yShape, scaleShape, rstdShape);
    ExposedRmsHi4ShapeTiling tiling(holder.GetContext<gert::TilingContext>());
    tiling.InjectState(63, 0);

    ClearErrors();
    auto ret = tiling.RunOriginalCheckOutputShapeValue();
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
    auto errors = TakeSnapshot();
    cout << "[13570 actual] ret=" << static_cast<int>(ret) << " count=" << errors.size() << " report=" << Report(errors)
         << endl;
    CheckOneReport(errors, "EZ0009", {"x"}, "Last input dimension must be even for packed HiF4 output");
}
