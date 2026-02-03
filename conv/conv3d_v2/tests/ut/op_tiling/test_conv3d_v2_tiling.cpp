/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv3d_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#define private public
#define protected public
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "../../../op_host/op_tiling/conv3d_base_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_utils.h"

using Ops::NN::Conv3dTilingEngineApi::Conv3dTilingEngine;

namespace {

static Conv3dTilingEngine BuildEngineFromBase(const optiling::Conv3dOpsTiling::Conv3dBaseTiling &base);
static optiling::Conv3dOpsTiling::BlockDimRes ComputeBlockDimViaEngine(
    const optiling::Conv3dOpsTiling::Conv3dBaseTiling &base);
static std::unique_ptr<Conv3dTilingEngine> MakeInitializedEngineForTest(uint32_t aicoreNum = 20,
                                                                        const char *socVersion = "Ascend910B");
static std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeConv3DV2Attrs(const std::vector<int64_t> &strides,
                                                                               const std::vector<int64_t> &pads,
                                                                               const std::vector<int64_t> &dilations,
                                                                               int64_t groups,
                                                                               const std::string &dataFormat,
                                                                               int64_t offsetX = 0,
                                                                               const std::string &padMode = "SPECIFIC",
                                                                               bool enableHf32 = false);

const char *const DEFAULT_ATTR_VAL = "NONE";
const std::vector<int64_t> kMinimalShape = {1, 1, 1, 1, 1};  // Minimal NCDHW shape for tiling context build.

struct Conv3DV2TilingTestParam {
    std::string caseName;
    std::string opType;
    std::string compileInfo;

    // input of tiling
    std::initializer_list<int64_t> xOriginShape;
    ge::Format xOriginFormat;

    std::initializer_list<int64_t> xStorageShape;
    ge::Format xStorageFormat;

    std::initializer_list<int64_t> filterOriginShape;
    ge::Format filterOriginFormat;

    std::initializer_list<int64_t> filterStorageShape;
    ge::Format filterStorageFormat;

    std::initializer_list<int64_t> yOriginShape;
    ge::Format yOriginFormat;

    std::initializer_list<int64_t> yStorageShape;
    ge::Format yStorageFormat;

    std::vector<int64_t> stridesList;
    std::vector<int64_t> padsList;
    std::vector<int64_t> dilationsList;
    int64_t groups = 1;
    std::string dataFormat;

    // output of tiling
    uint32_t blockDim = 0;
    uint64_t tilingKey = 0;
    int32_t aicoreNum = 0;
};

class Conv3DV2TilingRuntime : public testing::TestWithParam<Conv3DV2TilingTestParam> {
protected:
    struct CompileAndPlatformInfo {
        optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;
        fe::PlatFormInfos platform_info;
    };

    CompileAndPlatformInfo CreateDefaultCompileAndPlatformInfo() {
        CompileAndPlatformInfo info;
        info.compile_info.tilingType = "Conv3DV2";
        info.compile_info.aicoreNum = 20;
        info.compile_info.shortSocVersion = "Ascend910B";
        return info;
    }
};

TEST_F(Conv3DV2TilingRuntime, TestConv3DV2AllDimEqualTo1)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = conv3dBT.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = conv3dBT.shapeInfo_;
    shapeInfo.batch = 1;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 1;
    shapeInfo.wi = 1;
    shapeInfo.cOut = 16;
    shapeInfo.kd = 1;
    shapeInfo.kh = 1;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 1;
    shapeInfo.ho = 1;
    shapeInfo.wo = 1;

    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.groupOpt = conv3dBT.attrInfo_.groups;
    conv3dBT.shapeInfo_.cinOpt = shapeInfo.cIn;
    conv3dBT.shapeInfo_.coutOpt = shapeInfo.cOut;
    auto res = ComputeBlockDimViaEngine(conv3dBT);
    optiling::Conv3dOpsTiling::BlockDimRes expectRes = {1, 1, 1, 1, 1, 1};
    ASSERT_EQ(res.batchDim, expectRes.batchDim);
    ASSERT_EQ(res.mDim, expectRes.mDim);
    ASSERT_EQ(res.nDim, expectRes.nDim);
    ASSERT_EQ(res.doDim, expectRes.doDim);
    ASSERT_EQ(res.groupDim, expectRes.groupDim);
}

TEST_F(Conv3DV2TilingRuntime, TestConv3DV2BatchDimEqualToAicoreNum)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = conv3dBT.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = conv3dBT.shapeInfo_;
    shapeInfo.batch = 4096;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 1;
    shapeInfo.wi = 1;
    shapeInfo.cOut = 16;
    shapeInfo.kd = 1;
    shapeInfo.kh = 1;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 1;
    shapeInfo.ho = 1;
    shapeInfo.wo = 1;

    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.groupOpt = conv3dBT.attrInfo_.groups;
    conv3dBT.shapeInfo_.cinOpt = shapeInfo.cIn;
    conv3dBT.shapeInfo_.coutOpt = shapeInfo.cOut;
    auto res = ComputeBlockDimViaEngine(conv3dBT);
    optiling::Conv3dOpsTiling::BlockDimRes expectRes = {aicoreNum, 1, 1, 1, 1, 1};
    ASSERT_EQ(res.batchDim, expectRes.batchDim);
    ASSERT_EQ(res.mDim, expectRes.mDim);
    ASSERT_EQ(res.nDim, expectRes.nDim);
    ASSERT_EQ(res.doDim, expectRes.doDim);
    ASSERT_EQ(res.groupDim, expectRes.groupDim);
}


TEST_F(Conv3DV2TilingRuntime, TestConv3DV2KB)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = conv3dBT.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = conv3dBT.shapeInfo_;
    shapeInfo.batch = 16;
    shapeInfo.cIn = 3;
    shapeInfo.di = 26;
    shapeInfo.hi = 134;
    shapeInfo.wi = 134;
    shapeInfo.cOut = 64;
    shapeInfo.kd = 7;
    shapeInfo.kh = 7;
    shapeInfo.kw = 7;
    shapeInfo.dOut = 20;
    shapeInfo.ho = 128;
    shapeInfo.wo = 128;

    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.groupOpt = conv3dBT.attrInfo_.groups;
    conv3dBT.shapeInfo_.cinOpt = shapeInfo.cIn;
    conv3dBT.shapeInfo_.coutOpt = shapeInfo.cOut;
    conv3dBT.blockDimRes = ComputeBlockDimViaEngine(conv3dBT);

    conv3dBT.tilingData_.conv3dApiTiling.groups = 1;
    conv3dBT.tilingData_.conv3dApiTiling.orgCi = 3;
    conv3dBT.tilingData_.conv3dApiTiling.orgDi = 26;
    conv3dBT.tilingData_.conv3dApiTiling.orgHi = 134;
    conv3dBT.tilingData_.conv3dApiTiling.orgWi = 134;
    conv3dBT.tilingData_.conv3dApiTiling.orgDo = 20;
    conv3dBT.tilingData_.conv3dApiTiling.orgCo = 20;
    conv3dBT.tilingData_.conv3dApiTiling.orgHo = 128;
    conv3dBT.tilingData_.conv3dApiTiling.orgWo = 128;
    conv3dBT.tilingData_.conv3dApiTiling.kernelD = 7;
    conv3dBT.tilingData_.conv3dApiTiling.kernelH = 7;
    conv3dBT.tilingData_.conv3dApiTiling.kernelW = 7;
    conv3dBT.tilingData_.conv3dApiTiling.singleCoreDo = 10;
    conv3dBT.tilingData_.conv3dApiTiling.singleCoreCo = 64;
    conv3dBT.tilingData_.conv3dApiTiling.singleCoreM = 16384;
    conv3dBT.tilingData_.conv3dApiTiling.mL0 = 512;
    conv3dBT.tilingData_.conv3dApiTiling.kL0 = 32;
    conv3dBT.tilingData_.conv3dApiTiling.nL0 = 64;
    conv3dBT.tilingData_.conv3dApiTiling.kAL1 = 1568;
    conv3dBT.tilingData_.conv3dApiTiling.kBL1 = 1568;
    conv3dBT.tilingData_.conv3dApiTiling.nBL1 = 64;
    conv3dBT.tilingData_.conv3dApiTiling.mAL1 = 512;
    conv3dBT.tilingData_.conv3dApiTiling.strideD = 1;
    conv3dBT.tilingData_.conv3dApiTiling.strideH = 1;
    conv3dBT.tilingData_.conv3dApiTiling.strideW = 1;
    conv3dBT.tilingData_.conv3dApiTiling.dilationD = 1;
    conv3dBT.tilingData_.conv3dApiTiling.dilationH = 1;
    conv3dBT.tilingData_.conv3dApiTiling.dilationW = 1;
    conv3dBT.tilingData_.conv3dApiTiling.padHead = 0;
    conv3dBT.tilingData_.conv3dApiTiling.padTail = 0;
    conv3dBT.tilingData_.conv3dApiTiling.padTop = 0;
    conv3dBT.tilingData_.conv3dApiTiling.padBottom = 0;
    conv3dBT.tilingData_.conv3dApiTiling.padLeft = 0;
    conv3dBT.tilingData_.conv3dApiTiling.padRight = 0;
    conv3dBT.tilingData_.conv3dApiTiling.pBufferFlag = 11;
    conv3dBT.tilingData_.conv3dApiTiling.iterateMNOrder = 0;
    conv3dBT.tilingData_.conv3dApiTiling.bl1FullLoad = 0;
    conv3dBT.tilingData_.conv3dApiTiling.al1FullLoad = 0;
    conv3dBT.tilingData_.conv3dApiTiling.bl1BypassFlag = 0;
    conv3dBT.tilingData_.conv3dApiTiling.biasFullLoadFlag = 0;
    conv3dBT.tilingData_.conv3dApiTiling.fixpParamsFullLoadFlag = 0;
    conv3dBT.tilingData_.conv3dApiTiling.offsetx = 0;
    conv3dBT.SetAdditionalTilingInfo();
    ASSERT_EQ(conv3dBT.tilingData_.conv3dApiTiling.groupOpt, conv3dBT.attrInfo_.groupOpt);
}


TEST_F(Conv3DV2TilingRuntime, TestConv3DV2NdimEqualToAicoreNum)
{
    uint32_t aicoreNum = 16;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = conv3dBT.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = conv3dBT.shapeInfo_;
    shapeInfo.batch = 1;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 1;
    shapeInfo.wi = 1;
    shapeInfo.cOut = 4096;
    shapeInfo.kd = 1;
    shapeInfo.kh = 1;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 1;
    shapeInfo.ho = 1;
    shapeInfo.wo = 1;

    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.groupOpt = conv3dBT.attrInfo_.groups;
    conv3dBT.shapeInfo_.cinOpt = shapeInfo.cIn;
    conv3dBT.shapeInfo_.coutOpt = shapeInfo.cOut;
    auto res = ComputeBlockDimViaEngine(conv3dBT);
    optiling::Conv3dOpsTiling::BlockDimRes expectRes = {1, 1, aicoreNum, 1, 1, 1};
    ASSERT_EQ(res.batchDim, expectRes.batchDim);
    ASSERT_EQ(res.mDim, expectRes.mDim);
    ASSERT_EQ(res.nDim, expectRes.nDim);
    ASSERT_EQ(res.doDim, expectRes.doDim);
    ASSERT_EQ(res.groupDim, expectRes.groupDim);
}

TEST_F(Conv3DV2TilingRuntime, TestConv3DV2MdimEqualToAicoreNum)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = conv3dBT.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = conv3dBT.shapeInfo_;
    shapeInfo.batch = 1;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 1;
    shapeInfo.wi = 1;
    shapeInfo.cOut = 16;
    shapeInfo.kd = 1;
    shapeInfo.kh = 1;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 1;
    shapeInfo.ho = 16;
    shapeInfo.wo = 256;

    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.groupOpt = conv3dBT.attrInfo_.groups;
    conv3dBT.shapeInfo_.cinOpt = shapeInfo.cIn;
    conv3dBT.shapeInfo_.coutOpt = shapeInfo.cOut;
    auto res = ComputeBlockDimViaEngine(conv3dBT);
    optiling::Conv3dOpsTiling::BlockDimRes expectRes = {1, aicoreNum, 1, 1, 1, 1};
    ASSERT_EQ(res.batchDim, expectRes.batchDim);
    ASSERT_EQ(res.mDim, expectRes.mDim);
    ASSERT_EQ(res.nDim, expectRes.nDim);
    ASSERT_EQ(res.doDim, expectRes.doDim);
    ASSERT_EQ(res.groupDim, expectRes.groupDim);
}

TEST_F(Conv3DV2TilingRuntime, TestConv3DV2DodimEqualToAicoreNum)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = conv3dBT.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = conv3dBT.shapeInfo_;
    shapeInfo.batch = 1;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 1;
    shapeInfo.wi = 1;
    shapeInfo.cOut = 16;
    shapeInfo.kd = 1;
    shapeInfo.kh = 1;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 4096;
    shapeInfo.ho = 1;
    shapeInfo.wo = 1;

    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.groupOpt = conv3dBT.attrInfo_.groups;
    conv3dBT.shapeInfo_.cinOpt = shapeInfo.cIn;
    conv3dBT.shapeInfo_.coutOpt = shapeInfo.cOut;
    auto res = ComputeBlockDimViaEngine(conv3dBT);
    optiling::Conv3dOpsTiling::BlockDimRes expectRes = {1, 1, 1, aicoreNum, 1, 1};
    ASSERT_EQ(res.batchDim, expectRes.batchDim);
    ASSERT_EQ(res.mDim, expectRes.mDim);
    ASSERT_EQ(res.nDim, expectRes.nDim);
    ASSERT_EQ(res.doDim, expectRes.doDim);
    ASSERT_EQ(res.groupDim, expectRes.groupDim);
}

TEST_F(Conv3DV2TilingRuntime, TestConv3DV2GetGroupOptSuccess)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    conv3dBT.attrInfo_.groups = 64;
    conv3dBT.shapeInfo_.cIn = 64;
    conv3dBT.shapeInfo_.cOut = 128;
    conv3dBT.descInfo_.fMapDtype = ge::DT_BF16;

    auto engine = BuildEngineFromBase(conv3dBT);
    bool res = engine.GetGroupConvOpt();
    EXPECT_TRUE(res);
}

TEST_F(Conv3DV2TilingRuntime, TestConv3DV2BlockDimEqualToHoDim)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = conv3dBT.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = conv3dBT.shapeInfo_;
    shapeInfo.batch = 1;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 29;
    shapeInfo.wi = 800;
    shapeInfo.cOut = 1;
    shapeInfo.kd = 1;
    shapeInfo.kh = 20;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 1;
    shapeInfo.ho = 10;
    shapeInfo.wo = 800;

    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.groupOpt = conv3dBT.attrInfo_.groups;
    conv3dBT.shapeInfo_.cinOpt = shapeInfo.cIn;
    conv3dBT.shapeInfo_.coutOpt = shapeInfo.cOut;
    conv3dBT.outputOrder_ = 1;

    auto res = ComputeBlockDimViaEngine(conv3dBT);
    optiling::Conv3dOpsTiling::BlockDimRes expectRes = {1, static_cast<uint32_t>(shapeInfo.ho), 1, 1, 1, 1};
    ASSERT_EQ(res.batchDim, expectRes.batchDim);
    ASSERT_EQ(res.mDim, expectRes.mDim);
    ASSERT_EQ(res.nDim, expectRes.nDim);
    ASSERT_EQ(res.doDim, expectRes.doDim);
    ASSERT_EQ(res.groupDim, expectRes.groupDim);
}

TEST_F(Conv3DV2TilingRuntime, TestConv3DV2InitOutputOrderMMode)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    optiling::Conv3DTilingParseInfo &opInfo = conv3dBT.opInfo_;
    opInfo.aicoreNum = aicoreNum;
    opInfo.l2Rate = 110;
    opInfo.l1Size = 64; // make sure M_MODE fits into L1

    optiling::Conv3dOpsTiling::Conv3DAscendcShapesInfo &shapeInfo = conv3dBT.shapeInfo_;
    shapeInfo.batch = 1;
    shapeInfo.cIn = 1;
    shapeInfo.di = 1;
    shapeInfo.hi = 1;
    shapeInfo.wi = 1;
    shapeInfo.cOut = 16;
    shapeInfo.kd = 1;
    shapeInfo.kh = 1;
    shapeInfo.kw = 1;
    shapeInfo.dOut = 1;
    shapeInfo.ho = 1;
    shapeInfo.wo = 1;

    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.groupOpt = conv3dBT.attrInfo_.groups;
    conv3dBT.shapeInfo_.cinOpt = shapeInfo.cIn;
    conv3dBT.shapeInfo_.coutOpt = shapeInfo.cOut;

    auto engine = BuildEngineFromBase(conv3dBT);
    bool ok = engine.InitOutputOrder();
    ASSERT_TRUE(ok);
    // outputOrder_ should be M_MODE (0)
    ASSERT_EQ(engine.outputOrder_, 0);

    uint64_t minL1LoadSize = engine.CalcMinL1LoadSize(0);
    ASSERT_LE(minL1LoadSize, opInfo.l1Size);

    engine.blockDimRes_.batchDim = 1;
    engine.blockDimRes_.mDim = 4;
    engine.blockDimRes_.nDim = 1;
    engine.blockDimRes_.doDim = 1;
    engine.blockDimRes_.groupDim = 1;
    engine.SetSingleOutputShapeByMode();
    EXPECT_EQ(engine.conv3dApiTiling_.shapeInfo.singleCo, static_cast<int64_t>(shapeInfo.cOut));
    EXPECT_EQ(engine.conv3dApiTiling_.shapeInfo.singleDo, static_cast<int64_t>(shapeInfo.dOut));
    uint64_t expectedSingleM = optiling::Conv3dOpsTiling::CeilDiv(
        static_cast<uint64_t>(shapeInfo.ho * shapeInfo.wo),
        static_cast<uint64_t>(engine.blockDimRes_.mDim));
    EXPECT_EQ(engine.conv3dApiTiling_.shapeInfo.singleM, static_cast<uint64_t>(expectedSingleM));
}

TEST_F(Conv3DV2TilingRuntime, TestConv3DV2CheckPointWiseSuccess)
{
    uint32_t aicoreNum = 20;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .Build();
    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling conv3dBT(tilingContext);
    conv3dBT.isPointWise = true;
    conv3dBT.shapeInfo_.kh = 1;
    conv3dBT.shapeInfo_.kw = 1;
    conv3dBT.shapeInfo_.kd = 1;
    conv3dBT.attrInfo_.groups = 1;
    conv3dBT.attrInfo_.strideH = 1;
    conv3dBT.attrInfo_.strideW = 1;
    conv3dBT.attrInfo_.strideD = 1;
    conv3dBT.attrInfo_.dilationH = 1;
    conv3dBT.attrInfo_.dilationW = 1;
    conv3dBT.attrInfo_.dilationD = 1;
    conv3dBT.attrInfo_.padh = 0;
    conv3dBT.attrInfo_.padt = 0;
    conv3dBT.attrInfo_.padu = 0;
    conv3dBT.attrInfo_.padd = 0;
    conv3dBT.attrInfo_.padl = 0;
    conv3dBT.attrInfo_.padr = 0;

    auto engine = BuildEngineFromBase(conv3dBT);
    bool res = engine.CheckPointWiseParams();
    EXPECT_TRUE(res);
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractOriginShapes_NCDHW)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs(
                          /*strides*/ {1, 1, 1, 1, 1},
                          /*pads*/ {0, 0, 0, 0, 0, 0},
                          /*dilations*/ {1, 1, 1, 1, 1},
                          /*groups*/ 1,
                          /*dataFormat*/ "NCDHW"))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.InitConv3dOriginFormat();

    EXPECT_EQ(base.ExtractOriginFmapShape(), (std::vector<int64_t>{2, 3, 4, 5, 6}));
    EXPECT_EQ(base.ExtractOriginWeightShape(), (std::vector<int64_t>{7, 3, 1, 3, 3}));
    EXPECT_EQ(base.ExtractOriginOutputShape(), (std::vector<int64_t>{2, 7, 4, 5, 6}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractOriginShapes_NDHWC)
{
    // NDHWC input + DHWCN weight + NDHWC output.
    gert::StorageShape fmap = {{2, 4, 5, 6, 3}, {2, 4, 5, 6, 3}};           // N D H W C
    gert::StorageShape weight = {{1, 3, 3, 3, 7}, {1, 3, 3, 3, 7}};         // D H W C N
    gert::StorageShape out = {{2, 4, 5, 6, 7}, {2, 4, 5, 6, 7}};            // N D H W C

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_NDHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_DHWCN)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_NDHWC)
                      .NodeAttrs(MakeConv3DV2Attrs(
                          /*strides*/ {1, 1, 1, 1, 1},
                          /*pads*/ {0, 0, 0, 0, 0, 0},
                          /*dilations*/ {1, 1, 1, 1, 1},
                          /*groups*/ 1,
                          /*dataFormat*/ "NDHWC"))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.InitConv3dOriginFormat();

    EXPECT_EQ(base.ExtractOriginFmapShape(), (std::vector<int64_t>{2, 3, 4, 5, 6}));
    EXPECT_EQ(base.ExtractOriginWeightShape(), (std::vector<int64_t>{7, 3, 1, 3, 3}));
    EXPECT_EQ(base.ExtractOriginOutputShape(), (std::vector<int64_t>{2, 7, 4, 5, 6}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractPadList_ValidPadding)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3, 4, 5, 6})},
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    std::vector<int64_t> padList;
    ASSERT_TRUE(base.ExtractPadList(padList));
    EXPECT_EQ(padList, (std::vector<int64_t>{1, 2, 3, 4, 5, 6}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractPadList_MissingPadding)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    // Only provide strides at attr index 0; pads (index 1) should be null/out-of-range.
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    std::vector<int64_t> padList;
    EXPECT_FALSE(base.ExtractPadList(padList));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractPadList_InvalidDimension)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3})},  // Wrong size.
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    std::vector<int64_t> padList;
    EXPECT_FALSE(base.ExtractPadList(padList));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractStrideList_NCDHW)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 3, 4})},
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.originalFormat_.FORMAT_DATA_D_INDEX = 2;
    base.originalFormat_.FORMAT_DATA_H_INDEX = 3;
    base.originalFormat_.FORMAT_DATA_W_INDEX = 4;

    std::vector<int64_t> strideList;
    ASSERT_TRUE(base.ExtractStrideList(strideList));
    EXPECT_EQ(strideList, (std::vector<int64_t>{2, 3, 4}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractStrideList_NDHWC)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3, 4, 1})},
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_NDHWC)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_DHWCN)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_NDHWC)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.originalFormat_.FORMAT_DATA_D_INDEX = 1;
    base.originalFormat_.FORMAT_DATA_H_INDEX = 2;
    base.originalFormat_.FORMAT_DATA_W_INDEX = 3;

    std::vector<int64_t> strideList;
    ASSERT_TRUE(base.ExtractStrideList(strideList));
    EXPECT_EQ(strideList, (std::vector<int64_t>{2, 3, 4}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractStrideList_InvalidDimension)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3})},  // Wrong size.
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.originalFormat_.FORMAT_DATA_D_INDEX = 2;
    base.originalFormat_.FORMAT_DATA_H_INDEX = 3;
    base.originalFormat_.FORMAT_DATA_W_INDEX = 4;

    std::vector<int64_t> strideList;
    EXPECT_FALSE(base.ExtractStrideList(strideList));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractDilationList_DefaultValue)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    // Only provide strides/pads; dilations (index 2) is missing/out-of-range.
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.originalFormat_.FORMAT_DATA_D_INDEX = 2;
    base.originalFormat_.FORMAT_DATA_H_INDEX = 3;
    base.originalFormat_.FORMAT_DATA_W_INDEX = 4;

    std::vector<int64_t> dilationList;
    ASSERT_TRUE(base.ExtractDilationList(dilationList));
    EXPECT_EQ(dilationList, (std::vector<int64_t>{1, 1, 1}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractDilationList_Valid)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
        {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 2, 3, 4})},
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.originalFormat_.FORMAT_DATA_D_INDEX = 2;
    base.originalFormat_.FORMAT_DATA_H_INDEX = 3;
    base.originalFormat_.FORMAT_DATA_W_INDEX = 4;

    std::vector<int64_t> dilationList;
    ASSERT_TRUE(base.ExtractDilationList(dilationList));
    EXPECT_EQ(dilationList, (std::vector<int64_t>{2, 3, 4}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractDilationList_InvalidDimension)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
        {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2, 3})},  // Wrong size.
    };

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(attrs)
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.originalFormat_.FORMAT_DATA_D_INDEX = 2;
    base.originalFormat_.FORMAT_DATA_H_INDEX = 3;
    base.originalFormat_.FORMAT_DATA_W_INDEX = 4;

    std::vector<int64_t> dilationList;
    EXPECT_FALSE(base.ExtractDilationList(dilationList));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractGroups_DefaultAndPresent)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    // Default group (missing index 3) should return 1.
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrsDefault = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
        {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
    };

    auto info_default = CreateDefaultCompileAndPlatformInfo();

    auto holderDefault = gert::TilingContextFaker()
                             .SetOpType("Conv3DV2")
                             .NodeIoNum(2, 1)
                             .IrInstanceNum({1, 1})
                             .InputShapes({&fmap, &weight})
                             .OutputShapes({&out})
                             .CompileInfo(&info_default.compile_info)
                             .PlatformInfo(reinterpret_cast<char *>(&info_default.platform_info))
                             .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                             .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                             .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                             .NodeAttrs(attrsDefault)
                             .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling baseDefault(holderDefault.GetContext<gert::TilingContext>());
    EXPECT_EQ(baseDefault.ExtractGroups(), 1);

    // Present group at index 3.
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrsPresent = {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0, 0, 0, 0, 0})},
        {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1, 1})},
        {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
    };

    auto info_present = CreateDefaultCompileAndPlatformInfo();

    auto holderPresent = gert::TilingContextFaker()
                             .SetOpType("Conv3DV2")
                             .NodeIoNum(2, 1)
                             .IrInstanceNum({1, 1})
                             .InputShapes({&fmap, &weight})
                             .OutputShapes({&out})
                             .CompileInfo(&info_present.compile_info)
                             .PlatformInfo(reinterpret_cast<char *>(&info_present.platform_info))
                             .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                             .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                             .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                             .NodeAttrs(attrsPresent)
                             .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling basePresent(holderPresent.GetContext<gert::TilingContext>());
    EXPECT_EQ(basePresent.ExtractGroups(), 3);
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractBiasAndScaleShape_Optional)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};
    gert::StorageShape bias = {{7}, {7}};
    gert::StorageShape scale = {{7}, {7}};

    // No bias/scale.
    auto info_none = CreateDefaultCompileAndPlatformInfo();

    auto holderNone = gert::TilingContextFaker()
                          .SetOpType("Conv3DV2")
                          .NodeIoNum(4, 1)
                          .IrInstanceNum({1, 1, 1, 1})
                          .InputShapes({&fmap, &weight, nullptr, nullptr})
                          .OutputShapes({&out})
                          .CompileInfo(&info_none.compile_info)
                          .PlatformInfo(reinterpret_cast<char *>(&info_none.platform_info))
                          .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                      {0, 0, 0, 0, 0, 0},
                                                      {1, 1, 1, 1, 1},
                                                      1,
                                                      "NCDHW"))
                          .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling baseNone(holderNone.GetContext<gert::TilingContext>());
    std::vector<int64_t> biasShapeNone;
    std::vector<int64_t> scaleShapeNone;
    EXPECT_TRUE(baseNone.ExtractBiasShape(biasShapeNone));
    EXPECT_TRUE(baseNone.ExtractScaleShape(scaleShapeNone));
    EXPECT_TRUE(biasShapeNone.empty());
    EXPECT_TRUE(scaleShapeNone.empty());

    // With bias/scale.
    auto info_both = CreateDefaultCompileAndPlatformInfo();

    auto holderBoth = gert::TilingContextFaker()
                          .SetOpType("Conv3DV2")
                          .NodeIoNum(4, 1)
                          .IrInstanceNum({1, 1, 1, 1})
                          .InputShapes({&fmap, &weight, &bias, &scale})
                          .OutputShapes({&out})
                          .CompileInfo(&info_both.compile_info)
                          .PlatformInfo(reinterpret_cast<char *>(&info_both.platform_info))
                          .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                          .NodeInputTd(3, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                      {0, 0, 0, 0, 0, 0},
                                                      {1, 1, 1, 1, 1},
                                                      1,
                                                      "NCDHW"))
                          .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling baseBoth(holderBoth.GetContext<gert::TilingContext>());
    std::vector<int64_t> biasShape;
    std::vector<int64_t> scaleShape;
    EXPECT_TRUE(baseBoth.ExtractBiasShape(biasShape));
    EXPECT_TRUE(baseBoth.ExtractScaleShape(scaleShape));
    EXPECT_EQ(biasShape, (std::vector<int64_t>{7}));
    EXPECT_EQ(scaleShape, (std::vector<int64_t>{7}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractAndSetDataTypesAndFormats)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW"))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.engine_ = MakeInitializedEngineForTest();

    base.ExtractAndSetDataTypes();
    base.ExtractAndSetFormats();

    EXPECT_EQ(base.engine_->descInfo_.fMapDtype, Conv3dApiTiling::ConvDtype::BF16);
    EXPECT_EQ(base.engine_->descInfo_.weightDtype, Conv3dApiTiling::ConvDtype::BF16);
    EXPECT_EQ(base.engine_->descInfo_.outDtype, Conv3dApiTiling::ConvDtype::BF16);
    EXPECT_EQ(base.engine_->descInfo_.fMapFormat, Conv3dApiTiling::ConvFormat::NCDHW);
    EXPECT_EQ(base.engine_->descInfo_.weightFormat, Conv3dApiTiling::ConvFormat::NCDHW);
    EXPECT_EQ(base.engine_->descInfo_.outFormat, Conv3dApiTiling::ConvFormat::NCDHW);
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractAndSetBiasScale_HF32Enabled)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape bias = {{7}, {7}};
    gert::StorageShape scale = {{7}, {7}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&fmap, &weight, &bias, &scale})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW",
                                                  /*offsetX*/ 0,
                                                  /*padMode*/ "SPECIFIC",
                                                  /*enableHf32*/ true))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.engine_ = MakeInitializedEngineForTest();

    // HF32 enabling depends on BaseTiling descInfo_ (not Engine descInfo_).
    base.descInfo_.fMapDtype = ge::DT_FLOAT;
    base.descInfo_.weightDtype = ge::DT_FLOAT;
    base.descInfo_.outDtype = ge::DT_FLOAT;
    base.descInfo_.biasDtype = ge::DT_FLOAT;

    ASSERT_TRUE(base.ExtractAndSetBiasScale());
    EXPECT_TRUE(base.flagInfo_.hasBias);
    EXPECT_TRUE(base.flagInfo_.hasScale);
    EXPECT_TRUE(base.engine_->flagInfo_.hasBias);
    EXPECT_TRUE(base.engine_->flagInfo_.hasScale);
    EXPECT_EQ(base.engine_->biasShape_, (std::vector<int64_t>{7}));
    EXPECT_EQ(base.engine_->scaleShape_, (std::vector<int64_t>{7}));
    EXPECT_NE(base.engine_->attrInfo_.hf32Enable, 0);
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingPostTiling_BufferBounds)
{
    gert::StorageShape fmap = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape weight = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    gert::StorageShape out = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};

    optiling::Conv3DTilingParseInfo compileInfo;
    compileInfo.socVersion = "Ascend910B";
    compileInfo.aicoreNum = 20;

    fe::PlatFormInfos platform_info;

    // Failure: tiling buffer capacity too small.
    auto tilingSmall = gert::TilingData::CreateCap(1);
    ASSERT_NE(tilingSmall, nullptr);
    auto holderFail = gert::TilingContextFaker()
                          .SetOpType("Conv3DV2")
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({1, 1})
                          .InputShapes({&fmap, &weight})
                          .OutputShapes({&out})
                          .CompileInfo(&compileInfo)
                          .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                          .TilingData(tilingSmall.get())
                          .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                          .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                      {0, 0, 0, 0, 0, 0},
                                                      {1, 1, 1, 1, 1},
                                                      1,
                                                      "NCDHW"))
                          .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling baseFail(holderFail.GetContext<gert::TilingContext>());
    baseFail.tilingData_ = {};
    baseFail.blockDimRes = {1, 1, 1, 1, 1, 0};
    EXPECT_EQ(baseFail.PostTiling(), ge::GRAPH_FAILED);

    // Success: tiling buffer large enough.
    auto tilingOk = gert::TilingData::CreateCap(4096);
    ASSERT_NE(tilingOk, nullptr);
    auto holderOk = gert::TilingContextFaker()
                        .SetOpType("Conv3DV2")
                        .NodeIoNum(2, 1)
                        .IrInstanceNum({1, 1})
                        .InputShapes({&fmap, &weight})
                        .OutputShapes({&out})
                        .CompileInfo(&compileInfo)
                        .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                        .TilingData(tilingOk.get())
                        .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                        .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                        .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                        .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                    {0, 0, 0, 0, 0, 0},
                                                    {1, 1, 1, 1, 1},
                                                    1,
                                                    "NCDHW"))
                        .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling baseOk(holderOk.GetContext<gert::TilingContext>());
    baseOk.tilingData_ = {};
    baseOk.blockDimRes = {1, 1, 1, 1, 1, 0};
    EXPECT_EQ(baseOk.PostTiling(), ge::GRAPH_SUCCESS);
}

static std::unique_ptr<Conv3dTilingEngine> MakeInitializedEngineForTest(uint32_t aicoreNum, const char *socVersion)
{
    auto engine = std::make_unique<Conv3dTilingEngine>("Conv3DV2");
    engine->platformInfo_.aicoreNum = aicoreNum;
    engine->platformInfo_.l1Size = 524288;
    engine->platformInfo_.l0aSize = 65536;
    engine->platformInfo_.l0bSize = 65536;
    engine->platformInfo_.l0cSize = 262144;
    engine->platformInfo_.ubSize = 196608;
    engine->platformInfo_.btSize = 0;
    engine->platformInfo_.l2Rate = 100;
    engine->platformInfo_.socVersion = (socVersion != nullptr) ? socVersion : "Ascend910B";
    engine->SetInitialized(true);
    return engine;
}

static std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeConv3DV2Attrs(const std::vector<int64_t> &strides,
                                                                               const std::vector<int64_t> &pads,
                                                                               const std::vector<int64_t> &dilations,
                                                                               int64_t groups,
                                                                               const std::string &dataFormat,
                                                                               int64_t offsetX,
                                                                               const std::string &padMode,
                                                                               bool enableHf32)
{
    // NOTE: In these UT fakers, attr names are not used to bind by schema; values are appended in order.
    // Keep the append order aligned with conv3d_tiling_utils.h attr indices:
    // 0 strides, 1 pads, 2 dilations, 3 groups, 4 data_format, 5 offset_x, 6 pad_mode, 7 enable_hf32.
    return {
        {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
        {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
        {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilations)},
        {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(groups)},
        {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)},
        {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(offsetX)},
        {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>(padMode)},
        {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(enableHf32)},
    };
}

// ---------------------------------------------------------------------------
// BlockDimDecision helpers using Conv3dTilingEngine directly
// ---------------------------------------------------------------------------

static Conv3dTilingEngine BuildEngineFromBase(const optiling::Conv3dOpsTiling::Conv3dBaseTiling &base)
{
    Conv3dTilingEngine engine("Conv3DV2");
    engine.platformInfo_.aicoreNum = static_cast<uint32_t>(base.opInfo_.aicoreNum);
    engine.platformInfo_.l1Size = base.opInfo_.l1Size;
    engine.platformInfo_.l0aSize = base.opInfo_.l0aSize;
    engine.platformInfo_.l0bSize = base.opInfo_.l0bSize;
    engine.platformInfo_.l0cSize = base.opInfo_.l0cSize;
    engine.platformInfo_.ubSize = base.opInfo_.ubSize;
    engine.platformInfo_.btSize = base.opInfo_.btSize;
    engine.platformInfo_.l2Rate = base.opInfo_.l2Rate;
    engine.platformInfo_.socVersion = base.opInfo_.socVersion;

    engine.shapeInfo_ = base.shapeInfo_;
    engine.attrInfo_.groups = static_cast<int64_t>(base.attrInfo_.groups);
    engine.attrInfo_.groupOpt = static_cast<int64_t>(base.attrInfo_.groupOpt);
    engine.outputOrder_ = base.outputOrder_;
    engine.isPointWise = base.isPointWise;

    // Default dtype/format for tests (align with BF16 NCDHW setup)
    engine.descInfo_.fMapDtype = Conv3dApiTiling::ConvDtype::BF16;
    engine.descInfo_.weightDtype = Conv3dApiTiling::ConvDtype::BF16;
    engine.descInfo_.outDtype = Conv3dApiTiling::ConvDtype::BF16;
    engine.descInfo_.fMapFormat = Conv3dApiTiling::ConvFormat::NCDHW;
    engine.descInfo_.weightFormat = Conv3dApiTiling::ConvFormat::NCDHW;
    engine.descInfo_.outFormat = Conv3dApiTiling::ConvFormat::NCDHW;
    return engine;
}

static optiling::Conv3dOpsTiling::BlockDimRes ComputeBlockDimViaEngine(
    const optiling::Conv3dOpsTiling::Conv3dBaseTiling &base)
{
    auto engine = BuildEngineFromBase(base);
    engine.GetBlockDimRange();
    engine.GetBlockDimInit();
    engine.CoreBlockDimDecision();

    optiling::Conv3dOpsTiling::BlockDimRes res;
    engine.GetBlockDimDetail(res.batchDim, res.mDim, res.nDim, res.doDim, res.groupDim);
    return res;
}

// ============================================================================
// GetDescInfo() Coverage Tests
// ============================================================================

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingGetDescInfo_WithBiasAndScale)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape bias = {{7}, {7}};
    gert::StorageShape scale = {{7}, {7}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&fmap, &weight, &bias, &scale})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW"))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.flagInfo_.hasBias = true;
    base.flagInfo_.hasScale = true;

    base.GetDescInfo();

    EXPECT_EQ(base.descInfo_.fMapDtype, ge::DT_BF16);
    EXPECT_EQ(base.descInfo_.weightDtype, ge::DT_BF16);
    EXPECT_EQ(base.descInfo_.outDtype, ge::DT_BF16);
    EXPECT_EQ(base.descInfo_.biasDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.descInfo_.scaleDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.descInfo_.fMapFormat, ge::Format::FORMAT_NCDHW);
    EXPECT_EQ(base.descInfo_.biasFormat, ge::Format::FORMAT_ND);
    EXPECT_EQ(base.descInfo_.scaleFormat, ge::Format::FORMAT_ND);
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingGetDescInfo_WithoutBiasAndScale)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW"))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.flagInfo_.hasBias = false;
    base.flagInfo_.hasScale = false;

    base.GetDescInfo();

    EXPECT_EQ(base.descInfo_.fMapDtype, ge::DT_BF16);
    EXPECT_EQ(base.descInfo_.weightDtype, ge::DT_BF16);
    EXPECT_EQ(base.descInfo_.outDtype, ge::DT_BF16);
    EXPECT_EQ(base.descInfo_.fMapFormat, ge::Format::FORMAT_NCDHW);
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingGetDescInfo_HF32Enabled_FP32Inputs)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape bias = {{7}, {7}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&fmap, &weight, &bias})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW",
                                                  0,
                                                  "SPECIFIC",
                                                  true))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.flagInfo_.hasBias = true;

    base.GetDescInfo();

    EXPECT_EQ(base.descInfo_.fMapDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.descInfo_.weightDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.descInfo_.outDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.descInfo_.biasDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.attrInfo_.hf32Mode, 1);  // ENABLE_HF32
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingGetDescInfo_HF32Disabled_FP32Inputs)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW",
                                                  0,
                                                  "SPECIFIC",
                                                  false))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());

    base.GetDescInfo();

    EXPECT_EQ(base.descInfo_.fMapDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.descInfo_.weightDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.descInfo_.outDtype, ge::DT_FLOAT);
    EXPECT_EQ(base.attrInfo_.hf32Mode, 0);  // DISABLE_HF32
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingGetDescInfo_HF32NotApplicable_NonFP32Inputs)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW",
                                                  0,
                                                  "SPECIFIC",
                                                  false))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());

    base.GetDescInfo();

    EXPECT_EQ(base.descInfo_.fMapDtype, ge::DT_BF16);
    EXPECT_EQ(base.descInfo_.weightDtype, ge::DT_BF16);
    EXPECT_EQ(base.descInfo_.outDtype, ge::DT_BF16);
    EXPECT_EQ(base.attrInfo_.hf32Mode, 0);  // Not enabled due to non-FP32 inputs
}

// ============================================================================
// ExtractAndPassParamsToEngine() Coverage Tests
// ============================================================================

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractAndPassParams_WithBias)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape bias = {{7}, {7}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&fmap, &weight, &bias})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW"))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.engine_ = MakeInitializedEngineForTest();
    base.InitConv3dOriginFormat();

    EXPECT_TRUE(base.ExtractAndPassParamsToEngine());
    EXPECT_TRUE(base.flagInfo_.hasBias);
    EXPECT_TRUE(base.engine_->flagInfo_.hasBias);
    EXPECT_EQ(base.engine_->biasShape_, (std::vector<int64_t>{7}));
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractAndPassParams_WithoutBias)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fmap, &weight})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW"))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.engine_ = MakeInitializedEngineForTest();
    base.InitConv3dOriginFormat();

    EXPECT_TRUE(base.ExtractAndPassParamsToEngine());
    EXPECT_FALSE(base.flagInfo_.hasBias);
    EXPECT_FALSE(base.engine_->flagInfo_.hasBias);
    EXPECT_TRUE(base.engine_->biasShape_.empty());
}

TEST_F(Conv3DV2TilingRuntime, TestBaseTilingExtractAndPassParams_WithScale)
{
    gert::StorageShape fmap = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape weight = {{7, 3, 1, 3, 3}, {7, 3, 1, 3, 3}};
    gert::StorageShape scale = {{7}, {7}};
    gert::StorageShape out = {{2, 7, 4, 5, 6}, {2, 7, 4, 5, 6}};

    auto info = CreateDefaultCompileAndPlatformInfo();

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Conv3DV2")
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&fmap, &weight, nullptr, &scale})
                      .OutputShapes({&out})
                      .CompileInfo(&info.compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&info.platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(1, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW)
                      .NodeAttrs(MakeConv3DV2Attrs({1, 1, 1, 1, 1},
                                                  {0, 0, 0, 0, 0, 0},
                                                  {1, 1, 1, 1, 1},
                                                  1,
                                                  "NCDHW"))
                      .Build();

    optiling::Conv3dOpsTiling::Conv3dBaseTiling base(holder.GetContext<gert::TilingContext>());
    base.engine_ = MakeInitializedEngineForTest();
    base.InitConv3dOriginFormat();

    EXPECT_TRUE(base.ExtractAndPassParamsToEngine());
    EXPECT_FALSE(base.flagInfo_.hasBias);
    EXPECT_TRUE(base.flagInfo_.hasScale);
    EXPECT_TRUE(base.engine_->flagInfo_.hasScale);
    EXPECT_EQ(base.engine_->scaleShape_, (std::vector<int64_t>{7}));
}

} // namespace
