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
 * \file test_conv3d_v2_tiling_runtime2.cpp
 * \brief
 */

#include <iostream>
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

namespace {

const char *const DEFAULT_ATTR_VAL = "NONE";

struct Conv3DV2TilingTestParam {
    std::string caseName;
    std::string opType;
    std::string compileInfo;

    // input of op tiling
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

    // output of op tiling
    uint32_t blockDim = 0;
    uint64_t tilingKey = 0;
    int32_t aicoreNum = 0;
};

class Conv3DV2TilingRuntime : public testing::TestWithParam<Conv3DV2TilingTestParam> {};

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
    conv3dBT.BlockDimDecision();
    optiling::Conv3dOpsTiling::BlockDimRes res = {1, 1, 1, 1, 1, 1};
    ASSERT_EQ(conv3dBT.blockDimRes.batchDim, res.batchDim);
    ASSERT_EQ(conv3dBT.blockDimRes.mDim, res.mDim);
    ASSERT_EQ(conv3dBT.blockDimRes.nDim, res.nDim);
    ASSERT_EQ(conv3dBT.blockDimRes.doDim, res.doDim);
    ASSERT_EQ(conv3dBT.blockDimRes.groupDim, res.groupDim);
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
    conv3dBT.BlockDimDecision();
    optiling::Conv3dOpsTiling::BlockDimRes res = {aicoreNum, 1, 1, 1, 1, 1};
    ASSERT_EQ(conv3dBT.blockDimRes.batchDim, res.batchDim);
    ASSERT_EQ(conv3dBT.blockDimRes.mDim, res.mDim);
    ASSERT_EQ(conv3dBT.blockDimRes.nDim, res.nDim);
    ASSERT_EQ(conv3dBT.blockDimRes.doDim, res.doDim);
    ASSERT_EQ(conv3dBT.blockDimRes.groupDim, res.groupDim);
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
    conv3dBT.BlockDimDecision();

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
    conv3dBT.BlockDimDecision();
    optiling::Conv3dOpsTiling::BlockDimRes res = {1, 1, aicoreNum, 1, 1, 1};
    ASSERT_EQ(conv3dBT.blockDimRes.batchDim, res.batchDim);
    ASSERT_EQ(conv3dBT.blockDimRes.mDim, res.mDim);
    ASSERT_EQ(conv3dBT.blockDimRes.nDim, res.nDim);
    ASSERT_EQ(conv3dBT.blockDimRes.doDim, res.doDim);
    ASSERT_EQ(conv3dBT.blockDimRes.groupDim, res.groupDim);
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
    conv3dBT.BlockDimDecision();
    optiling::Conv3dOpsTiling::BlockDimRes res = {1, aicoreNum, 1, 1, 1, 1};
    ASSERT_EQ(conv3dBT.blockDimRes.batchDim, res.batchDim);
    ASSERT_EQ(conv3dBT.blockDimRes.mDim, res.mDim);
    ASSERT_EQ(conv3dBT.blockDimRes.nDim, res.nDim);
    ASSERT_EQ(conv3dBT.blockDimRes.doDim, res.doDim);
    ASSERT_EQ(conv3dBT.blockDimRes.groupDim, res.groupDim);
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
    conv3dBT.BlockDimDecision();
    optiling::Conv3dOpsTiling::BlockDimRes res = {1, 1, 1, aicoreNum, 1, 1};
    ASSERT_EQ(conv3dBT.blockDimRes.batchDim, res.batchDim);
    ASSERT_EQ(conv3dBT.blockDimRes.mDim, res.mDim);
    ASSERT_EQ(conv3dBT.blockDimRes.nDim, res.nDim);
    ASSERT_EQ(conv3dBT.blockDimRes.doDim, res.doDim);
    ASSERT_EQ(conv3dBT.blockDimRes.groupDim, res.groupDim);
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

    ge::graphStatus res = conv3dBT.GetGroupConvOpt();
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);
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

    conv3dBT.BlockDimDecision();
    optiling::Conv3dOpsTiling::BlockDimRes res = {1, static_cast<uint32_t>(shapeInfo.ho), 1, 1, 1, 1};
    ASSERT_EQ(conv3dBT.blockDimRes.batchDim, res.batchDim);
    ASSERT_EQ(conv3dBT.blockDimRes.mDim, res.mDim);
    ASSERT_EQ(conv3dBT.blockDimRes.nDim, res.nDim);
    ASSERT_EQ(conv3dBT.blockDimRes.doDim, res.doDim);
    ASSERT_EQ(conv3dBT.blockDimRes.groupDim, res.groupDim);
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

    ge::graphStatus ret = conv3dBT.InitOutputOrder();
    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
    // outputOrder_ should be M_MODE (0)
    ASSERT_EQ(conv3dBT.outputOrder_, 0);

    uint64_t minL1LoadSize = conv3dBT.CalcMinL1LoadSize(0);
    ASSERT_LE(minL1LoadSize, opInfo.l1Size);

    conv3dBT.blockDimRes.batchDim = 1;
    conv3dBT.blockDimRes.mDim = 4;
    conv3dBT.blockDimRes.nDim = 1;
    conv3dBT.blockDimRes.doDim = 1;
    conv3dBT.blockDimRes.groupDim = 1;
    conv3dBT.SetSingleOutputShapeByMode();
    EXPECT_EQ(conv3dBT.conv3dApiTiling_.shapeInfo.singleCo, static_cast<int64_t>(shapeInfo.cOut));
    EXPECT_EQ(conv3dBT.conv3dApiTiling_.shapeInfo.singleDo, static_cast<int64_t>(shapeInfo.dOut));
    uint64_t expectedSingleM = optiling::Conv3dOpsTiling::CeilDiv(
        static_cast<uint64_t>(shapeInfo.ho * shapeInfo.wo),
        static_cast<uint64_t>(conv3dBT.blockDimRes.mDim));
    EXPECT_EQ(conv3dBT.conv3dApiTiling_.shapeInfo.singleM, static_cast<uint64_t>(expectedSingleM));
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

    ge::graphStatus res = conv3dBT.CheckPointWise();
    EXPECT_EQ(res, ge::GRAPH_SUCCESS);
}

} // namespace
