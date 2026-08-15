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
 * \file test_conv_base_utils_tiling.cpp
 * \brief UT for conv_base_utils.h and cube_tiling.h
 */

#include <gtest/gtest.h>
#include <map>
#include <string>
#include "conv/common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base.h"
#include "conv/common/op_host/op_tiling/cube_tiling.h"

using namespace optiling;
using namespace optiling::conv_ops_tiling;

TEST(ConvBaseUtilsTest, CubeTilingCommonParseInfoDefaults)
{
    CubeTilingCommonParseInfo info;
    EXPECT_EQ(info.fmapC1, 0);
    EXPECT_FALSE(info.correctRangeFlag);
    EXPECT_TRUE(info.tilingType.empty());
}

TEST(ConvBaseUtilsTest, InputShapeInfoConstruction)
{
    std::vector<int64_t> shape = {1, 16, 32, 32};
    InputShapeInfo info(shape, "NCHW");
    EXPECT_EQ(info.xShape, shape);
    EXPECT_EQ(info.xFormat, "NCHW");
}

TEST(ConvBaseUtilsTest, ConvTilingParseInfoDefaults)
{
    ConvTilingParseInfo info;
    EXPECT_EQ(info.aicoreNum, 0u);
    EXPECT_EQ(info.l2Size, 0u);
    EXPECT_EQ(info.l1Size, 0u);
    EXPECT_EQ(info.l0aSize, 0u);
    EXPECT_EQ(info.l0bSize, 0u);
    EXPECT_EQ(info.l0cSize, 0u);
    EXPECT_EQ(info.ubSize, 0u);
    EXPECT_EQ(info.btSize, 0u);
    EXPECT_EQ(info.l2Rate, 0u);
    EXPECT_TRUE(info.socVersion.empty());
    EXPECT_TRUE(info.shortSocVersion.empty());
    EXPECT_EQ(info.npuArch, NpuArch::DAV_RESV);
    EXPECT_EQ(info.aivNum, 0u);
    EXPECT_EQ(info.fbSize, 0u);
    EXPECT_FALSE(info.isCubeVectorFuse);
}

TEST(ConvBaseUtilsTest, ConvTilingParseInfoOperatorAssign)
{
    ConvTilingParseInfo src;
    src.aicoreNum = 32;
    src.l1Size = 524288;
    src.socVersion = "ascend910b";
    src.npuArch = NpuArch::DAV_3510;
    src.isCubeVectorFuse = true;
    ConvTilingParseInfo dst;
    dst.operator=(&src);
    EXPECT_EQ(dst.aicoreNum, 32u);
    EXPECT_EQ(dst.l1Size, 524288u);
    EXPECT_EQ(dst.socVersion, "ascend910b");
    EXPECT_EQ(dst.npuArch, NpuArch::DAV_3510);
    EXPECT_TRUE(dst.isCubeVectorFuse);
}

TEST(ConvBaseUtilsTest, ConvAscendcOriginShapeAttrInfoDefaults)
{
    ConvAscendcOriginShapeAttrInfo info;
    EXPECT_EQ(info.oriFmapN, 1);
    EXPECT_EQ(info.oriFmapC, 1);
    EXPECT_EQ(info.oriWeightN, 1);
    EXPECT_EQ(info.oriOutputN, 1);
    EXPECT_EQ(info.oriGroups, 1);
    EXPECT_EQ(info.fixedShiftValue, 0);
}

TEST(ConvBaseUtilsTest, ConvParamInfoDefaults)
{
    ConvParamInfo info;
    EXPECT_EQ(info.paramsFormat.size(), 3u);
    EXPECT_EQ(info.paramsFormat[0], ge::Format::FORMAT_MAX);
    EXPECT_EQ(info.FMAP_PARAM_IDX, 0u);
    EXPECT_EQ(info.WEIGHT_PARAM_IDX, 1u);
    EXPECT_EQ(info.OUT_PARAM_IDX, 2u);
    EXPECT_TRUE(info.nodeType.empty());
}

TEST(ConvBaseUtilsTest, QuantModeEnumValues)
{
    EXPECT_EQ(static_cast<uint8_t>(QuantMode::NO_QUANT), 0);
    EXPECT_EQ(static_cast<uint8_t>(QuantMode::SCALAR_QUANT), 1);
    EXPECT_EQ(static_cast<uint8_t>(QuantMode::VECTOR_QUANT), 2);
    EXPECT_EQ(static_cast<uint8_t>(QuantMode::UNDEFINED), 3);
}

TEST(ConvBaseUtilsTest, ReluModeEnumValues)
{
    EXPECT_EQ(static_cast<uint8_t>(ReluMode::NORELU), 0);
    EXPECT_EQ(static_cast<uint8_t>(ReluMode::NORMALRELU), 1);
    EXPECT_EQ(static_cast<uint8_t>(ReluMode::SCALARRELU), 2);
    EXPECT_EQ(static_cast<uint8_t>(ReluMode::VECTORRELU), 3);
    EXPECT_EQ(static_cast<uint8_t>(ReluMode::UNDEFINED), 4);
}

TEST(ConvBaseUtilsTest, ClipModeEnumValues)
{
    EXPECT_EQ(static_cast<uint8_t>(ClipMode::NOCLIPRELU), 0);
    EXPECT_EQ(static_cast<uint8_t>(ClipMode::SCALARCLIPRELU), 1);
    EXPECT_EQ(static_cast<uint8_t>(ClipMode::UNDEFINED), 2);
}

TEST(ConvBaseUtilsTest, RoundModeMap)
{
    EXPECT_EQ(STR_TO_ROUNDMODE.at("rint"), ROUND_MODE_RINT);
    EXPECT_EQ(STR_TO_ROUNDMODE.at("round"), ROUND_MODE_ROUND);
    EXPECT_EQ(STR_TO_ROUNDMODE.at("hybrid"), ROUND_MODE_HYBRID);
    EXPECT_EQ(STR_TO_ROUNDMODE.size(), 3u);
}

TEST(ConvBaseUtilsTest, SupportConv2dFormatListNotEmpty)
{
    EXPECT_FALSE(SUPPORT_CONV2D_FORMAT_LIST.empty());
    EXPECT_EQ(SUPPORT_CONV2D_FORMAT_LIST.size(), 2u);
}

TEST(ConvBaseUtilsTest, SupportConv2dFormatListFuseNotEmpty)
{
    EXPECT_FALSE(SUPPORT_CONV2D_FORMAT_LIST_FUSE.empty());
    EXPECT_EQ(SUPPORT_CONV2D_FORMAT_LIST_FUSE.size(), 8u);
}

TEST(ConvBaseUtilsTest, SupportConv3dFormatListNotEmpty)
{
    EXPECT_FALSE(SUPPORT_CONV3D_FORMAT_LIST.empty());
    EXPECT_EQ(SUPPORT_CONV3D_FORMAT_LIST.size(), 3u);
}

TEST(ConvBaseUtilsTest, SupportQuantConvFormatListNotEmpty)
{
    EXPECT_FALSE(SUPPORT_QUANT_CONV2D_FORMAT_LIST.empty());
    EXPECT_FALSE(SUPPORT_QUANT_CONV3D_FORMAT_LIST.empty());
}

TEST(ConvBaseUtilsTest, ExtendConv2dFormatListNotEmpty)
{
    EXPECT_FALSE(EXTENDCONV2D_SUPPORT_FORMAT_LIST.empty());
    EXPECT_FALSE(EXTENDCONV2D_SUPPORT_FORMAT_LIST_FUSE.empty());
}

// ============================================================================
// IsCubeVectorFuseSoc: check whether cube_vector_combine == "fuse"
// ============================================================================
TEST(ConvBaseUtilsTest, IsCubeVectorFuseSocFuseReturnsTrue)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socInfos = {{"cube_vector_combine", "fuse"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    EXPECT_TRUE(IsCubeVectorFuseSoc(platformInfo));
}

TEST(ConvBaseUtilsTest, IsCubeVectorFuseSocSplitReturnsFalse)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socInfos = {{"cube_vector_combine", "split"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    EXPECT_FALSE(IsCubeVectorFuseSoc(platformInfo));
}

TEST(ConvBaseUtilsTest, IsCubeVectorFuseSocEmptyValueReturnsFalse)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socInfos = {{"cube_vector_combine", ""}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    EXPECT_FALSE(IsCubeVectorFuseSoc(platformInfo));
}

TEST(ConvBaseUtilsTest, IsCubeVectorFuseSocMissingKeyReturnsFalse)
{
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socInfos = {{"ai_core_cnt", "32"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    EXPECT_FALSE(IsCubeVectorFuseSoc(platformInfo));
}

// ============================================================================
// ConvTilingParseInfo::isCubeVectorFuse field default and operator=
// ============================================================================
TEST(ConvBaseUtilsTest, ConvTilingParseInfoIsCubeVectorFuseDefault)
{
    ConvTilingParseInfo info;
    EXPECT_FALSE(info.isCubeVectorFuse);
}

TEST(ConvBaseUtilsTest, ConvTilingParseInfoIsCubeVectorFuseOperatorAssign)
{
    ConvTilingParseInfo src;
    src.isCubeVectorFuse = true;
    src.npuArch = NpuArch::DAV_3510;
    ConvTilingParseInfo dst;
    dst.operator=(&src);
    EXPECT_TRUE(dst.isCubeVectorFuse);
    EXPECT_EQ(dst.npuArch, NpuArch::DAV_3510);

    src.isCubeVectorFuse = false;
    dst.operator=(&src);
    EXPECT_FALSE(dst.isCubeVectorFuse);
}

// ============================================================================
// conv_tiling::PlatformInfo::isCubeVectorFuse field default
// ============================================================================
TEST(ConvBaseUtilsTest, PlatformInfoIsCubeVectorFuseDefault)
{
    conv_tiling::PlatformInfo info;
    EXPECT_FALSE(info.isCubeVectorFuse);
    EXPECT_EQ(info.npuArch, NpuArch::DAV_RESV);
}

// ============================================================================
// GetSupportedDataTypes: arch key selects FUSE vs DAV type lists
// ============================================================================
TEST(ConvBaseUtilsTest, GetSupportedDataTypesFuseNonQuantNonExtend)
{
    std::vector<std::vector<ge::DataType>> result;
    GetSupportedDataTypes("FUSE", false, ge::FORMAT_NCHW, false, result);
    EXPECT_EQ(result, CONV_SUPPORTED_TYPES_FUSE);
}

TEST(ConvBaseUtilsTest, GetSupportedDataTypesDavNonQuantNonExtend)
{
    std::vector<std::vector<ge::DataType>> result;
    GetSupportedDataTypes("3510", false, ge::FORMAT_NCHW, false, result);
    EXPECT_EQ(result, CONV_SUPPORTED_TYPES_DAV);
}

TEST(ConvBaseUtilsTest, GetSupportedDataTypesFuseExtendNchw)
{
    std::vector<std::vector<ge::DataType>> result;
    GetSupportedDataTypes("FUSE", false, ge::FORMAT_NCHW, true, result);
    EXPECT_EQ(result, EXTENDCONV2D_SUPPORTED_TYPES_FUSE);
}

TEST(ConvBaseUtilsTest, GetSupportedDataTypesDavExtendNchw)
{
    std::vector<std::vector<ge::DataType>> result;
    GetSupportedDataTypes("3510", false, ge::FORMAT_NCHW, true, result);
    EXPECT_EQ(result, EXTENDCONV_SUPPORTED_TYPES_NCHW);
}

TEST(ConvBaseUtilsTest, GetSupportedDataTypesFuseExtendNhwc)
{
    std::vector<std::vector<ge::DataType>> result;
    GetSupportedDataTypes("FUSE", false, ge::FORMAT_NHWC, true, result);
    EXPECT_EQ(result, EXTENDCONV2D_SUPPORTED_TYPES_FUSE);
}

TEST(ConvBaseUtilsTest, GetSupportedDataTypesDavExtendNhwc)
{
    std::vector<std::vector<ge::DataType>> result;
    GetSupportedDataTypes("3510", false, ge::FORMAT_NHWC, true, result);
    EXPECT_EQ(result, EXTENDCONV_SUPPORTED_TYPES_NHWC);
}

TEST(ConvBaseUtilsTest, GetSupportedDataTypesQuantReturnsQuantTypesRegardlessArch)
{
    std::vector<std::vector<ge::DataType>> resultFuse;
    GetSupportedDataTypes("FUSE", true, ge::FORMAT_NCHW, false, resultFuse);
    EXPECT_EQ(resultFuse, QUANTCONV_SUPPORTED_TYPES);

    std::vector<std::vector<ge::DataType>> resultDav;
    GetSupportedDataTypes("3510", true, ge::FORMAT_NCHW, false, resultDav);
    EXPECT_EQ(resultDav, QUANTCONV_SUPPORTED_TYPES);
}
