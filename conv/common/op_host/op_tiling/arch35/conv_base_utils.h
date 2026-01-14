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
 * \file conv_base_utils.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_BASE_UTILS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_BASE_UTILS_H
#include "../cube_tiling.h"
#include "conv_template_utils.h"
namespace optiling {
namespace conv_ops_tiling {

enum class QuantMode : std::uint8_t {
    NO_QUANT = 0,
    SCALAR_QUANT,
    VECTOR_QUANT,
    UNDEFINED
};

enum class ReluMode : std::uint8_t {
    NORELU = 0,
    NORMALRELU = 1,
    SCALARRELU = 2,
    VECTORRELU = 3,
    UNDEFINED
};

enum class  ClipMode : std::uint8_t {
    NOCLIPRELU = 0,
    SCALARCLIPRELU = 1,
    UNDEFINED
};

struct ConvTilingParseInfo: CubeTilingCommonParseInfo {
    uint32_t aicoreNum = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    uint64_t btSize = 0;
    uint64_t l2Rate = 0;
    std::string socVersion = "";
    std::string shortSocVersion = "";
    ConvTilingParseInfo& operator=(const ConvTilingParseInfo* other) {
        if (this != other) {  // 防止自赋值
            // 复制所有的成员变量
            aicoreNum = other->aicoreNum;
            l2Size = other->l2Size;
            l1Size = other->l1Size;
            l0aSize = other->l0aSize;
            l0bSize = other->l0bSize;
            l0cSize = other->l0cSize;
            ubSize = other->ubSize;
            btSize = other->btSize;
            l2Rate = other->l2Rate;
            socVersion = other->socVersion;
            shortSocVersion = other->shortSocVersion;
        }
        return *this;
    }
};

struct ConvAscendcOriginShapeAttrInfo {
    int64_t oriFmapN = 1;
    int64_t oriFmapC = 1;
    int64_t oriFmapD = 1;
    int64_t oriFmapH = 1;
    int64_t oriFmapW = 1;
    int64_t oriWeightN = 1;
    int64_t oriWeightC = 1;
    int64_t oriWeightD = 1;
    int64_t oriWeightH = 1;
    int64_t oriWeightW = 1;
    int64_t oriOutputN = 1;
    int64_t oriOutputC = 1;
    int64_t oriOutputD = 1;
    int64_t oriOutputH = 1;
    int64_t oriOutputW = 1;
    int64_t oriOutput1N = 1;
    int64_t oriOutput1C = 1;
    int64_t oriOutput1D = 1;
    int64_t oriOutput1H = 1;
    int64_t oriOutput1W = 1;
    int64_t oriStrideN = 1;
    int64_t oriStrideC = 1;
    int64_t oriStrideD = 1;
    int64_t oriStrideH = 1;
    int64_t oriStrideW = 1;
    int64_t oriDilationN = 1;
    int64_t oriDilationC = 1;
    int64_t oriDilationD = 1;
    int64_t oriDilationH = 1;
    int64_t oriDilationW = 1;
    int64_t oriPadHead = 1;
    int64_t oriPadTail = 1;
    int64_t oriPadTop = 1;
    int64_t oriPadBottom = 1;
    int64_t oriPadLeft = 1;
    int64_t oriPadRight = 1;
    int64_t oriGroups = 1;
    int64_t oriOffsetX = 1;
};

const std::map<std::string, int8_t> STR_TO_ROUNDMODE = {
    {"rint", ROUND_MODE_RINT}, {"round", ROUND_MODE_ROUND},
    {"hybrid", ROUND_MODE_HYBRID}
};

// fmap, weight, output
const std::vector<std::vector<ge::Format>> SUPPORT_CONV2D_FORMAT_LIST = {
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW},
    {ge::Format::FORMAT_NHWC, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_NHWC}
};

const std::vector<std::vector<ge::Format>> SUPPORT_CONV2D_FORMAT_LIST_MC62CM12A = {
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW},
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_NCHW},
    {ge::Format::FORMAT_NHWC, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_NHWC}
};

const std::vector<std::vector<ge::Format>> SUPPORT_QUANT_CONV2D_FORMAT_LIST = {
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW}
};

const std::vector<std::vector<ge::Format>> SUPPORT_CONV3D_FORMAT_LIST = {
    {ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW},
    {ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_NDHWC},
    {ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NDHWC}
};

const std::vector<std::vector<ge::Format>> SUPPORT_QUANT_CONV3D_FORMAT_LIST = {
    {ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW}
};

const std::vector<std::vector<ge::Format>> SUPPORT_CONV2D_DEFAULT_FORMAT_LIST = {
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW}
};

const std::vector<std::vector<ge::Format>> SUPPORT_CONV3D_DEFAULT_FORMAT_LIST = {
    {ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW}
};

// ExtendConv2D fmap, weight, output supprot format list
const std::vector<std::vector<ge::Format>> EXTENDCONV2D_SUPPORT_FORMAT_LIST = {
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW},
    {ge::Format::FORMAT_NHWC, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_NHWC}
};

// ExtendConv2D fmap, weight, output supprot format list
const std::vector<std::vector<ge::Format>> EXTENDCONV2D_SUPPORT_FORMAT_LIST_MC62CM12A = {
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW},
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_NCHW},
    {ge::Format::FORMAT_NHWC, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_NHWC}
};

struct ConvParamInfo {
    // Fmap, Weight, Output, FmapOri(for attr) param info
    std::vector<ge::Format> paramsFormat = {ge::Format::FORMAT_MAX, ge::Format::FORMAT_MAX, ge::Format::FORMAT_MAX};
    // index: N C D H W
    std::vector<std::vector<size_t>> paramsIdxVec = {{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    const size_t FMAP_PARAM_IDX = 0;
    const size_t WEIGHT_PARAM_IDX = 1;
    const size_t OUT_PARAM_IDX = 2;
    std::string nodeType = "";
};
}
struct Conv2DTilingParseInfo: CubeTilingCommonParseInfo {
    std::string opType = "";
    // hardware info (required by binary mode)
    uint32_t aicoreNum = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    uint64_t btSize = 0;
    uint32_t ddrReadRate = 0;
    uint32_t ddrWriteRate = 0;
    uint32_t l2Rate = 0;
    uint32_t l2ReadRate = 0;
    uint32_t l2WriteRate = 0;
    uint32_t l1ToL0aRate = 0;
    uint32_t l1ToL0bRate = 0;
    uint32_t l1ToUbRate = 0;
    uint32_t l0cToUbRate = 0;
    uint32_t ubToL2Rate = 0;
    uint32_t ubToDdrRate = 0;
    uint32_t ubToL1Rate = 0;
    uint32_t cubeBandwidth = 0;
    uint32_t vectorBandwidth = 0;
    bool cubeVectorSplit = false;
    std::string socVersion = "";
    std::string shortSocVersion = "";
    // fusion utilize info
    float preFusionUbEltwise = 0;
    float postFusionUbEltwise = 0;
    float preFusionUbEltwiseNx1 = 0;
    float postFusionUbEltwiseNx1 = 0;
    float preFusionUbBroadcast = 0;
    float postFusionUbBroadcast = 0;
    float preFusionUbBroadcastNx1 = 0;
    float postFusionUbBroadcastNx1 = 0;
    float postFusionUbChannelwise = 0;
    int64_t preFusionVectorUtilize = 0;
    int64_t postFusionVectorUtilize = 0;
    std::string ubFusionPattern = "";
    // conv2d inputs info
    uint32_t conv2dInputBit = 0;
    // Check the current soc supports fixpipe or not.
    bool fixpipeFlag = false;
    // fix_tiling
    bool compile_get_tiling_flag = false;
    // quantconv2d FeatureFlag
    bool isLoad3dFlag = false;
};
}
#endif
