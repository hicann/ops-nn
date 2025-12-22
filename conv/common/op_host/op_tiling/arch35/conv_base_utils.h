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
namespace optiling {
namespace conv_ops_tiling {
constexpr uint32_t CONST_VALUE_0 = 0;
constexpr uint32_t CONST_VALUE_1 = 1;
constexpr uint32_t CONST_VALUE_2 = 2;
constexpr uint32_t CONST_VALUE_3 = 3;
constexpr uint32_t CONST_VALUE_4 = 4;
constexpr uint32_t CONST_VALUE_5 = 5;
constexpr uint32_t CONST_VALUE_6 = 6;
constexpr uint32_t CONST_VALUE_7 = 7;
constexpr uint32_t BW_COEFF = 4;
constexpr uint32_t BW_COEFF_UB = 1;
constexpr uint32_t C0_SIZE = 32;
constexpr uint32_t MIN_M_L1_SIZE = 16;
constexpr uint64_t MIN_L2_BAND_WIDTH = 128;
constexpr uint64_t BATCH_AICORE_COF = 2;
constexpr uint32_t BLOCKDIM_MSPLIT_DEC_NUM = 5;
constexpr uint32_t BLOCKDIM_MSPLIT_BATCH_IDX = 0;
constexpr uint32_t BLOCKDIM_MSPLIT_M_IDX = 1;
constexpr uint32_t BLOCKDIM_MSPLIT_N_IDX = 2;
constexpr uint32_t BLOCKDIM_MSPLIT_DO_IDX = 3;
constexpr uint32_t BLOCKDIM_MSPLIT_GROUP_IDX = 4;
constexpr uint32_t C04_CI1_SIZE = 1;
constexpr uint32_t C04_CIN_SIZE = 4;
constexpr uint32_t BLOCKDIM_HWSPLIT_DEC_NUM = 6;
constexpr uint32_t BLOCKDIM_HWSPLIT_BATCH_IDX = 0;
constexpr uint32_t BLOCKDIM_HWSPLIT_HO_IDX = 1;
constexpr uint32_t BLOCKDIM_HWSPLIT_WO_IDX = 2;
constexpr uint32_t BLOCKDIM_HWSPLIT_N_IDX = 3;
constexpr uint32_t BLOCKDIM_HWSPLIT_DO_IDX = 4;
constexpr uint32_t BLOCKDIM_HWSPLIT_GROUP_IDX = 5;

constexpr uint32_t INPUT_FMAP_INDEX = 0;
constexpr uint32_t INPUT_WEIGHT_INDEX = 1;
constexpr uint32_t INPUT_BIAS_INDEX = 2;
constexpr uint32_t INPUT_OFFSET_W_INDEX = 3;
constexpr uint32_t OUTPUT_INDEX = 0;
constexpr uint32_t INPUT_SCALE_INDEX = 2;
constexpr uint32_t QUANT_INPUT_BIAS_INDEX = 3;
constexpr size_t UINT64_BIT_COUNT = 64;
constexpr size_t UINT64_BYTE_COUNT = 8;
constexpr size_t UINT32_BIT_COUNT = 32;
constexpr size_t BITS_PER_BYTE = 8;

constexpr uint32_t ATTR_STRIDE_INDEX = 0;
constexpr uint32_t ATTR_PAD_INDEX = 1;
constexpr uint32_t ATTR_DILATION_INDEX = 2;
constexpr uint32_t ATTR_GROUP_INDEX = 3;
constexpr uint32_t ATTR_DATAFORMAT_INDEX = 4;
constexpr uint32_t ATTR_PAD_MODE_INDEX = 6;
constexpr uint32_t ATTR_ENABLE_HF32_INDEX = 7;

constexpr uint32_t ATTR_QUANT_DTYPE_INDEX = 0;
constexpr uint32_t ATTR_QUANT_STRIDE_INDEX = 1;
constexpr uint32_t ATTR_QUANT_PAD_INDEX = 2;
constexpr uint32_t ATTR_QUANT_DILATION_INDEX = 3;
constexpr uint32_t ATTR_QUANT_GROUP_INDEX = 4;
constexpr uint32_t ATTR_QUANT_DATAFORMAT_INDEX = 5;
constexpr uint32_t ATTR_QUANT_OFFSETX_INDEX = 6;
constexpr uint32_t ATTR_QUANT_ROUNDMODE_INDEX = 7;
constexpr uint32_t ATTR_QUANT_PADMODE_INDEX = 8;

constexpr uint32_t SWEET_POINT_AI_FP16 = 144;

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

struct ConvAscendcAttrInfo {
    uint32_t dilationH = 1;
    uint32_t dilationW = 1;
    uint32_t dilationD = 1;
    uint32_t strideH = 1;
    uint32_t strideW = 1;
    uint32_t strideD = 1;
    uint32_t padHead = 0;
    uint32_t padTail = 0;
    uint32_t padTop = 0;
    uint32_t padBottom = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t hf32Mode = 0;
    int32_t offsetx = 0;
    uint32_t groups = 0;
    uint32_t roundMode = 0; // 1-rint 2-halfToAway(hif8) 3-hybrid(hif8)
    uint8_t dualOutput = 0;
};

struct ConvAscendcShapesInfo {
    uint64_t batch = 1;
    uint64_t ci = 1;
    uint64_t di = 1;
    uint64_t hi = 1;
    uint64_t wi = 1;
    uint64_t co = 1;
    uint64_t kd = 1;
    uint64_t kh = 1;
    uint64_t kw = 1;
    uint64_t dout = 1;
    uint64_t ho = 1;
    uint64_t wo = 1;
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
constexpr int8_t ROUND_MODE_RINT = 1;
constexpr int8_t ROUND_MODE_ROUND  = 2;
constexpr int8_t ROUND_MODE_HYBRID = 3;

constexpr uint64_t MAX_40_BIT_NUM = 1099511627775;
constexpr uint64_t MAX_32_BIT_NUM = 4294967295;
constexpr uint64_t MAX_16_BIT_NUM = 65535;

constexpr uint64_t MAX_N_B16_SHAPE = 1000000;
constexpr uint64_t MAX_N_B8_SHAPE = 1000000;
constexpr uint64_t MAX_N_B32_SHAPE = 1000000;
constexpr uint64_t MAX_CIN_B16_SHAPE = 1000000;
constexpr uint64_t MAX_CIN_B8_SHAPE = 1000000;
constexpr uint64_t MAX_CIN_B32_SHAPE = 1000000;
constexpr uint64_t MAX_FM_H_B16_SHAPE = 1000000;
constexpr uint64_t MAX_FM_H_B8_SHAPE = 1000000;
constexpr uint64_t MAX_FM_H_B32_SHAPE = 1000000;
constexpr uint64_t MAX_FM_W_B16_SHAPE = 1000000;
constexpr uint64_t MAX_FM_W_B8_SHAPE = 1000000;
constexpr uint64_t MAX_FM_W_B32_SHAPE = 1000000;
constexpr uint64_t MAX_COUT_B16_SHAPE = 1000000;
constexpr uint64_t MAX_COUT_B8_SHAPE = 1000000;
constexpr uint64_t MAX_COUT_B32_SHAPE = 1000000;
constexpr uint64_t MAX_KH_B16_SHAPE = 1000000;
constexpr uint64_t MAX_KH_B8_SHAPE = 1000000;
constexpr uint64_t MAX_KH_B32_SHAPE = 1000000;
constexpr uint64_t MAX_KW_B16_SHAPE = 1000000;
constexpr uint64_t MAX_KW_B8_SHAPE = 1000000;
constexpr uint64_t MAX_KW_B32_SHAPE = 1000000;
constexpr uint64_t MAX_ATTRS_SHAPE = 1000000;
constexpr uint64_t MAX_OUT_SHAPE = 1000000;
constexpr uint64_t MAX_GROUP_SHAPE = 65535;

constexpr uint32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t C0_BYTE_SIZE = 32;
constexpr uint32_t TMP_BATCH_SIZE = 2;
constexpr int64_t PAD_MODE_DIV_FACTOR = 2;

constexpr uint32_t CUBE_UNIT = 16;
constexpr uint32_t FP16_CUBE_UNIT = 16;
constexpr uint32_t FP32_CUBE_UNIT = 8;
constexpr uint32_t INT8_CUBE_UNIT = 32;
constexpr uint32_t MKN_MAX_SIZE = 3;
constexpr uint32_t MKN_M_INDEX = 0;
constexpr uint32_t MKN_K_INDEX = 1;
constexpr uint32_t MKN_N_INDEX = 2;
constexpr uint32_t MKN_VALUE_DEFAULT = 16;

constexpr uint64_t L1_SIZE = 524288;
constexpr uint64_t L0C_SIZE = 262144;
constexpr uint64_t FB_SIZE_910_55 = 6144;
constexpr uint64_t FB_SIZE_910_95 = 4096;
constexpr uint64_t FB_SIZE_910B = 2048;
constexpr uint64_t BT_SIZE_910_55 = 4096;
constexpr uint64_t BT_SIZE_910_95 = 4096;
constexpr uint64_t BT_SIZE_910B = 1024;
constexpr uint64_t BIAS_TABLE_SIZE = 4096;
constexpr uint64_t OPT_GROUP_RESERVED_SIZE = 256;
constexpr uint64_t WEIGHT_UB_NUMS = 2;
constexpr uint32_t MKN_M_IDX = 0;
constexpr uint32_t MKN_K_IDX = 1;
constexpr uint32_t MKN_N_IDX = 2;

constexpr int32_t OFFSET_X_MAX_VALUE = 127;
constexpr int32_t OFFSET_X_MIN_VALUE = -128;
constexpr int64_t QUANT_DTYPE_VALUE = 0; // INT32-->FP16

constexpr int64_t L0A_L0B_PB_FLAG_MASK = 0x03;
constexpr int64_t L1A_L1B_PB_FLAG_MASK = 0x18;
constexpr uint64_t L1_PB_OFFSET = 3;
constexpr uint64_t WEIGHT_L1_PB_OFFSET = 4;
constexpr uint64_t TILINGkEY_OFFSET_UINT = 10;
constexpr uint64_t FULLLOAD_AL1 = 0;
constexpr uint64_t ONLY_M_FULLLOAD_AL1_AL0 = 1;
constexpr uint64_t FMP_OTHER = 2;
constexpr uint64_t FULLLOAD_BL1 = 0;
constexpr uint64_t ONLY_N_FULLLOAD_BL1_BL0 = 1;
constexpr uint64_t WEIGHT_OTHER = 2;
constexpr uint64_t L1_PB_ALL_CLOSE = 0;
constexpr uint64_t L1_PB_BL1_OPEN = 2;
constexpr uint64_t L1_PB_ALL_OPEN = 3;
// load3dv2 win max value
constexpr uint64_t LOAD3DV2_WIN_LIMIT_VALUE = 32767;
constexpr uint64_t WEIGHT_UB_TRANS_CLOSE = 0;
constexpr uint64_t WEIGHT_UB_TRANS_OPEN = 1;
constexpr uint64_t FMAP_LOAD3D_MODE = 0;
constexpr uint64_t FMAP_DMA_MODE = 1;

constexpr uint64_t CONV_INNER_BATCH_SINGLE = 0;
constexpr uint64_t CONV_INNER_BATCH_KERNEL_1X1_MULTI = 1;
constexpr uint64_t CONV_INNER_BATCH_MULTI = 2;

constexpr uint32_t FP16_DTYPE_SIZE = 2;
constexpr uint32_t INT64_DTYPE_SIZE_COMPARE_FP16 = 4;

// idxList
constexpr size_t IDX_LIST_N_IDX = 0;
constexpr size_t IDX_LIST_C_IDX = 1;
constexpr size_t IDX_LIST_D_IDX = 2;
constexpr size_t IDX_LIST_H_IDX = 3;
constexpr size_t IDX_LIST_W_IDX = 4;
constexpr size_t IDX_LIST_END_IDX = 5;
constexpr size_t MAX_STR_LEN = 1024;

const std::map<std::string, int8_t> STR_TO_ROUNDMODE = {
    {"rint", ROUND_MODE_RINT}, {"round", ROUND_MODE_ROUND},
    {"hybrid", ROUND_MODE_HYBRID}
};

// fmap, weight, output
const std::vector<std::vector<ge::Format>> SUPPORT_CONV2D_FORMAT_LIST = {
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW},
    {ge::Format::FORMAT_NHWC, ge::Format::FORMAT_HWCN, ge::Format::FORMAT_NHWC}
};

const std::vector<std::vector<ge::Format>> SUPPORT_QUANT_CONV2D_FORMAT_LIST = {
    {ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW, ge::Format::FORMAT_NCHW}
};

const std::vector<std::vector<ge::Format>> SUPPORT_CONV3D_FORMAT_LIST = {
    {ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW, ge::Format::FORMAT_NCDHW},
    {ge::Format::FORMAT_NDHWC, ge::Format::FORMAT_DHWCN, ge::Format::FORMAT_NDHWC}
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
