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
 * \file conv_template_utils.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_TEMPLATE_UTILS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV_TEMPLATE_UTILS_H

#include <vector>
#include <string>
#include "graph/operator_reg.h"

namespace optiling {
namespace conv_ops_tiling {

enum class ConvFormat {
    ND = 0,
    NCHW,
    NHWC,
    HWCN,
    DHWNC,
    DHWCN,
    NDHWC,
    NCDHW,
    NC1HWC0,
    NDC1HWC0,
    FRACTAL_Z_C04,
    FRACTAL_Z_3D,
    FRACTAL_Z,
    UNDEFINED
};

enum class ConvGroupType {
    NORMAL_CONV = 0,
    ORI_GROUP_CONV,
    OPT_GROUP_CONV
};

enum class ConvDtype {
    FLOAT16 = 0,
    FLOAT32,
    BFLOAT16,
    INT4,
    INT8,
    INT16,
    INT32,
    INT64,
    UINT64,
    UINT8,
    UINT16,
    UINT32,
    HIFLOAT8,
    FLOAT8_E4M3FN,
    DOUBLE,
    UNDEFINED
};

struct ConvOriGroupInfo {
    uint64_t ciPerGroup = 0;
    uint64_t coPerGroup = 0;
    uint64_t groups = 0;
    ConvDtype weightDtype = ConvDtype::UNDEFINED;
};

struct ConvOptGroupInfo {
    uint64_t enlarge = 0;
    uint64_t groupOpt = 0;
    uint64_t cinOpt = 0;
    uint64_t coutOpt = 0;
};

struct FixpipeInfo {
    uint8_t quantMode0 = 0;
    uint8_t reluMode0 = 0;
    uint8_t clipMode0 = 0;
    uint8_t quantMode1 = 0;
    uint8_t reluMode1 = 0;
    uint8_t clipMode1 = 0;
    uint8_t dualOutput = 0;
    float channelWiseCoeff = 0;
// channelWiseCoeff represents the sum of the multiples of the byte width of the data type of the channelwise
// parameter in fixpipe compared to the byte width of FP16 (two bytes).
};

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
constexpr uint32_t BW_COEFF_C04 = 10;
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

constexpr uint32_t SWEET_POINT_AI_FP16 = 144;

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
constexpr uint64_t FB_SIZE_950 = 4096;
constexpr uint64_t FB_SIZE_910B = 2048;
constexpr uint64_t FB_SIZE_MC62CM12A = 6144;
constexpr uint64_t BT_SIZE_910_55 = 4096;
constexpr uint64_t BT_SIZE_950 = 4096;
constexpr uint64_t BT_SIZE_910B = 1024;
constexpr uint64_t BT_SIZE_MC62CM12A = 4096;
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

static std::map<ge::DataType, ConvDtype> dtypeMap = {
    {ge::DT_FLOAT16, ConvDtype::FLOAT16}, {ge::DT_FLOAT, ConvDtype::FLOAT32},
    {ge::DT_BF16, ConvDtype::BFLOAT16}, {ge::DT_INT8, ConvDtype::INT8},
    {ge::DT_INT64, ConvDtype::INT64}, {ge::DT_UINT64, ConvDtype::UINT64},
    {ge::DT_INT32, ConvDtype::INT32}, {ge::DT_HIFLOAT8, ConvDtype::HIFLOAT8},
    {ge::DT_FLOAT8_E4M3FN, ConvDtype::FLOAT8_E4M3FN},
    {ge::DT_INT16, ConvDtype::INT16},
    {ge::DT_UINT8, ConvDtype::UINT8},
    {ge::DT_UINT16, ConvDtype::UINT16},
    {ge::DT_UINT32, ConvDtype::UINT32},
    {ge::DT_DOUBLE, ConvDtype::DOUBLE},
    {ge::DT_INT4, ConvDtype::INT4},
    {ge::DT_UNDEFINED, ConvDtype::UNDEFINED}
};

struct AscendOpsMknMap {
    int32_t getByIndex(uint32_t idx) const {
        if (idx > MKN_MAX_SIZE - 1) {
            return MKN_VALUE_DEFAULT;
        }
        return map[idx];
    }
    int8_t map[MKN_MAX_SIZE];
};

struct ascendOpsCubeMkn {
    int8_t toIdx[static_cast<uint8_t>(ConvDtype::UNDEFINED) + 1] = {0};
    AscendOpsMknMap maps[3] = {{CUBE_UNIT, FP16_CUBE_UNIT, CUBE_UNIT}, // fp16
                               {CUBE_UNIT, FP32_CUBE_UNIT, CUBE_UNIT}, // fp32
                               {CUBE_UNIT, INT8_CUBE_UNIT, CUBE_UNIT}}; // int8 hif8 fp8
    uint32_t GetMKN(ConvDtype dType, uint32_t idx) const {
        return maps[toIdx[static_cast<uint8_t>(dType)]].getByIndex(idx);
    }
    ascendOpsCubeMkn()
    {
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT16)] = 0;
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT32)] = 1;
        toIdx[static_cast<uint8_t>(ConvDtype::BFLOAT16)] = 0;
        toIdx[static_cast<uint8_t>(ConvDtype::INT8)] = CONST_VALUE_2;
        toIdx[static_cast<uint8_t>(ConvDtype::HIFLOAT8)] = CONST_VALUE_2;
        toIdx[static_cast<uint8_t>(ConvDtype::FLOAT8_E4M3FN)] = CONST_VALUE_2;
    }
};

const ascendOpsCubeMkn CUBE_MKN_MAP = ascendOpsCubeMkn();

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

struct ConvAscendcNodeInfo{
    std::string nodeName = "";
    std::string nodeType = "";
};

struct ConvAscendcPlatformInfo{
    uint32_t aicoreNum = 0;
    uint64_t l1Size = 0;
    uint64_t l0aSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    uint64_t btSize = 0;
    uint64_t l2Rate = 0;
    std::string socVersion = "";
};

struct ConvAscendcTilingFlag {
    bool hasBias = false;
    bool hasScale = false;
    bool quantFlag = false; // hif8, fp8, int8 with scale is true; hif8 without scale is false
    bool extendConvFlag = false;
    bool enableC04Flag = false;
    bool mSplitModeFlag = false;
    ConvGroupType convGroupType = ConvGroupType::NORMAL_CONV;
    bool mBasicBlockFlag = false;
    bool useTilingRepo = false;
    bool useTilingCache = false;
    bool isConv3dDequant = false;
    ge::Format scaleFormat = ge::FORMAT_ND;
};

struct ConvAscendcDescInfo {
    ge::DataType weightDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType fMapDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType biasDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType outDtype = ge::DataType::DT_UNDEFINED;
    ge::DataType out1Dtype = ge::DataType::DT_UNDEFINED;
    ge::DataType scaleDtype = ge::DataType::DT_INT64;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format fMapFormat = ge::FORMAT_NCHW;
    ge::Format biasFormat = ge::FORMAT_ND;
    ge::Format outFormat = ge::FORMAT_NCHW;
    ge::Format out1Format = ge::FORMAT_NCHW;
    ge::Format scaleFormat = ge::FORMAT_ND;
};

struct ConvOpsConstParams {
    uint64_t m0 = 0;
    uint64_t n0 = 0;
    uint64_t k0 = 0;
    uint64_t ci1 = 0;
    uint64_t co1 = 0;
};

struct BlockDimRes {
    uint32_t batchDim = 0;
    uint32_t mDim = 0;
    uint32_t nDim = 0;
    uint32_t doDim = 0;
    uint32_t hoDim = 0;
    uint32_t woDim = 0;
    uint32_t groupDim = 0;
    uint64_t minCost = 0;
};

struct BlockDimRange {
    std::vector<uint32_t> aicNumRange;
    std::vector<uint32_t> batchRange;
    std::vector<uint32_t> mRange;
    std::vector<uint32_t> nRange;
    std::vector<uint32_t> doRange;
    std::vector<uint32_t> hoRange;
    std::vector<uint32_t> hoSpareRange;
    std::vector<uint32_t> woRange;
    std::vector<uint32_t> groupRange;
};

enum class ConvAscendcFeatureFlag : uint8_t {
    IS_LOAD3D_FLAG = 0,
    IS_CONV1D_FLAG,
    IS_DMA_FLAG,
    INVALID
};

struct ConvAscendcTilingInfo {
    ConvAscendcShapesInfo shapeInfo;
    ConvAscendcAttrInfo attrInfo;
    ConvAscendcDescInfo descInfo;
    ConvAscendcTilingFlag flagInfo;
    ConvAscendcPlatformInfo platformInfo;
    ConvAscendcNodeInfo nodeInfo;
    BlockDimRes blockDimRes;
    ConvOpsConstParams convOpsConstParams;
};

struct ConvTilingKeyPara {
    uint64_t fmpTiling = 0;
    uint64_t weightTiling = 0;
    uint64_t l1PingPong = 0;
    uint64_t l0PingPong = 0;
    uint64_t outputOrder = 0;
    uint64_t iterOrder = 0;
    uint64_t groupType = 0;
    uint64_t enableSmallChannel = 0;
    uint64_t weightUbTrans = 0;
    uint64_t fmapCppyMode = 0;
    uint64_t innerBatch = 0;
};

static std::map<ge::DataType, uint32_t> dtypeSizeTab = {
    {ge::DataType::DT_FLOAT16, 2},
    {ge::DataType::DT_FLOAT, 4},
    {ge::DataType::DT_BF16, 2},
    {ge::DataType::DT_INT8, 1},
    {ge::DataType::DT_INT64, 8},
    {ge::DataType::DT_UINT64, 8},
    {ge::DataType::DT_INT32, 4},
    {ge::DataType::DT_HIFLOAT8, 1},
    {ge::DataType::DT_FLOAT8_E4M3FN, 1},
    {ge::DataType::DT_INT16, 2},
    {ge::DataType::DT_UINT8, 1},
    {ge::DataType::DT_UINT16, 2},
    {ge::DataType::DT_UINT32, 4},
    {ge::DataType::DT_UNDEFINED, 0}
};
}
}
#endif
