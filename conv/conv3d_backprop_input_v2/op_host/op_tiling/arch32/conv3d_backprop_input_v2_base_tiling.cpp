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
 * \file conv3d_backprop_input_v2_base_tiling.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <util/math_util.h>
#include <platform/platform_infos_def.h>
#include <graph/utils/type_utils.h>
#include <log/log.h>
#include <register/op_impl_registry.h>
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_key.h"
#include "error_util.h"
#include "conv/common/op_host/op_tiling/math_util.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "tbe_tiling_api.h"
#include "../common/conv_backprop_input_context_utils.h"
#include "conv/conv3d_backprop_input_v2/op_kernel/conv3d_backprop_input_v2_tiling_key.h"
#include "conv/conv3d_backprop_input_v2/op_kernel/arch32/conv3d_backprop_input_v2_tiling_data.h"
#include "conv3d_backprop_input_v2_base_tiling.h"

namespace {
constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t DB_ON = 2;
constexpr uint32_t B16_BITS = 4;
constexpr uint32_t FP32_BITS = 3;
constexpr uint32_t FP32_DATA_SIZE = 4;
constexpr uint32_t F16_DATA_SIZE = 2; // BF16和FP16共用
constexpr uint32_t NUM_FP32_C1OUT = 2;
constexpr int32_t BLOCK_CUBE = 16;
constexpr int32_t CORE_NUM_910B3 = 20;
constexpr int32_t CORE_NUM_910B2 = 24;
constexpr int64_t L0_SIZE = 65536;
constexpr int64_t WORKSIZE = 16 * 1024 * 1024;
constexpr uint64_t L1_SIZE = 512 * 1024 - 128;
constexpr uint32_t BEST_BASE_M = 128;
constexpr uint32_t BEST_BASE_K = 128;
constexpr uint32_t BEST_BASE_N = 256;
constexpr int32_t FACTOR_910B3 = 5;
constexpr int32_t DIM_THRESHOLD = 4;
constexpr int32_t DIM_FACTOR = 2;
constexpr int32_t FMAP_H_NUM = 2;
constexpr int32_t BUFFER_NUM_DB = 2;
constexpr int32_t BUFFER_NUM_L1 = 4;
constexpr float CORE_USED_THRESHOLD = 0.6f;

const size_t Y_INDEX = 0;
const size_t FILTER_INDEX = 1;
const size_t OUTPUT_BP_INDEX = 2;

const int32_t PAD_DIM_LOW = 0;
const int32_t PAD_DIM_UP = 255;
const int32_t K_START_POSITION_MAX = 65535;

const std::vector<int32_t> CORE_FACTOR_910B3 = {20, 10, 5, 4, 2};
const std::vector<int32_t> CORE_FACTOR_910B2 = {24, 12, 8, 6, 4, 3, 2};

const std::map<std::string, Ops::NN::Conv::TilingValue> TILING_DATA_MAP_B2{
    {"4_4_16_64_64_6_16_66_66_3_3_3_1_1_1_0_0_0_0_0_0_1_1_1",
     {22, 1, 1, 22, 1,   1,  1,   4, 1, 198, 256, 16, 256, 16, 6, 1, 2,
      2,  1, 2, 2,  128, 48, 256, 2, 1, 1,   1,   1,  45,  3,  1, 1, 1}},

    {"1_62_16_66_66_120_16_128_128_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
     {24, 1, 1, 8, 1,   1,  3,   1, 1, 2048, 256, 16, 256, 16, 40, 1, 2,
      2,  2, 2, 2, 128, 64, 128, 1, 1, 1,    1,   1,  8,   8,  1,  1, 1}},

    {"1_122_16_130_130_240_1_256_256_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
     {24, 1, 1, 8, 1,   1,  3,  1, 1, 8192, 256, 16, 16, 1,  80, 1, 2,
      2,  2, 1, 2, 256, 64, 16, 1, 1, 1,    1,   1,  16, 32, 1,  1, 1}},

    {"4_4_86_64_64_4_16_64_64_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
     {24, 2, 1, 3, 1,   1,  4,   2, 1, 1408, 1364, 86, 256, 16, 1, 1, 2,
      2,  1, 2, 2, 256, 48, 128, 1, 1, 1,    1,    1,  4,   2,  1, 1, 1}},
};

const std::map<std::string, Ops::NN::Conv::TilingValue> TILING_DATA_MAP_B3{
    // key:
    // "N_Do_Co1_Ho_Wo_Di_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
    // _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
    // value:
    // {coreNum, batchDim, groupDim, mDim, kDim, nDim, dDim,
    // singleCoreBatch, singleCoreGroup, singleCoreM, singleCoreCout,
    // singleCoreCout1, singleCoreCin, singleCoreCin1, singleCoreDin,
    // singleCoreHo,
    // al0Pbuffer, bl0Pbuffer, cl0Pbuffer, al1Pbuffer, bl1Pbuffer, baseM, baseK,
    // baseN, baseD, baseBatch, baseGroup,
    // stepM, stepN, stepKa, stepKb, stepBatch, stepGroup, iterateOrder}
    {"1_62_16_66_66_120_16_128_128_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
     {20, 1, 1, 4, 1,   1,  5,   1, 1, 4096, 256, 16, 256, 16, 24, 1, 2,
      2,  2, 2, 2, 128, 64, 128, 1, 1, 1,    1,   1,  8,   8,  1,  1, 1}},
    {"1_122_16_130_130_240_1_256_256_4_4_4_2_2_2_3_3_3_3_3_3_1_1_1",
     {20, 1, 1, 4, 1,   1,  5,  1, 1, 16384, 256, 16, 16, 1,  48, 1, 2,
      2,  2, 1, 2, 256, 64, 16, 1, 1, 1,     1,   1,  16, 32, 1,  1, 1}},
    {"4_4_171_32_32_4_32_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
     {16, 4, 1, 2, 1,   2,  1,   1, 1, 512, 2730, 171, 256, 16, 4, 1, 2,
      2,  1, 2, 2, 256, 16, 128, 1, 1, 1,   1,    1,   4,   4,  1, 1, 1}},
    {"1_25_20_40_64_25_20_40_64_3_1_1_1_1_1_1_1_0_0_0_0_1_1_1",
     {20, 1, 1, 4, 1,   1,  5,   1, 1, 640, 320, 20, 320, 20, 5, 1, 2,
      2,  1, 2, 2, 256, 64, 128, 1, 1, 1,   1,   1,  4,   2,  1, 1, 1}},
    {"4_4_86_64_64_4_16_64_64_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
     {16, 4, 1, 1, 1,   1,  4,   1, 1, 4096, 1364, 86, 256, 16, 1, 1, 2,
      2,  1, 2, 2, 256, 48, 128, 1, 1, 1,    1,    1,  4,   2,  1, 1, 1}},
    {"8_20_86_32_32_20_16_32_32_1_1_1_1_1_1_0_0_0_0_0_0_1_1_1",
     {20, 1, 1, 1, 1,   1,  20,  8, 1, 1024, 1364, 86, 256, 16, 1, 1, 2,
      2,  1, 2, 2, 256, 48, 128, 1, 1, 1,    1,    1,  4,   4,  1, 1, 1}},
    {"16_20_8_64_64_20_4_128_128_1_3_3_1_2_2_0_0_1_1_1_1_1_1_1",
     {20, 1, 1, 1, 1,   1,  20, 16, 1, 16384, 128, 8, 64, 4,  1, 1, 2,
      2,  1, 2, 1, 336, 48, 64, 1,  1, 1,     1,   1, 6,  24, 1, 1, 1}},
};

const std::map<std::string, Ops::NN::Conv::TilingValue> FP32_TILING_DATA_MAP_B2{
    {"2_4_64_18_18_3_32_17_17_4_4_4_1_1_1_2_2_2_2_2_2_1_1_1",
     {24, 2, 1, 1, 1,   4,  3,  1, 1, 289, 512, 64, 64, 8, 1, 1, 2,
      2,  1, 2, 2, 256, 32, 64, 3, 1, 1,   1,   1,  48, 8, 1, 1, 1}},
};

const std::map<std::string, Ops::NN::Conv::TilingValue> FP32_TILING_DATA_MAP_B3{
    {"2_4_64_18_18_3_32_17_17_4_4_4_1_1_1_2_2_2_2_2_2_1_1_1",
     {12, 2, 1, 1, 1,   2,  3,   1, 1, 289, 512, 64, 128, 16, 1, 1, 2,
      2,  1, 2, 2, 256, 32, 128, 3, 1, 1,   1,   1,  48,  8,  1, 1, 1}}};

const std::map<std::string, Ops::NN::Conv::TilingValue> FP32_BASIC_BLOCK_MAP_B3{
    {"4_6_8_112_112_17_1_229_229_7_7_7_2_2_2_0_0_0_0_0_0_1_1_1",
     {20, 1, 1, 1, 1,    1, 1,  1, 1, 10534, 64, 8, 8,  1,   1, 1, 2,
      2,  1, 2, 2, 1024, 8, 16, 1, 1, 1,     1,  1, 49, 196, 1, 1, 1}},
};
const std::map<std::string, Ops::NN::Conv::TilingValue> FP32_BASIC_BLOCK_MAP_B2{
    {"4_6_8_112_112_17_1_229_229_7_7_7_2_2_2_0_0_0_0_0_0_1_1_1",
     {24, 1, 1, 1, 1,    1, 1,  1, 1, 10534, 64, 8, 8,  1,   1, 1, 2,
      2,  1, 2, 2, 1024, 8, 16, 1, 1, 1,     1,  1, 49, 196, 1, 1, 1}},
};
} // namespace

namespace Ops {
namespace NN {
namespace Conv {

static inline bool CheckRange(int32_t value, int32_t valueLow, int32_t valueUp)
{
    if (value < valueLow || value > valueUp) {
        return false;
    }
    return true;
}

static inline uint32_t GetMaxDivisor(uint32_t a, uint32_t b, uint32_t step)
{
    while (b >= step) {
        if (a % b == 0) {
            return b;
        }
        b -= step;
    }
    return 0;
}

bool Conv3DBackpropInputV2Tiling::CheckL0Size(uint32_t baseM, uint32_t baseN, uint32_t baseK, uint32_t byteSize)
{
    int64_t l0aSize = static_cast<int64_t>(baseM) * baseK * byteSize * DB_ON;
    int64_t l0bSize = static_cast<int64_t>(baseN) * baseK * byteSize * DB_ON;
    if (byteSize == FP32_DATA_SIZE && runInfo_.kernel_h * runInfo_.kernel_w > 1) {
        // fp32场景下，当HkWk>1时，L0B需要预留额外空间进行转置处理
        l0bSize = static_cast<int64_t>(baseN) * (baseK + blockSize_) * byteSize * DB_ON;
    }

    return l0aSize <= L0_SIZE && l0bSize <= L0_SIZE;
}

void Conv3DBackpropInputV2Tiling::AlignCout1(uint32_t& cout1A, uint32_t& cout1B, const bool adaptFP32) const
{
    if (cout1A == cout1B) {
        return;
    } else if (cout1B > cout1A) {
        cout1A = GetMaxDivisor(cout1B, cout1A, 1);
        return;
    }

    if (!adaptFP32) {
        cout1B = GetMaxDivisor(cout1A, cout1B, 1);
        return;
    }

    uint32_t tempCout1A = cout1A;
    while (tempCout1A % cout1B > 0) {
        tempCout1A--;
    }
    uint64_t cout1AB = static_cast<uint64_t>(tempCout1A) * cout1B;
    uint32_t step = BLOCK_CUBE / blockSize_;
    uint32_t tempCout1B = GetMaxDivisor(cout1A, cout1B, step);
    if (tempCout1B == 0) {
        cout1A = tempCout1A;
        return;
    }

    uint64_t cout1ABSmallerB = tempCout1B * cout1A;
    if (cout1ABSmallerB > cout1AB) {
        cout1B = tempCout1B;
    } else {
        cout1A = tempCout1A;
    }
}

void Conv3DBackpropInputV2Tiling::Reset()
{
    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                             0, context_->GetRawTilingData()->GetCapacity()) != EOK,
                    CUBE_INNER_ERR_REPORT(opName_, "Fail to clear tiling data"), return);
    opName_ = nullptr;
}

ge::graphStatus Conv3DBackpropInputV2Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropInputV2Tiling::GetShapeAttrsInfo()
{
    if (context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->shortSocVersion ==
            platform_ascendc::SocVersion::ASCEND910_95 ||
        context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->shortSocVersion ==
            platform_ascendc::SocVersion::ASCEND910_55) {
        return ge::GRAPH_SUCCESS;
    }
    opName_ = context_->GetNodeName();
    OP_TILING_CHECK(
        !AnalyzeDtype(), CUBE_INNER_ERR_REPORT(opName_, "fail to analyze context info"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool Conv3DBackpropInputV2Tiling::IsCapable()
{
    if (context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->shortSocVersion ==
            platform_ascendc::SocVersion::ASCEND910_95 ||
        context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->shortSocVersion ==
            platform_ascendc::SocVersion::ASCEND910_55) {
        return false;
    }
    return true;
}

ge::graphStatus Conv3DBackpropInputV2Tiling::DoOpTiling()
{
    if (!SetRunInfoToV2(context_, runInfo_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "SetRunInfoToV2 failed");
        return ge::GRAPH_FAILED;
    }
    if (!GetTbeTiling(context_, tbeTiling_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "GetTbeTiling failed");
        return ge::GRAPH_FAILED;
    }
    blockSize_ = BYTE_BLOCK / runInfo_.a_dtype_bytes;
    dtypeByte_ = runInfo_.a_dtype_bytes;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropInputV2Tiling::DoLibApiTiling()
{
    SetDxTilingFromTbeTiling();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

uint64_t Conv3DBackpropInputV2Tiling::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(loadB2Condition_, enableKernelSplit_, useBasicBlock_);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    OP_LOGD(context_->GetNodeName(), "loadB2Condition_, enableKernelSplit_, useBasicBlock_ is: [%u, %u, %u]", \
            loadB2Condition_, enableKernelSplit_, useBasicBlock_);
    return tilingKey;
}

ge::graphStatus Conv3DBackpropInputV2Tiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    // 框架预留16M
    workspaces[0] = WORKSIZE;

    // 仅在支持TBE的架构上，dx融合输出transdata时，需要分配临时GM空间
    OP_TILING_CHECK(
        context_->GetOutputDesc(Y_INDEX) == nullptr,
        CUBE_INNER_ERR_REPORT(opName_, "failed to get y tensor desc from context"), return ge::GRAPH_FAILED);
    if (context_->GetOutputDesc(Y_INDEX)->GetStorageFormat() == ge::FORMAT_NCDHW) {
        size_t singleCoreUsrSpaceSize = context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->l0c_size;
        uint32_t coreNum = context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->core_num;
        size_t usrSpaceSize = coreNum * singleCoreUsrSpaceSize;
        workspaces[0] += usrSpaceSize;
        OP_LOGD(opName_, "output storage format is FORMAT_NCDHW, usrSpaceSize = %ld", usrSpaceSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropInputV2Tiling::PostTiling()
{
    size_t tilingDataSize = sizeof(Conv3DBackpropInputV2TilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           &tilingData_, tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(
        tilingDataSize % sizeof(uint64_t) != 0,
        CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
        return ge::GRAPH_FAILED);
    context_->SetBlockDim(tilingData_.params.coreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

bool Conv3DBackpropInputV2Tiling::AnalyzeDtype() const
{
    size_t aMatrixesIndex = OUTPUT_BP_INDEX;
    size_t bMatrixesIndex = FILTER_INDEX;

    if (opType_ == optiling::OpTypeV2::kConv3DTransposeV2) {
        aMatrixesIndex = FILTER_INDEX;
        bMatrixesIndex = OUTPUT_BP_INDEX;
    }
    OP_TILING_CHECK(
        context_->GetInputDesc(aMatrixesIndex) == nullptr || context_->GetInputDesc(bMatrixesIndex) == nullptr ||
            context_->GetOutputDesc(Y_INDEX) == nullptr,
        CUBE_INNER_ERR_REPORT(opName_, "failed to get out_backprop/filter/y tensor desc from context"), return false);
    ge::DataType aDtype = context_->GetInputDesc(aMatrixesIndex)->GetDataType();
    ge::DataType bDtype = context_->GetInputDesc(bMatrixesIndex)->GetDataType();
    ge::DataType cDtype = context_->GetOutputDesc(Y_INDEX)->GetDataType();

    OP_TILING_CHECK(
        !((aDtype == ge::DT_BF16 && bDtype == ge::DT_BF16 && cDtype == ge::DT_BF16) ||
          (aDtype == ge::DT_FLOAT && bDtype == ge::DT_FLOAT && cDtype == ge::DT_FLOAT) ||
          (aDtype == ge::DT_FLOAT16 && bDtype == ge::DT_FLOAT16 && cDtype == ge::DT_FLOAT16)),
        CUBE_INNER_ERR_REPORT(
            opName_,
            "x/weight/y only support DT_BF16/DT_FLOAT16/DT_FLOAT dtype, get actual aDtype[%s] bDtype[%s] cDtype[%s]",
            ge::TypeUtils::DataTypeToAscendString(aDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(bDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(cDtype).GetString()),
        return false);
    return true;
}

// currently only supports conv3d_transpose_videogpt_f240_h256_net_ID_1 and conv3d_transpose_videogpt_f240_h256_net_ID_2
bool Conv3DBackpropInputV2Tiling::CheckKernelSplitEnable() const
{
    return (//Below are the fixed values for these two use cases.
        (runInfo_.real_g == 1 && runInfo_.batch_n == 1 && runInfo_.dedy_d == 62 && runInfo_.dedy_cout1 == 16 &&
         runInfo_.dedy_h == 66 && runInfo_.dedy_w == 66 && runInfo_.dedx_d == 120 && runInfo_.dedx_cin1 == 16 &&
         runInfo_.dedx_h == 128 && runInfo_.dedx_w == 128 && runInfo_.kernel_d == 4 && runInfo_.kernel_h == 4 &&
         runInfo_.kernel_w == 4 && runInfo_.stride_d == 2 && runInfo_.stride_h == 2 && runInfo_.stride_w == 2 &&
         runInfo_.pad_h == 3 && runInfo_.pad_t == 3 && runInfo_.pad_u == 3 && runInfo_.pad_d == 3 &&
         runInfo_.pad_l == 3 && runInfo_.pad_r == 3 && runInfo_.dilation_d == 1 && runInfo_.dilation_h == 1 &&
         runInfo_.dilation_w == 1) ||
        (runInfo_.batch_n == 1 && runInfo_.dedy_d == 122 && runInfo_.dedy_cout1 == 16 && runInfo_.dedy_h == 130 &&
         runInfo_.dedy_w == 130 && runInfo_.dedx_d == 240 && runInfo_.dedx_cin1 == 1 && runInfo_.dedx_h == 256 &&
         runInfo_.dedx_w == 256 && runInfo_.kernel_d == 4 && runInfo_.kernel_h == 4 && runInfo_.kernel_w == 4 &&
         runInfo_.stride_d == 2 && runInfo_.stride_h == 2 && runInfo_.stride_w == 2 && runInfo_.pad_h == 3 &&
         runInfo_.pad_t == 3 && runInfo_.pad_u == 3 && runInfo_.pad_d == 3 && runInfo_.pad_l == 3 &&
         runInfo_.pad_r == 3 && runInfo_.dilation_d == 1 && runInfo_.dilation_h == 1 && runInfo_.dilation_w == 1));
}

int32_t Conv3DBackpropInputV2Tiling::GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLits) const
{
    int32_t dimFactor = 1;
    for (uint32_t i = 0; i < factorLits.size(); i++) {
        if (value % factorLits[i] == 0) {
            dimFactor = factorLits[i];
            break;
        }
    }
    return dimFactor;
}

void Conv3DBackpropInputV2Tiling::GetCoreDim(
    int32_t& batchDim, int32_t& dDim, int32_t& mDim, int32_t& nDim, const int32_t curCoreNum)
{
    if (curCoreNum == 0 || curCoreNum < static_cast<int32_t>(coreNum_ * CORE_USED_THRESHOLD)) {
        return;
    }
    int64_t maxM = static_cast<int64_t>(runInfo_.dedx_h) * static_cast<int64_t>(runInfo_.dedx_w);
    int64_t maxN = static_cast<int64_t>(runInfo_.dedx_cin1);
    int64_t maxNBytes = maxN * blockSize_;
    // 分核目标：
    // (1) 保证singleCoreM和singleCoreN足够大, 可以使能128/64/256的基本块tiling;
    // (2) 数据读取的方向连续且顺序访问，提高cache复用率
    // (3) 所有核因子中间变量以Factor后缀命名，公约数和除法只操作因子返回因子，避免除0

    std::vector<int32_t> coreFactors = {};
    MathUtil::GetFactors(coreFactors, curCoreNum, curCoreNum);
    std::sort(coreFactors.rbegin(), coreFactors.rend());

    // B和D方向的最大公因子乘积是核的倍数，直接均匀分核，结束
    int32_t dMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.dedx_d), coreFactors);
    int32_t bMaxFactor = GetDimFactor(static_cast<int64_t>(runInfo_.batch_n), coreFactors);
    if ((dMaxFactor * bMaxFactor) % curCoreNum == 0) {
        dDim = dMaxFactor;
        batchDim = curCoreNum / dMaxFactor;
        return;
    }

    // B和D分不完，找B*D方向的最大公因子，尽可能在B和D多分
    int32_t batchDepthMaxFactor = GetDimFactor(static_cast<int64_t>(dMaxFactor * bMaxFactor), coreFactors);
    int32_t remainFactor = curCoreNum / batchDepthMaxFactor;

    // 剩余的在M, N方向如果能均匀分核且切块粒度不小于基本块，也结束
    int32_t mMaxFactor = GetDimFactor(maxM, coreFactors);
    int32_t nMaxFactor = GetDimFactor(maxN, coreFactors);
    if ((mMaxFactor * nMaxFactor) % remainFactor == 0) {
        // 先从M方向考虑且粒度合适，结束
        int32_t mFactor = MathUtil::GetGcd(mMaxFactor, remainFactor);
        int32_t nFactor = remainFactor / mFactor;
        if ((nFactor == 1 || maxNBytes >= (nFactor * BEST_BASE_N)) &&
            (mFactor == 1 || maxM >= (mFactor * BEST_BASE_M))) {
            dDim = MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
            batchDim = batchDepthMaxFactor / dDim;
            mDim = mFactor;
            nDim = nFactor;
            return;
        }

        // 再从N方向考虑且粒度合适，结束
        nFactor = MathUtil::GetGcd(nMaxFactor, remainFactor);
        mFactor = remainFactor / nFactor;
        if ((nFactor == 1 || maxNBytes >= nFactor * BEST_BASE_N) && (mFactor == 1 || maxM >= mFactor * BEST_BASE_M)) {
            dDim = MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
            batchDim = batchDepthMaxFactor / dDim;
            mDim = mFactor;
            nDim = nFactor;
            return;
        }

        // 这里假设M和N的范围差异是比较大的, 不再从M和N里面找其他形式的均衡
    }

    // m的粒度合适，执行M的非因子分核
    if (maxM >= (remainFactor * BEST_BASE_M)) {
        dDim = MathUtil::GetGcd(dMaxFactor, batchDepthMaxFactor);
        batchDim = batchDepthMaxFactor / dDim;
        mDim = remainFactor;
        return;
    }

    // 当前核数无法分核，尝试[coreNum_ - 1, coreNum_ * 60%]
    GetCoreDim(batchDim, dDim, mDim, nDim, curCoreNum - 1);

    // 都不满足，do nothing, 继承TBE
    return;
}

void Conv3DBackpropInputV2Tiling::SetTilingParamByDimInfo(
    TilingValue& tilingParams, const int32_t batchDim, const int32_t dDim, const int32_t mDim, const int32_t nDim)
{
    tilingParams.coreNum =
        static_cast<uint64_t>(batchDim) * tbeTiling_.group_dim * mDim * tbeTiling_.k_dim * nDim * dDim;
    tilingParams.batchDim = batchDim;
    tilingParams.groupDim = tbeTiling_.group_dim;
    tilingParams.mDim = mDim;
    tilingParams.kDim = tbeTiling_.k_dim;
    tilingParams.nDim = nDim;
    tilingParams.dDim = dDim;
    tilingParams.singleCoreBatch = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.batch_n), batchDim);
    tilingParams.singleCoreGroup = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.real_g), tbeTiling_.group_dim);
    tilingParams.singleCoreM = static_cast<uint64_t>(Ops::Base::CeilDiv(runInfo_.dedx_h, mDim)) * runInfo_.dedx_w;
    tilingParams.singleCoreCout = runInfo_.dedy_cout1_g * blockSize_;
    tilingParams.singleCoreCout1 = runInfo_.dedy_cout1_g;
    tilingParams.singleCoreCin =
        static_cast<uint64_t>(Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.dedx_cin1_g), nDim)) * blockSize_;
    tilingParams.singleCoreCin1 = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.dedx_cin1_g), nDim);
    tilingParams.singleCoreDin = Ops::Base::CeilDiv(static_cast<int32_t>(runInfo_.dedx_d), dDim);
    tilingParams.singleCoreHo = 1U;
}

void Conv3DBackpropInputV2Tiling::CalCoreDimTiling(
    TilingValue& tilingParams, const uint32_t coreNum, bool& enableTbeBlock)
{
    // 4根轴可以分核：其中batch和m支持非因子分核
    int32_t batchDim = 1;
    int32_t dDim = 1;
    int32_t mDim = 1;
    int32_t nDim = 1;

    bool enableTBEtiling = false;
    // 优先从Depth进行分核，主要考虑增加D维度分核
    GetCoreDim(batchDim, dDim, mDim, nDim, coreNum);
    // 至少可以用满60%的核
    int64_t coreNumUsed = static_cast<int64_t>(batchDim) * dDim * mDim * nDim;
    enableTBEtiling = coreNumUsed < (coreNum * CORE_USED_THRESHOLD) || coreNumUsed > coreNum || runInfo_.real_g > 1;

    if (enableTBEtiling) {
        // 采用TBE分核策略
        batchDim = tbeTiling_.batch_dim;
        dDim = tbeTiling_.d_dim;
        mDim = Ops::Base::CeilDiv(runInfo_.dedx_h, Ops::Base::CeilDiv(runInfo_.dedx_h, tbeTiling_.m_dim));
        nDim = tbeTiling_.n_dim;
        enableTbeBlock = true;
    } else {
        // 因M轴非因子切分可能导致实际使用的mDim小于原始mDim，此处需要修正
        mDim = Ops::Base::CeilDiv(runInfo_.dedx_h, Ops::Base::CeilDiv(runInfo_.dedx_h, mDim));
    }

    SetTilingParamByDimInfo(tilingParams, batchDim, dDim, mDim, nDim);
}

void Conv3DBackpropInputV2Tiling::UpdateBaseBlock(
    uint32_t& baseM, uint32_t& baseK, uint32_t& baseN, const TilingValue& tilingParams)
{
    uint64_t singleC0BaseK = static_cast<uint64_t>(runInfo_.kernel_w) * runInfo_.kernel_h * blockSize_;
    if (baseK % singleC0BaseK == 0) {
        baseK = BEST_BASE_K / dtypeByte_;
    } else if (singleC0BaseK < baseK) {
        baseK = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;
    } else if (singleC0BaseK > baseK) {
        baseK = runInfo_.kernel_w * blockSize_;
    }
    // 调换基本块tiling的M和N方向，确保singleCoreM和singleCoreN方向够用
    if (tilingParams.singleCoreM > tilingParams.singleCoreCin) {
        baseM = BEST_BASE_N;
        baseN = BEST_BASE_M;
    }
    // 超限处理
    if (baseM > tilingParams.singleCoreM) {
        baseM = Ops::Base::CeilDiv(static_cast<int32_t>(tilingParams.singleCoreM), BLOCK_CUBE) * BLOCK_CUBE;
    }
    if (baseN > tilingParams.singleCoreCin) {
        baseN = Ops::Base::CeilDiv(static_cast<int32_t>(tilingParams.singleCoreCin), BLOCK_CUBE) * BLOCK_CUBE;
    }
    uint64_t singleDepthMaxK =
        static_cast<uint64_t>(tilingParams.singleCoreCout1) * blockSize_ * runInfo_.kernel_h * runInfo_.kernel_w;
    // kernel侧应该保证下述条件的功能正确，当前在tiling侧进行约束
    if (baseK > singleDepthMaxK) {
        baseK = singleDepthMaxK;
    } else if (singleDepthMaxK % baseK != 0) {
        baseK = runInfo_.kernel_w * blockSize_;
    }
}

int32_t Conv3DBackpropInputV2Tiling::CalFmapH(const int32_t& mL1Size) const
{
    int32_t hiCal;
    int32_t divisorW = std::max(1, runInfo_.dedx_w);
    int32_t divisorM = std::max(1, mL1Size);
    if (mL1Size % divisorW == 0 || runInfo_.dedx_w % divisorM == 0) {
        hiCal = Ops::Base::CeilDiv(mL1Size, divisorW);
    } else if (mL1Size > runInfo_.dedx_w) {
        hiCal = mL1Size / divisorW + FMAP_H_NUM;
    } else {
        hiCal = FMAP_H_NUM;
    }
    int32_t khDilation = (runInfo_.kernel_h - 1) * runInfo_.dilation_h + 1;
    int32_t hoCal = (hiCal - 1) + khDilation;
    int64_t hoExpand = static_cast<int64_t>(runInfo_.dedy_h - 1) * runInfo_.stride_h + 1;
    return static_cast<int32_t>(std::min(static_cast<int64_t>(hoCal), hoExpand));
}

void Conv3DBackpropInputV2Tiling::UpdateStepFp32(uint32_t& stepKa, uint32_t& stepKb, const TilingValue& tilingParams)
{
    // 调整fp32场景先L1中缓存step参数，当前有3个场景需要约束：
    // (1)B1不开double buffer场景，此时不使能preload功能，导致尾块处理逻辑变复杂,通过tiling约束;
    // (2)Cout1*C0是8对齐但是非16对齐场景，此时尾块可能存在跨越2个C0out=16，但是只取一个C0out=16的情况;
    // (3)Hk*Wk==1场景，此时无逆序操作，LoadToB2可以连续转换;
    // 如果不完整载入C0out=16，会导致跨C0out场景，kernel逻辑调整复杂，对性能无收益
    // tiling约束载入C0out=16的倍数，可以有效防止冗余数据载入
    uint64_t lenHkWkC0out = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * BLOCK_CUBE;
    if ((tilingParams.bl1Pbuffer == 1) || (runInfo_.dedy_cout1_g * blockSize_ % BLOCK_CUBE != 0) ||
        (runInfo_.kernel_h * runInfo_.kernel_w == 1 && tilingParams.baseK < BLOCK_CUBE)) {
        stepKb = (stepKb * tilingParams.baseK + lenHkWkC0out - 1) / lenHkWkC0out * lenHkWkC0out / tilingParams.baseK;
    }
    // 调整A矩阵载入量，不影响上述stepKb的计算结果，让stepKa和stepKb满足整数倍关系
    uint64_t lenHkWkC0 = lenHkWkC0out / 2;
    if (stepKa > stepKb) {
        while (stepKa % stepKb > 0) {
            stepKa--;
        }
    } else {
        while (stepKb % stepKa > 0) {
            stepKa--;
        }
    }
    // 调整stepKa可能导致非整份载入HkWkC0
    stepKa = stepKa * tilingParams.baseK / lenHkWkC0 * lenHkWkC0 / tilingParams.baseK;
}

void Conv3DBackpropInputV2Tiling::UpdateBaseStep(uint32_t& stepKa, uint32_t& stepKb, TilingValue& tilingParams)
{
    uint32_t hoCal = CalFmapH(tilingParams.baseM); // 此处默认stepM=1
    uint64_t lenHkWkC0 = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;
    uint64_t baseNHkWkC0Size = lenHkWkC0 * tilingParams.baseN * dtypeByte_;
    uint64_t l1BSize = L1_SIZE / BUFFER_NUM_L1;
    uint64_t l1ASize = l1BSize;
    bool adaptFP32 = (dtypeByte_ == FP32_DATA_SIZE && runInfo_.kernel_h * runInfo_.kernel_w > 1);
    // fp32场景下Cout0为16，c0为8，而tiling中的Cout1是以C0对其，因此需保证加载的cout1要为2的倍数
    uint32_t fp32Cout1Factor = 2;
    uint32_t cout1B1 = std::max(static_cast<uint64_t>(1), l1BSize / baseNHkWkC0Size);
    if (adaptFP32) {
        if (cout1B1 == 1) {
            cout1B1 = fp32Cout1Factor;
            l1ASize = l1ASize - (baseNHkWkC0Size * cout1B1 - l1BSize);
        } else {
            cout1B1 = cout1B1 / fp32Cout1Factor * fp32Cout1Factor; // fp32场景下确保cout1为2的倍数
        }
    }

    uint64_t curHiWiSize = static_cast<uint64_t>(dtypeByte_) * hoCal * runInfo_.dedy_w * runInfo_.stride_w * blockSize_;
    uint32_t cout1A1 = std::max(static_cast<uint64_t>(1), l1ASize / curHiWiSize);
    if (cout1A1 >= tilingParams.singleCoreCout1) {
        cout1A1 = tilingParams.singleCoreCout1;
        tilingParams.al1Pbuffer = 1;
    }

    if (cout1B1 >= tilingParams.singleCoreCout1) {
        cout1B1 = tilingParams.singleCoreCout1;
        tilingParams.bl1Pbuffer = 1;
    }
    AlignCout1(cout1A1, cout1B1, adaptFP32);

    stepKa = std::max(
        static_cast<uint64_t>(1),
        Ops::Base::CeilDiv(
            static_cast<uint64_t>(cout1A1) * runInfo_.kernel_h * runInfo_.kernel_w * blockSize_,
            static_cast<uint64_t>(tilingParams.baseK)));
    stepKa = std::min(stepKa, K_START_POSITION_MAX / tilingParams.baseK);
    stepKb = std::max(
        static_cast<uint64_t>(1),
        Ops::Base::CeilDiv(
            static_cast<uint64_t>(cout1B1) * runInfo_.kernel_h * runInfo_.kernel_w * blockSize_,
            static_cast<uint64_t>(tilingParams.baseK)));
    if (stepKa > stepKb) {
        stepKa = Ops::Base::FloorAlign(stepKa, stepKb);
    } else {
        stepKb = Ops::Base::FloorAlign(stepKb, stepKa);
    }
    // fp32场景下，对tiling进行修改，以符合fp32场景要求
    if (dtypeByte_ == FP32_DATA_SIZE) {
        UpdateStepFp32(stepKa, stepKb, tilingParams);
    }
}

bool Conv3DBackpropInputV2Tiling::CalBaseBlockTiling(TilingValue& tilingParams)
{
    // 默认开启double buffer
    tilingParams.al1Pbuffer = DB_ON;
    tilingParams.bl1Pbuffer = DB_ON;
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.cl0Pbuffer = 1;

    // 默认采用最优基本块tiling
    // 910B最优基本块tiling
    uint32_t stepKa = 1U;
    uint32_t stepKb = 1U;
    uint32_t baseM = BEST_BASE_M;
    uint32_t baseK = BEST_BASE_K / dtypeByte_;
    uint32_t baseN = BEST_BASE_N;

    // 更新并设置基本块tiling
    UpdateBaseBlock(baseM, baseK, baseN, tilingParams);
    tilingParams.baseM = baseM;
    tilingParams.baseK = baseK;
    tilingParams.baseN = baseN;
    tilingParams.baseD = 1U;
    tilingParams.baseBatch = 1U;
    tilingParams.baseGroup = 1U;
    // 更新并设置step tiling
    UpdateBaseStep(stepKa, stepKb, tilingParams);
    tilingParams.stepKa = stepKa;
    tilingParams.stepKb = stepKb;
    tilingParams.stepM = 1U;
    tilingParams.stepN = 1U;
    tilingParams.stepBatch = 1U;
    tilingParams.stepGroup = 1U;
    tilingParams.iterateOrder = 1U;

    uint64_t b1Size = 0;
    uint64_t lenHkWkC0 = static_cast<uint64_t>(runInfo_.kernel_h) * runInfo_.kernel_w * blockSize_;
    if (dtypeByte_ == FP32_DATA_SIZE) {
        uint64_t numHkWkC0 =
            Ops::Base::CeilDiv(static_cast<uint64_t>(tilingParams.stepKb * tilingParams.baseK), lenHkWkC0);
        b1Size = tilingParams.bl1Pbuffer * dtypeByte_ * baseN *
                 Ops::Base::CeilDiv(numHkWkC0, static_cast<uint64_t>(NUM_FP32_C1OUT)) * NUM_FP32_C1OUT * lenHkWkC0;
    } else {
        b1Size = tilingParams.bl1Pbuffer * dtypeByte_ * stepKb * baseN * baseK;
    }
    uint64_t coutNum = static_cast<uint64_t>(stepKa) * baseK / (runInfo_.kernel_h * runInfo_.kernel_w);
    uint64_t a1PixelNum = static_cast<uint64_t>(CalFmapH(baseM)) * runInfo_.dedy_w * runInfo_.stride_w * coutNum;
    uint64_t l1UsedSize = a1PixelNum * dtypeByte_ * tilingParams.al1Pbuffer + b1Size;
    if (stepKa * stepKb == 0) {
        return false;
    }

    // 校验是否满足整数倍关系
    uint32_t stepParaCheck = (stepKa > stepKb) ? (stepKa % stepKb) : (stepKb % stepKa);
    // 校验是否满足整数倍HkWkC0关系
    bool stepParaCheck2 =
        ((static_cast<uint64_t>(tilingParams.stepKa * tilingParams.baseK) % lenHkWkC0 == 0) &&
         (static_cast<uint64_t>(tilingParams.stepKb * tilingParams.baseK) % lenHkWkC0 == 0));
    // 校验能否采用基本块tiling策略，否则切换到TBE tiling策略
    // 为保证A和B矩阵使能double buffer，两者加和要小于L1 size的一半
    if (CheckL0Size(baseM, baseN, baseK, dtypeByte_) && (l1UsedSize <= L1_SIZE) && stepParaCheck == 0 &&
        stepParaCheck2 == 1) {
        OP_LOGD(opName_, "Use basic-block tiling.");
        return true;
    }
    return false;
}

void Conv3DBackpropInputV2Tiling::CalTbeBlockTiling(TilingValue& tilingParams)
{
    tilingParams.al0Pbuffer = DB_ON;
    tilingParams.bl0Pbuffer = DB_ON;
    tilingParams.cl0Pbuffer = static_cast<uint32_t>(tbeTiling_.db_l0c);
    uint32_t baseM = tbeTiling_.m_l0 * BLOCK_CUBE;
    uint32_t baseK = tbeTiling_.k_l0 * blockSize_;
    uint32_t baseN = tbeTiling_.n_l0 * BLOCK_CUBE;
    uint32_t tmpBaseKMax = std::max(runInfo_.kernel_h * blockSize_, runInfo_.kernel_w * blockSize_);
    uint32_t tmpBaseKMin = std::min(runInfo_.kernel_h * blockSize_, runInfo_.kernel_w * blockSize_);

    if (CheckL0Size(baseM, baseN, runInfo_.kernel_h * runInfo_.kernel_w * blockSize_, dtypeByte_)) {
        baseK = runInfo_.kernel_h * runInfo_.kernel_w * blockSize_;
    } else if (CheckL0Size(baseM, baseN, tmpBaseKMax, dtypeByte_)) {
        baseK = tmpBaseKMax;
    } else if (CheckL0Size(baseM, baseN, tmpBaseKMin, dtypeByte_)) {
        baseK = tmpBaseKMin;
    } else {
        baseK = blockSize_;
    }

    tilingParams.baseM = baseM;
    tilingParams.baseK = baseK;
    tilingParams.baseN = baseN;
    tilingParams.baseD = 1U;
    tilingParams.baseBatch = 1U;
    tilingParams.baseGroup = 1U;

    tilingParams.al1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_al1);
    tilingParams.bl1Pbuffer = static_cast<uint32_t>(tbeTiling_.db_bl1);
    tilingParams.stepKa = Ops::Base::CeilDiv(
        static_cast<uint64_t>(tbeTiling_.k_al1) * runInfo_.kernel_h * runInfo_.kernel_w,
        static_cast<uint64_t>(baseK / blockSize_));
    tilingParams.stepKb = Ops::Base::CeilDiv(
        static_cast<uint64_t>(tbeTiling_.k_bl1) * runInfo_.kernel_h * runInfo_.kernel_w,
        static_cast<uint64_t>(baseK / blockSize_));

    if (dtypeByte_ == FP32_DATA_SIZE) {
        UpdateStepFp32(tilingParams.stepKa, tilingParams.stepKb, tilingParams);
    }

    tilingParams.stepM = 1U;
    tilingParams.stepN = 1U;
    tilingParams.stepBatch = 1U;
    tilingParams.stepGroup = 1U;
    tilingParams.iterateOrder = 1U;
}

void Conv3DBackpropInputV2Tiling::InitTilingValue(TilingValue& tilingParams, const uint32_t coreNum)
{
    bool enableTbeBlock = false;
    CalCoreDimTiling(tilingParams, coreNum, enableTbeBlock);
    if (enableTbeBlock || !CalBaseBlockTiling(tilingParams)) {
        OP_LOGD(opName_, "Use tbe cache tiling.");
        int32_t mDim = Ops::Base::CeilDiv(runInfo_.dedx_h, Ops::Base::CeilDiv(runInfo_.dedx_h, tbeTiling_.m_dim));
        SetTilingParamByDimInfo(tilingParams, tbeTiling_.batch_dim, tbeTiling_.d_dim, mDim, tbeTiling_.n_dim);
        CalTbeBlockTiling(tilingParams);
    }
}

void Conv3DBackpropInputV2Tiling::SetTilingValue(TConv3DInputV2Tiling& dxt, const TilingValue& tilingParams)
{
    // singleCore
    dxt.singleCoreBatch = tilingParams.singleCoreBatch;
    dxt.singleCoreGroup = tilingParams.singleCoreGroup;
    dxt.singleCoreM = tilingParams.singleCoreM;
    dxt.singleCoreCout = tilingParams.singleCoreCout;
    dxt.singleCoreCout1 = tilingParams.singleCoreCout1;
    dxt.singleCoreCin1 = tilingParams.singleCoreCin1;
    dxt.singleCoreCin = tilingParams.singleCoreCin;
    dxt.singleCoreDin = tilingParams.singleCoreDin;
    dxt.singleCoreHo = tilingParams.singleCoreHo;

    tilingData_.params.batchDim = tilingParams.batchDim;
    tilingData_.params.groupDim = tilingParams.groupDim;
    tilingData_.params.mDim = tilingParams.mDim;
    tilingData_.params.kDim = tilingParams.kDim;
    tilingData_.params.nDim = tilingParams.nDim;
    tilingData_.params.dDim = tilingParams.dDim;
    tilingData_.params.coreNum = tilingParams.coreNum;

    dxt.baseM = tilingParams.baseM;
    dxt.baseK = tilingParams.baseK;
    dxt.baseN = tilingParams.baseN;
    dxt.baseD = tilingParams.baseD;
    dxt.baseBatch = tilingParams.baseBatch;
    dxt.baseGroup = tilingParams.baseGroup;

    dxt.stepM = tilingParams.stepM;
    dxt.stepN = tilingParams.stepN;

    dxt.stepKa = tilingParams.stepKa;
    dxt.stepKb = tilingParams.stepKb;
    dxt.stepBatch = tilingParams.stepBatch;
    dxt.stepGroup = tilingParams.stepGroup;

    dxt.al0Pbuffer = tilingParams.al0Pbuffer; // 默认开
    dxt.bl0Pbuffer = tilingParams.bl0Pbuffer; // 默认开
    dxt.cl0Pbuffer = tilingParams.cl0Pbuffer;
    dxt.al1Pbuffer = tilingParams.al1Pbuffer;
    dxt.bl1Pbuffer = tilingParams.bl1Pbuffer;

    dxt.iterateOrder = tilingParams.iterateOrder;

    if (runInfo_.kernel_h * runInfo_.kernel_w == 1) {
        loadB2Condition_ = 2; // 2表示Hk*Wk = 1的情况
    } else if (tilingParams.baseK / blockSize_ >= static_cast<uint32_t>(runInfo_.kernel_h * runInfo_.kernel_w)) {
        loadB2Condition_ = 1;
    } else {
        loadB2Condition_ = 0;
    }
    if (coreNum_ == CORE_NUM_910B2 || coreNum_ == CORE_NUM_910B3) {
        enableKernelSplit_ = CheckKernelSplitEnable();
    }
}

void Conv3DBackpropInputV2Tiling::SetBackpropPadInfo(TConv3DInputV2Tiling& dxt)
{
    int64_t bpPadTail = runInfo_.dedx_d - (static_cast<int64_t>(runInfo_.dedy_d - 1) * runInfo_.stride_d + 1) +
                        (runInfo_.kernel_d - 1) * runInfo_.dilation_d - runInfo_.backprop_pad_h;
    if (bpPadTail < PAD_DIM_LOW || bpPadTail > PAD_DIM_UP) {
        dxt.backpropPadTail = runInfo_.backprop_pad_t;
    } else {
        dxt.backpropPadTail = static_cast<uint32_t>(bpPadTail);
    }
    OP_LOGD(opName_, "backprop tail pad: %ld, origin backprop_pad_t: %d", bpPadTail, runInfo_.backprop_pad_t);

    dxt.backpropPadUp = runInfo_.backprop_pad_u;
    int64_t bpPadDown = runInfo_.dedx_h - (static_cast<int64_t>(runInfo_.dedy_h - 1) * runInfo_.stride_h + 1) +
                        (runInfo_.kernel_h - 1) * runInfo_.dilation_h - runInfo_.backprop_pad_u;
    if (bpPadDown < PAD_DIM_LOW || bpPadDown > PAD_DIM_UP) {
        dxt.backpropPadDown = runInfo_.backprop_pad_d;
    } else {
        dxt.backpropPadDown = static_cast<uint32_t>(bpPadDown);
    }
    OP_LOGD(opName_, "backprop down pad: %ld, origin backprop_pad_d: %d", bpPadDown, runInfo_.backprop_pad_d);

    dxt.backpropPadLeft = runInfo_.backprop_pad_l;
    int64_t bpPadRight = runInfo_.dedx_w - (static_cast<int64_t>(runInfo_.dedy_w - 1) * runInfo_.stride_w + 1) +
                         (runInfo_.kernel_w - 1) * runInfo_.dilation_w - runInfo_.backprop_pad_l;
    if (bpPadRight < PAD_DIM_LOW || bpPadRight > PAD_DIM_UP) {
        dxt.backpropPadRight = runInfo_.backprop_pad_r;
    } else {
        dxt.backpropPadRight = static_cast<uint32_t>(bpPadRight);
    }
    OP_LOGD(opName_, "backprop right pad: %ld, origin backprop_pad_r: %d", bpPadRight, runInfo_.backprop_pad_r);
}

void Conv3DBackpropInputV2Tiling::SetRunInfoTiling(TConv3DInputV2Tiling& dxt)
{
    // shape
    dxt.batch = runInfo_.batch_n;
    dxt.cin = runInfo_.dedx_cin;
    dxt.cout = runInfo_.dedy_cout;
    dxt.cin1G = runInfo_.dedx_cin1_g;
    dxt.cout1G = runInfo_.dedy_cout1_g;
    dxt.cin1 = runInfo_.dedx_cin1;
    dxt.cout1 = runInfo_.dedy_cout1;
    dxt.c0 = blockSize_;
    if (dtypeByte_ == F16_DATA_SIZE) {
        dxt.c0Bits = B16_BITS;
    } else if (dtypeByte_ == FP32_DATA_SIZE) {
        dxt.c0Bits = FP32_BITS;
    }
    dxt.ho = runInfo_.dedy_h;
    dxt.wo = runInfo_.dedy_w;
    dxt.dout = runInfo_.dedy_d;
    dxt.di = runInfo_.dedx_d;
    dxt.hi = runInfo_.dedx_h;
    dxt.wi = runInfo_.dedx_w;
    dxt.hk = runInfo_.kernel_h;
    dxt.wk = runInfo_.kernel_w;
    dxt.dk = runInfo_.kernel_d;

    dxt.group = runInfo_.real_g;
    dxt.strideH = runInfo_.stride_h;
    dxt.strideW = runInfo_.stride_w;
    dxt.strideD = runInfo_.stride_d;
    dxt.padFront = runInfo_.pad_h;
    dxt.padBack = runInfo_.pad_t;
    dxt.padUp = runInfo_.pad_u;
    dxt.padDown = runInfo_.pad_d;
    dxt.padLeft = runInfo_.pad_l;
    dxt.padRight = runInfo_.pad_r;
    SetBackpropPadInfo(dxt);

    dxt.dilationH = runInfo_.dilation_h;
    dxt.dilationW = runInfo_.dilation_w;
    dxt.dilationD = runInfo_.dilation_d;
    dxt.hf32Flag = runInfo_.hf32_flag;
    dxt.initOutputFlag = runInfo_.initOutputFlag;
}

void Conv3DBackpropInputV2Tiling::SetDxTilingFromTbeTiling()
{
    TConv3DInputV2Tiling& dxt = tilingData_.conv3DDxTiling;
    TilingValue tilingParams;
    // key:
    // "N_Do_Co1_Ho_Wo_Di_Ci1_Hi_Wi_Dk_Hk_Wk_strideD_strideH_strideW_
    // _padFront_padBack_padUp_padDown_padLeft_padRight_dilationD_dilationH_dilationW"
    std::string key = std::to_string(runInfo_.batch_n) + "_" + std::to_string(runInfo_.dedy_d) + "_" +
                      std::to_string(runInfo_.dedy_cout1) + "_" + std::to_string(runInfo_.dedy_h) + "_" +
                      std::to_string(runInfo_.dedy_w) + "_" + std::to_string(runInfo_.dedx_d) + "_" +
                      std::to_string(runInfo_.dedx_cin1) + "_" + std::to_string(runInfo_.dedx_h) + "_" +
                      std::to_string(runInfo_.dedx_w) + "_" + std::to_string(runInfo_.kernel_d) + "_" +
                      std::to_string(runInfo_.kernel_h) + "_" + std::to_string(runInfo_.kernel_w) + "_" +
                      std::to_string(runInfo_.stride_d) + "_" + std::to_string(runInfo_.stride_h) + "_" +
                      std::to_string(runInfo_.stride_w) + "_" + std::to_string(runInfo_.pad_h) + "_" +
                      std::to_string(runInfo_.pad_t) + "_" + std::to_string(runInfo_.pad_u) + "_" +
                      std::to_string(runInfo_.pad_d) + "_" + std::to_string(runInfo_.pad_l) + "_" +
                      std::to_string(runInfo_.pad_r) + "_" + std::to_string(runInfo_.dilation_d) + "_" +
                      std::to_string(runInfo_.dilation_h) + "_" + std::to_string(runInfo_.dilation_w);

    coreNum_ = context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->core_num;

    OP_TILING_CHECK(
        coreNum_ <= 0,
        CUBE_INNER_ERR_REPORT(
            opName_, "Failed to get valid core number from platform information. core num: %d", coreNum_),
        return);
    bool isHitKnowledgeMap = false;
    if (runInfo_.real_g == 1) {
        if (dtypeByte_ == F16_DATA_SIZE) {
            if (coreNum_ == CORE_NUM_910B3 && TILING_DATA_MAP_B3.find(key) != TILING_DATA_MAP_B3.end()) {
                tilingParams = TILING_DATA_MAP_B3.at(key);
                isHitKnowledgeMap = true;
            } else if (coreNum_ == CORE_NUM_910B2 && TILING_DATA_MAP_B2.find(key) != TILING_DATA_MAP_B2.end()) {
                tilingParams = TILING_DATA_MAP_B2.at(key);
                isHitKnowledgeMap = true;
            }
        } else if (dtypeByte_ == FP32_DATA_SIZE) {
            if (coreNum_ == CORE_NUM_910B2 && FP32_TILING_DATA_MAP_B2.find(key) != FP32_TILING_DATA_MAP_B2.end()) {
                tilingParams = FP32_TILING_DATA_MAP_B2.at(key);
                isHitKnowledgeMap = true;
            } else if (
                coreNum_ == CORE_NUM_910B3 && FP32_TILING_DATA_MAP_B3.find(key) != FP32_TILING_DATA_MAP_B3.end()) {
                tilingParams = FP32_TILING_DATA_MAP_B3.at(key);
                isHitKnowledgeMap = true;
            } else if (coreNum_ == CORE_NUM_910B2 && FP32_BASIC_BLOCK_MAP_B2.count(key) > 0) {
                tilingParams = FP32_BASIC_BLOCK_MAP_B2.at(key);
                useBasicBlock_ = isHitKnowledgeMap = true;
            } else if (coreNum_ == CORE_NUM_910B3 && FP32_BASIC_BLOCK_MAP_B3.count(key) > 0) {
                tilingParams = FP32_BASIC_BLOCK_MAP_B3.at(key);
                useBasicBlock_ = isHitKnowledgeMap = true;
            }
        }
    }
    if (isHitKnowledgeMap) {
        OP_LOGD(opName_, "hit tiling knowledge map");
    } else {
        InitTilingValue(tilingParams, coreNum_);
    }
    SetRunInfoTiling(dxt);
    SetTilingValue(dxt, tilingParams);
}

void Conv3DBackpropInputV2Tiling::PrintTilingData()
{
    TConv3DInputV2Tiling &tiling = tilingData_.conv3DDxTiling;
    std::stringstream ss;
    ss << " batch: " << tiling.batch << " cin: " << tiling.cin << " cout: " << tiling.cout
       << " di: " << tiling.di << " dout: " << tiling.dout << " dk: " << tiling.dk
       << " ho: " << tiling.ho << " wo: " << tiling.wo << " hi: " << tiling.hi
       << " wi: " << tiling.wi << " hk: " << tiling.hk << " wk: " << tiling.wk
       << " singleCoreBatch: " << tiling.singleCoreBatch << " singleCoreGroup: " << tiling.singleCoreGroup
       << " singleCoreCout: " << tiling.singleCoreCout << " singleCoreCin: " << tiling.singleCoreCin
       << " singleCoreHo: " << tiling.singleCoreHo << " baseM: " << tiling.baseM
       << " baseK: " << tiling.baseK << " baseN: " << tiling.baseN << " baseD: " << tiling.baseD
       << " baseBatch: " << tiling.baseBatch << " baseGroup: " << tiling.baseGroup
       << " stepM: " << tiling.stepM << " stepN: " << tiling.stepN << " stepKa: " << tiling.stepKa
       << " stepKb: " << tiling.stepKb << " stepBatch: " << tiling.stepBatch
       << " stepGroup: " << tiling.stepGroup << " al0Pbuffer: " << tiling.al0Pbuffer
       << " bl0Pbuffer: " << tiling.bl0Pbuffer << " cl0Pbuffer: " << tiling.cl0Pbuffer
       << " al1Pbuffer: " << tiling.al1Pbuffer << " bl1Pbuffer: " << tiling.bl1Pbuffer
       << " group: " << tiling.group << " strideH: " << tiling.strideH
       << " strideW: " << tiling.strideW << " strideD: " << tiling.strideD
       << " padFront: " << tiling.padFront << " padBack: " << tiling.padBack
       << " padUp: " << tiling.padUp << " padDown: " << tiling.padDown
       << " padLeft: " << tiling.padLeft << " padRight: " << tiling.padRight
       << " dilationH: " << tiling.dilationH << " dilationW: " << tiling.dilationW
       << " dilationD: " << tiling.dilationD << " iterateOrder: " << tiling.iterateOrder
       << " hf32Flag: " << tiling.hf32Flag;
    OP_LOGD(opName_, "api tiling: %s", ss.str().c_str());
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DBackpropInputV2Tiling, 1);

} // namespace Conv
} // namespace NN
} // namespace Ops
