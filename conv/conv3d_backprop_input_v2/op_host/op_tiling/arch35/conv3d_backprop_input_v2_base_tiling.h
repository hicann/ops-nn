/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_input_v2_base_tiling.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_TILING_ADVANCE_H
#define CONV3D_BACKPROP_INPUT_V2_TILING_ADVANCE_H

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "tiling_base/tiling_base.h"
#include "../common/conv_backprop_input_context_utils.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace optiling;
using namespace Ops::NN::Optiling;

constexpr uint8_t TILING_GROUP_MODE_ORIGIN = 0;
constexpr uint8_t TILING_GROUP_MODE_ENLARGE = 1;

constexpr int64_t L0_AB_SIZE = 65536;
constexpr int64_t L0C_SIZE = 256 * 1024;
constexpr uint64_t L1_SIZE = 512 * 1024;
constexpr uint64_t UB_SIZE = 248 * 1024;
constexpr int64_t WORKSIZE = static_cast<int64_t>(16 * 1024 * 1024);

constexpr uint32_t DB_ON = 2;
constexpr uint32_t DB_OFF = 1;
constexpr int32_t BLOCK_CUBE = 16;
constexpr uint32_t FP32_DATA_SIZE = 4;
constexpr int32_t FMAP_H_NUM = 2;
constexpr int32_t BUFFER_NUM_DB = 2;

const size_t Y_INDEX = 0;
const size_t INPUT_SIZE_INDEX = 0;
const size_t FILTER_INDEX = 1;
const size_t OUTPUT_BP_INDEX = 2;
const size_t BAIS_INDEX = 3;
const size_t OFFSET_W_INDEX = 4;
const size_t OUTPUT_PADDING_INDEX = 5;
const size_t OFFSET_X_INDEX = 6;

// TConv3DInputV2TilingAdvance对齐到第56行
// TConv3DInputV2TilingAdvance对齐到第56行
BEGIN_TILING_DATA_DEF(TConv3DInputV2TilingAdvance)
    TILING_DATA_FIELD_DEF(uint8_t, al0Pbuffer);
    TILING_DATA_FIELD_DEF(uint8_t, bl0Pbuffer);
    TILING_DATA_FIELD_DEF(uint8_t, cl0Pbuffer);
    TILING_DATA_FIELD_DEF(uint8_t, al1Pbuffer);
    TILING_DATA_FIELD_DEF(uint8_t, bl1Pbuffer);
    TILING_DATA_FIELD_DEF(uint8_t, iterateOrder);
    TILING_DATA_FIELD_DEF(uint8_t, c0);
    TILING_DATA_FIELD_DEF(uint8_t, c0Bits);
    TILING_DATA_FIELD_DEF(uint8_t, enlarge);
    TILING_DATA_FIELD_DEF(uint8_t, hf32Flag);
    TILING_DATA_FIELD_DEF(uint8_t, initOutputFlag);
    TILING_DATA_FIELD_DEF(uint8_t, isBiasFullLoad);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, cin);
    TILING_DATA_FIELD_DEF(uint32_t, cout);
    TILING_DATA_FIELD_DEF(uint32_t, cinG);
    TILING_DATA_FIELD_DEF(uint32_t, coutG);
    TILING_DATA_FIELD_DEF(uint32_t, cout1);
    TILING_DATA_FIELD_DEF(uint32_t, cin1);
    TILING_DATA_FIELD_DEF(uint32_t, cout1G);
    TILING_DATA_FIELD_DEF(uint32_t, cin1G);
    TILING_DATA_FIELD_DEF(uint32_t, dout);
    TILING_DATA_FIELD_DEF(uint32_t, ho);
    TILING_DATA_FIELD_DEF(uint32_t, wo);
    TILING_DATA_FIELD_DEF(uint32_t, di);
    TILING_DATA_FIELD_DEF(uint32_t, hi);
    TILING_DATA_FIELD_DEF(uint32_t, wi);
    TILING_DATA_FIELD_DEF(uint32_t, dk);
    TILING_DATA_FIELD_DEF(uint32_t, hk);
    TILING_DATA_FIELD_DEF(uint32_t, wk);
    TILING_DATA_FIELD_DEF(uint32_t, group);
    TILING_DATA_FIELD_DEF(uint32_t, oriGroup);
    TILING_DATA_FIELD_DEF(uint32_t, strideD);
    TILING_DATA_FIELD_DEF(uint32_t, strideH);
    TILING_DATA_FIELD_DEF(uint32_t, strideW);
    TILING_DATA_FIELD_DEF(uint32_t, padFront);
    TILING_DATA_FIELD_DEF(uint32_t, padBack);
    TILING_DATA_FIELD_DEF(uint32_t, padUp);
    TILING_DATA_FIELD_DEF(uint32_t, padDown);
    TILING_DATA_FIELD_DEF(uint32_t, padLeft);
    TILING_DATA_FIELD_DEF(uint32_t, padRight);
    TILING_DATA_FIELD_DEF(uint32_t, backpropPadTail);
    TILING_DATA_FIELD_DEF(uint32_t, backpropPadUp);
    TILING_DATA_FIELD_DEF(uint32_t, backpropPadDown);
    TILING_DATA_FIELD_DEF(uint32_t, backpropPadLeft);
    TILING_DATA_FIELD_DEF(uint32_t, backpropPadRight);
    TILING_DATA_FIELD_DEF(uint32_t, dilationD);
    TILING_DATA_FIELD_DEF(uint32_t, dilationH);
    TILING_DATA_FIELD_DEF(uint32_t, dilationW);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreGroup);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreCout);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreCin);
    TILING_DATA_FIELD_DEF(uint32_t, singleCoreDin);
    TILING_DATA_FIELD_DEF(uint32_t, baseM);
    TILING_DATA_FIELD_DEF(uint32_t, baseK);
    TILING_DATA_FIELD_DEF(uint32_t, baseN);
    TILING_DATA_FIELD_DEF(uint32_t, stepM);
    TILING_DATA_FIELD_DEF(uint32_t, stepN);
    TILING_DATA_FIELD_DEF(uint32_t, stepKa);
    TILING_DATA_FIELD_DEF(uint32_t, stepKb);
    TILING_DATA_FIELD_DEF(uint32_t, singleIterateDk);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreBatch);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreM);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(Conv3DBackpropInputV2ParamsAdvance)
    TILING_DATA_FIELD_DEF(uint32_t, batchDim);
    TILING_DATA_FIELD_DEF(uint32_t, groupDim);
    TILING_DATA_FIELD_DEF(uint32_t, mDim);
    TILING_DATA_FIELD_DEF(uint32_t, kDim);
    TILING_DATA_FIELD_DEF(uint32_t, nDim);
    TILING_DATA_FIELD_DEF(uint32_t, dDim);
    TILING_DATA_FIELD_DEF(uint64_t, coreNum);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(TConv3DInputV2KSTilingAdvance)
    TILING_DATA_FIELD_DEF(uint32_t, kSCoutFullLoad);
    TILING_DATA_FIELD_DEF(uint32_t, kSUseWorkSpace);
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(Conv3DBackpropInputV2TilingDataAdvance)
    TILING_DATA_FIELD_DEF_STRUCT(Conv3DBackpropInputV2ParamsAdvance, params);
    TILING_DATA_FIELD_DEF_STRUCT(TConv3DInputV2TilingAdvance, conv3DDxTiling);
    TILING_DATA_FIELD_DEF_STRUCT(TConv3DInputV2KSTilingAdvance, conv3DDxKSTiling);
END_TILING_DATA_DEF;

struct TilingValueDavid {
    uint64_t coreNum;
    uint32_t batchDim;
    uint32_t groupDim;
    uint32_t mDim;
    uint32_t kDim;
    uint32_t nDim;
    uint32_t dDim;
    uint64_t singleCoreBatch;
    uint32_t singleCoreGroup;
    uint64_t singleCoreM;
    uint32_t singleCoreCout;
    uint32_t singleCoreCout1;
    uint64_t singleCoreCin;
    uint32_t singleCoreCin1;
    uint32_t singleCoreDin;
    uint32_t singleCoreHo;
    uint8_t al0Pbuffer;
    uint8_t bl0Pbuffer;
    uint8_t cl0Pbuffer;
    uint8_t al1Pbuffer;
    uint8_t bl1Pbuffer;
    uint32_t baseM;
    uint32_t baseK;
    uint32_t baseN;
    uint32_t baseD;
    uint32_t baseBatch;
    uint32_t baseGroup;
    uint32_t stepM;
    uint32_t stepN;
    uint32_t stepKa;
    uint32_t stepKb;
    uint32_t stepBatch;
    uint32_t stepGroup;
    uint8_t iterateOrder;
};

struct RunInfoParaV2 {
    // shape vars
    int32_t batch_n;
    int32_t dedy_cout;
    int32_t dedy_d;
    int32_t dedy_h;
    int32_t dedy_w;
    int32_t dedx_cin;
    int32_t dedx_d;
    int32_t dedx_h;
    int32_t dedx_w;
    int32_t kernel_d;
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t dedy_cout_g; // 910_95
    int32_t dedx_cin_g;  // 910_95
    int32_t dedy_cout1;
    int32_t dedx_cin1;
    int32_t real_g;
    int32_t dedy_cout1_g;
    int32_t dedx_cin1_g;
    int32_t kernel_g_dk_cin1g_hk_wk;
    // attr vars
    int32_t stride_d;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_t;
    int32_t pad_u;
    int32_t pad_d;
    int32_t pad_l;
    int32_t pad_r;
    int32_t dilation_d;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t shape_up_modify;
    int32_t shape_left_modify;
    int32_t shape_down_modify;
    int32_t shape_right_modify;
    int32_t backprop_pad_h;
    int32_t backprop_pad_t;
    int32_t backprop_pad_u;
    int32_t backprop_pad_d;
    int32_t backprop_pad_l;
    int32_t backprop_pad_r;
    // tiling vars
    int32_t batch_dim;
    int32_t n_dim;
    int32_t m_dim;
    int32_t group_dim;
    int32_t d_dim;
    int32_t m_al1;
    int32_t n_bl1;
    int32_t m_l0;
    int32_t n_l0_div_ub;
    int32_t n_cub;
    int32_t k_l0;
    int32_t min_kl1_div_kl0;
    int32_t max_kl1_div_min_kl1;
    int32_t k_div_max_kl1;
    int32_t d_al1;
    int32_t d_bl1;
    int32_t d_al0;
    int32_t d_bl0;
    int32_t d_cl0;
    int32_t k_aub;
    int32_t m_aub;
    int32_t wo_aub;
    int32_t al1_bound;
    int32_t bl1_bound;
    int32_t aub_bound;

    int32_t load3d_special;
    int32_t hf32_flag;
    int32_t groups;  // 910_95
    int32_t enlarge; // 910_95
    // format vars
    ge::Format outBackpropFormat;
    ge::Format filterFormat;
    ge::Format yFormat;
};

struct TilingCoreDimDx {
    int32_t batchDim = 1;
    int32_t dDim = 1;
    int32_t mDim = 1;
    int32_t nDim = 1;
};

struct TilingBestBaseBlock {
    uint32_t baseM;
    uint32_t baseK;
    uint32_t baseN;
};

enum class IterateOrder
{
    ORDER_M = 0,
    ORDER_N,
    ORDER_K,
    UNDEF,
};

class Conv3DBackpropInputV2TilingArch35 : public TilingBaseClass {
public:
    explicit Conv3DBackpropInputV2TilingArch35(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~Conv3DBackpropInputV2TilingArch35() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    void Reset();
    bool GetShapeFormatInfo();
    ge::graphStatus CheckContext();
    bool CheckDtypeFormatAttrs(size_t aMatrixesIndex, size_t bMatrixesIndex, bool hif8flag, bool fp8e4m3flag) const;
    bool AnalyzeDtype() const;
    void SetDxTilingFromTbeTiling();
    void PrintTilingData();
    void PrintTbeTiling();
    int32_t GetDimFactor(const int64_t& value, const std::vector<int32_t>& factorLits) const;
    int32_t CalFmapH(const int32_t& mL1Size) const;
    void GetSubFactors(const std::vector<int32_t>& src, int32_t factor, int32_t base, std::vector<int32_t>& dst);
    bool GetCoreDimForMN(
        int32_t curCoreNum, TilingCoreDimDx& coreDim, const std::vector<int32_t>& coreFactors, int32_t remainFactor);
    bool GetCoreDim(int32_t curCoreNum, TilingCoreDimDx& coreDim);
    void UpdateBaseBlock(
        uint32_t& baseM, uint32_t& baseK, uint32_t& baseN, const TilingValueDavid& tilingParams,
        const TilingBestBaseBlock& baseBlock);
    void UpdateBaseStep(uint32_t& stepKa, uint32_t& stepKb, TilingValueDavid& tilingParams);
    bool CalCoreDimTiling(TilingValueDavid& tilingParams, const uint32_t coreNum);
    void SetTilingParamByDimInfo(TilingValueDavid& tilingParams, const TilingCoreDimDx& coreDim);
    bool CheckBaseBlockTiling(TilingValueDavid& tilingParams);
    bool CalBaseBlockTiling(TilingValueDavid& tilingParams, const TilingBestBaseBlock& baseBlock);
    void CalTbeBlockTiling(TilingValueDavid& tilingParams);
    void InitTilingValue(TilingValueDavid& tilingParams, const uint32_t coreNum);
    void SetRunBaseShapeInfoTiling(TConv3DInputV2TilingAdvance& dxt);
    void SetRunInfoTiling(TConv3DInputV2TilingAdvance& dxt);
    void SetLoadB2Condition(const TilingValueDavid& tilingParams);
    void SetGroupConvMode(TConv3DInputV2TilingAdvance& dxt);
    void SetBackpropPadInfo(TConv3DInputV2TilingAdvance& dxt);
    void SetTilingValue(TConv3DInputV2TilingAdvance& dxt, const TilingValueDavid& tilingParams);

    bool a1DbFlag_ = false;
    bool b1DbFlag_ = false;
    bool c0DbFlag_ = false;
    bool enableSplitDk_ = false;
    bool hasBiasFlag_ = false;
    uint8_t loadB2Condition_ = 0;
    uint8_t loadB1Condition_ = 0;
    uint8_t groupConvMode_ = TILING_GROUP_MODE_ORIGIN;
    platform_ascendc::SocVersion shortSocVersion_;
    int32_t coreNum_ = 1;
    int32_t isBiasFullLoad = 0;
    uint32_t singleIterateDk_ = 1;

    int32_t blockSize_ = 16;
    uint32_t dtypeByte_ = 2;
    uint32_t dtypeByteL0c_ = 4;
    const char* opName_ = "";
    uint64_t usrSpaceSizeForSplitDk_ = 0;
    optiling::OpTypeV2 opType_ = optiling::OpTypeV2::kConv3DBackpropInputV2;
    Conv3DBackpropInputV2TilingDataAdvance tilingData_ = {};
    Conv3dBpInputV2RunInfo runInfo_ = {};

private:
    Conv3dBackpropV2TBETilingData tbeTiling_ = {};
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_TILING_ADVANCE_H