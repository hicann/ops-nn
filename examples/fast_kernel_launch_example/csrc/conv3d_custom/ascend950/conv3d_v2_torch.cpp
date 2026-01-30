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
* \file conv3d_v2_torch.cpp
* \brief
*/

#include <fstream>
#include <iostream>
#include <cstring>
#include <string>
#include "tiling/platform/platform_ascendc.h"
#include "acl/acl.h"
#include "graph/types.h"
#include "conv3d_v2_tiling_data.h"
#include "conv_template_utils.h"
#include "conv_base_blockdim_decision.h"
#include "kernel_operator.h"
#include "conv3d_v2_api_tiling.h"
#include "conv3d_v2_base_tiling_template_tilingkey.h"
#include "conv3d_v2_kernel.h"
#include "conv_api_tiling_util.h"
#include <torch/all.h>
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#define DEMO_RET_FAIL       (-1)
#define DEMO_RET_SUC        (0)
constexpr uint32_t FORMAT_NCDHW_N_INDEX = 0;
constexpr uint32_t FORMAT_NCDHW_C_INDEX = 1;
constexpr uint32_t FORMAT_NCDHW_D_INDEX = 2;
constexpr uint32_t FORMAT_NCDHW_H_INDEX = 3;
constexpr uint32_t FORMAT_NCDHW_W_INDEX = 4;
constexpr uint8_t ATTRS_D_DIM_IDX_NCDHW = 0;
constexpr uint8_t ATTRS_H_DIM_IDX_NCDHW = 1;
constexpr uint8_t ATTRS_W_DIM_IDX_NCDHW = 2;
constexpr uint32_t PAD_HEAD_INDEX = 0;
constexpr uint32_t PAD_TAIL_INDEX = 1;
constexpr uint32_t PAD_UP_INDEX = 2;
constexpr uint32_t PAD_DOWN_INDEX = 3;
constexpr uint32_t PAD_LEFT_INDEX = 4;
constexpr uint32_t PAD_RIGHT_INDEX = 5;
constexpr static uint8_t NCDHW_DIM = 5;
constexpr static uint8_t CONV_ATTRS_DIM = 3;

namespace ascend_ops {
namespace Conv3dCustom {

static void InitConv3dRunInfo(Ops::NN::Conv3dV2::Conv3DRunInfo& conv3dRunInfo,
                              optiling::conv_ops_tiling::ConvAscendcTilingInfo& tilingInfo)
{
    conv3dRunInfo.batch     = static_cast<uint32_t>(tilingInfo.shapeInfo.batch);
    conv3dRunInfo.cin       = static_cast<uint32_t>(tilingInfo.shapeInfo.ci);
    conv3dRunInfo.din       = static_cast<uint32_t>(tilingInfo.shapeInfo.di);
    conv3dRunInfo.hin       = static_cast<uint32_t>(tilingInfo.shapeInfo.hi);
    conv3dRunInfo.win       = static_cast<uint32_t>(tilingInfo.shapeInfo.wi);
    conv3dRunInfo.cout      = static_cast<uint32_t>(tilingInfo.shapeInfo.co);
    conv3dRunInfo.kd        = static_cast<uint32_t>(tilingInfo.shapeInfo.kd);
    conv3dRunInfo.kh        = static_cast<uint32_t>(tilingInfo.shapeInfo.kh);
    conv3dRunInfo.kw        = static_cast<uint32_t>(tilingInfo.shapeInfo.kw);
    conv3dRunInfo.dout      = static_cast<uint32_t>(tilingInfo.shapeInfo.dout);
    conv3dRunInfo.hout      = static_cast<uint32_t>(tilingInfo.shapeInfo.ho);
    conv3dRunInfo.wout      = static_cast<uint32_t>(tilingInfo.shapeInfo.wo);
    conv3dRunInfo.batchDim  = tilingInfo.blockDimRes.batchDim;
    conv3dRunInfo.doDim     = tilingInfo.blockDimRes.doDim;
    conv3dRunInfo.mDim      = tilingInfo.blockDimRes.mDim;
    conv3dRunInfo.wDim      = tilingInfo.blockDimRes.woDim;
    conv3dRunInfo.nDim      = tilingInfo.blockDimRes.nDim;
    conv3dRunInfo.groupDim  = tilingInfo.blockDimRes.groupDim;
    conv3dRunInfo.hoDim     = tilingInfo.blockDimRes.hoDim;
    conv3dRunInfo.strideD   = static_cast<uint32_t>(tilingInfo.attrInfo.strideD);
    conv3dRunInfo.strideH   = static_cast<uint32_t>(tilingInfo.attrInfo.strideH);
    conv3dRunInfo.strideW   = static_cast<uint32_t>(tilingInfo.attrInfo.strideW);
    conv3dRunInfo.dilationD = static_cast<uint32_t>(tilingInfo.attrInfo.dilationD);
    conv3dRunInfo.dilationH = static_cast<uint32_t>(tilingInfo.attrInfo.dilationH);
    conv3dRunInfo.dilationW = static_cast<uint32_t>(tilingInfo.attrInfo.dilationW);
    conv3dRunInfo.padHead   = static_cast<uint32_t>(tilingInfo.attrInfo.padHead);
    conv3dRunInfo.padTail   = static_cast<uint32_t>(tilingInfo.attrInfo.padTail);
    conv3dRunInfo.padTop    = static_cast<uint32_t>(tilingInfo.attrInfo.padTop);
    conv3dRunInfo.padBottom = static_cast<uint32_t>(tilingInfo.attrInfo.padBottom);
    conv3dRunInfo.padLeft   = static_cast<uint32_t>(tilingInfo.attrInfo.padLeft);
    conv3dRunInfo.padRight  = static_cast<uint32_t>(tilingInfo.attrInfo.padRight);
    conv3dRunInfo.groups    = 1;
    conv3dRunInfo.enlarge   = 1;
    conv3dRunInfo.cinOpt    = static_cast<uint32_t>(tilingInfo.shapeInfo.ci); // GROUP场景下
    conv3dRunInfo.coutOpt   = static_cast<uint32_t>(tilingInfo.shapeInfo.co);
    conv3dRunInfo.groupOpt  = 1;
    conv3dRunInfo.hasBias   = static_cast<uint8_t>(tilingInfo.flagInfo.hasBias);
    if (tilingInfo.flagInfo.mSplitModeFlag) {
        conv3dRunInfo.hoDim = static_cast<uint32_t>(tilingInfo.blockDimRes.mDim);
    } else {
        conv3dRunInfo.hoDim = static_cast<uint32_t>(tilingInfo.blockDimRes.hoDim);
    }
}

static void InitTilingData(Ops::NN::Conv3dV2::Conv3DV2TilingData& tilingData,
    optiling::conv_ops_tiling::ConvAscendcTilingInfo& tilingInfo)
{
    InitConv3dRunInfo(tilingData.conv3dRunInfo, tilingInfo);
}


static int32_t InitConvBase(const vector<int64_t>& oriInputShapeList,
    const vector<int64_t>& oriWeightShapeList, const vector<int64_t>& oriOutputShapeList,
    optiling::conv_ops_tiling::ConvAscendcTilingInfo& tilingInfo, ge::DataType dataType)
{
    tilingInfo.shapeInfo.batch  = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_N_INDEX]);
    tilingInfo.shapeInfo.ci     = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_C_INDEX]);
    tilingInfo.shapeInfo.di     = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_D_INDEX]);
    tilingInfo.shapeInfo.hi     = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_H_INDEX]);
    tilingInfo.shapeInfo.wi     = static_cast<uint64_t>(oriInputShapeList[FORMAT_NCDHW_W_INDEX]);
    tilingInfo.shapeInfo.kd     = static_cast<uint64_t>(oriWeightShapeList[FORMAT_NCDHW_D_INDEX]);
    tilingInfo.shapeInfo.kh     = static_cast<uint64_t>(oriWeightShapeList[FORMAT_NCDHW_H_INDEX]);
    tilingInfo.shapeInfo.kw     = static_cast<uint64_t>(oriWeightShapeList[FORMAT_NCDHW_W_INDEX]);
    tilingInfo.shapeInfo.co     = static_cast<uint64_t>(oriWeightShapeList[FORMAT_NCDHW_N_INDEX]);
    tilingInfo.shapeInfo.dout   = static_cast<uint64_t>(oriOutputShapeList[FORMAT_NCDHW_D_INDEX]);
    tilingInfo.shapeInfo.ho     = static_cast<uint64_t>(oriOutputShapeList[FORMAT_NCDHW_H_INDEX]);
    tilingInfo.shapeInfo.wo     = static_cast<uint64_t>(oriOutputShapeList[FORMAT_NCDHW_W_INDEX]);

    tilingInfo.descInfo.weightDtype    = dataType;
    tilingInfo.descInfo.fMapDtype      = dataType;
    tilingInfo.descInfo.biasDtype      = dataType;
    tilingInfo.descInfo.outDtype       = dataType;
    tilingInfo.descInfo.out1Dtype      = dataType;

    tilingInfo.descInfo.weightFormat   = ge::FORMAT_NCDHW;
    tilingInfo.descInfo.fMapFormat     = ge::FORMAT_NCDHW;
    tilingInfo.descInfo.biasFormat     = ge::FORMAT_NCDHW;
    tilingInfo.descInfo.outFormat      = ge::FORMAT_NCDHW;
    tilingInfo.descInfo.out1Format     = ge::FORMAT_NCDHW;
    tilingInfo.descInfo.scaleFormat = ge::FORMAT_NCDHW;

    tilingInfo.descInfo.scaleDtype = ge::DataType::DT_INT64;
    tilingInfo.flagInfo.quantFlag      = false;
    tilingInfo.flagInfo.extendConvFlag = false;
    tilingInfo.flagInfo.enableC04Flag  = false;
    tilingInfo.flagInfo.mSplitModeFlag = false;
    tilingInfo.flagInfo.convGroupType  = optiling::conv_ops_tiling::ConvGroupType::NORMAL_CONV;
    tilingInfo.flagInfo.mBasicBlockFlag = false;
    tilingInfo.flagInfo.useTilingRepo  = false;
    tilingInfo.flagInfo.useTilingCache = false;

    return DEMO_RET_SUC;
}

static int32_t InitAttrInfo(const vector<int64_t>& strideList, const vector<int64_t>& paddingList,
    const vector<int64_t>& dilationList, optiling::conv_ops_tiling::ConvAscendcAttrInfo& attrInfo)
{
    attrInfo.dilationD  = dilationList[ATTRS_D_DIM_IDX_NCDHW];
    attrInfo.dilationH  = dilationList[ATTRS_H_DIM_IDX_NCDHW];
    attrInfo.dilationW  = dilationList[ATTRS_W_DIM_IDX_NCDHW];
    attrInfo.strideD    = strideList[ATTRS_D_DIM_IDX_NCDHW];
    attrInfo.strideH    = strideList[ATTRS_H_DIM_IDX_NCDHW];
    attrInfo.strideW    = strideList[ATTRS_W_DIM_IDX_NCDHW];
    attrInfo.padHead    = paddingList[PAD_HEAD_INDEX];
    attrInfo.padTail    = paddingList[PAD_TAIL_INDEX];
    attrInfo.padTop     = paddingList[PAD_UP_INDEX];
    attrInfo.padBottom  = paddingList[PAD_DOWN_INDEX];
    attrInfo.padLeft    = paddingList[PAD_LEFT_INDEX];
    attrInfo.padRight   = paddingList[PAD_RIGHT_INDEX];
    attrInfo.hf32Mode   = 0;
    attrInfo.offsetx    = 0;
    attrInfo.groups     = 1;
    attrInfo.roundMode  = 0;
    attrInfo.dualOutput = 0;
    return DEMO_RET_SUC;
}

static int32_t InitPlatformInfo(optiling::conv_ops_tiling::ConvAscendcPlatformInfo& platformInfo)
{
    platform_ascendc::PlatformAscendC* ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr) {
        return DEMO_RET_FAIL;
    }

    platformInfo.aicoreNum = ascendcPlatform->GetCoreNumAic();
    uint64_t size {};
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L1, size);
    platformInfo.l1Size = size;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, size);
    platformInfo.l0aSize = size;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, size);
    platformInfo.l0bSize = size;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, size);
    platformInfo.l0cSize = size;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, size);
    platformInfo.ubSize = size;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::BT, size);
    platformInfo.btSize = size;
    uint64_t bwSize = 0;
    ascendcPlatform->GetCoreMemBw(platform_ascendc::CoreMemType::L2, bwSize);
    platformInfo.l2Rate = bwSize;
    platformInfo.npuArch = NpuArch::DAV_3510;

    return DEMO_RET_SUC;
}

static int32_t InitNodeInfo(optiling::conv_ops_tiling::ConvAscendcNodeInfo& nodeInfo)
{
    nodeInfo.nodeName = "conv3d_v2";
    nodeInfo.nodeType = "conv3d_v2";
    return DEMO_RET_SUC;
}

template <typename T>
void Conv3dV2CustomApi(
    aclrtStream stream, const at::Tensor& input, const at::Tensor& weight, const torch::Tensor& output,
    const vector<int64_t> strideList, const vector<int64_t> paddingList, const vector<int64_t> dilationList,
    const vector<int64_t> oriInputShapeList, const vector<int64_t> oriWeightShapeList,
    const c10::optional<torch::Tensor>& bias)
{
    vector<int64_t> oriOutputShapeList = {
        output.size(FORMAT_NCDHW_N_INDEX), output.size(FORMAT_NCDHW_C_INDEX), output.size(FORMAT_NCDHW_D_INDEX),
        output.size(FORMAT_NCDHW_H_INDEX), output.size(FORMAT_NCDHW_W_INDEX)};

    __gm__ uint8_t* input_ptr = (__gm__ uint8_t*)input.data_ptr<T>();
    __gm__ uint8_t* weight_ptr = (__gm__ uint8_t*)weight.data_ptr<T>();
    __gm__ uint8_t* bias_ptr = nullptr;
    __gm__ uint8_t* output_ptr = (__gm__ uint8_t*)output.data_ptr<T>();

    optiling::conv_ops_tiling::ConvBaseDeci convBaseDeci {};
    optiling::conv_ops_tiling::ConvAscendcTilingInfo tilingInfo {};
    TORCH_CHECK(InitAttrInfo(strideList, paddingList, dilationList, tilingInfo.attrInfo) == 0,
                "Failed to initialize attribute information");
    TORCH_CHECK(InitPlatformInfo(tilingInfo.platformInfo) == 0,
                "Failed to initialize platform information");
    TORCH_CHECK(InitNodeInfo(tilingInfo.nodeInfo) == 0,
                "Failed to initialize node information");

    string inDtype;
    if constexpr (std::is_same_v<T, c10::Half>) {
        TORCH_CHECK(InitConvBase(oriInputShapeList, oriWeightShapeList, oriOutputShapeList, tilingInfo,
                    ge::DataType::DT_FLOAT16) == 0,
                    "Failed to initialize convolution base configuration");
        inDtype = "float16";
    } else if constexpr (std::is_same_v<T, c10::BFloat16>) {
        TORCH_CHECK(InitConvBase(oriInputShapeList, oriWeightShapeList, oriOutputShapeList, tilingInfo,
                    ge::DataType::DT_BF16) == 0,
                    "Failed to initialize convolution base configuration with BF16 data type");
        inDtype = "bfloat16";
    } else if constexpr (std::is_same_v<T, float>) {
        TORCH_CHECK(InitConvBase(oriInputShapeList, oriWeightShapeList, oriOutputShapeList, tilingInfo,
                    ge::DataType::DT_FLOAT) == 0,
                    "Failed to initialize convolution base configuration with FP32 data type");
        inDtype = "float";
    } else {
        TORCH_CHECK(false, "Unsupported data type: only FP16, BF16, and FP32 are supported");
    }
    if (bias.has_value()) {
        bias_ptr = (__gm__ uint8_t*)bias.value().data_ptr<T>();
        tilingInfo.flagInfo.hasBias = static_cast<bool>(bias.has_value());
    }
    TORCH_CHECK(convBaseDeci.GetBlockDimInfo(tilingInfo) == 0,
        "Failed to get block dimension information from convolution base decision");

    Ops::NN::Conv3dV2::Conv3DV2TilingData tilingData;
    InitTilingData(tilingData, tilingInfo);

    conv_tiling::PlatformInfo platform;
    platform.npuArch = NpuArch::DAV_3510;
    platform.l1Size = tilingInfo.platformInfo.l1Size;
    platform.l0ASize = tilingInfo.platformInfo.l0aSize;
    platform.l0BSize = tilingInfo.platformInfo.l0bSize;
    platform.l0CSize = tilingInfo.platformInfo.l0cSize;
    platform.ubSize = tilingInfo.platformInfo.ubSize;
    platform.btSize = tilingInfo.platformInfo.btSize;
    conv_tiling::Conv3dTiling conv3dApiTiling(platform);
    conv3dApiTiling.GetTilingData(tilingInfo.attrInfo, tilingInfo.descInfo, tilingInfo.flagInfo, tilingInfo.shapeInfo,
        tilingInfo.convOpsConstParams, tilingInfo.blockDimRes, tilingData);

    uint32_t g_blockDim = tilingData.conv3dRunInfo.batchDim * tilingData.conv3dRunInfo.doDim *
                        tilingData.conv3dRunInfo.hoDim * tilingData.conv3dRunInfo.nDim;
    optiling::conv_ops_tiling::ConvTilingKeyPara tilingKeyPara {};
    optiling::conv_ops_tiling::Conv3dV2BaseTilingKey tilingKey {tilingData, tilingInfo.flagInfo,
        tilingInfo.descInfo, tilingInfo.shapeInfo, tilingInfo.blockDimRes, tilingInfo.convOpsConstParams};
    tilingKey.GetTemplateTilingKey(tilingKeyPara);

    Conv3dv2Template(input_ptr, weight_ptr, bias_ptr, nullptr, nullptr, nullptr,
        output_ptr, nullptr, tilingData,
        static_cast<int8_t>(tilingKeyPara.fmpTiling), static_cast<int8_t>(tilingKeyPara.weightTiling),
        static_cast<int8_t>(tilingKeyPara.l1PingPong), static_cast<int8_t>(tilingKeyPara.l0PingPong),
        static_cast<int8_t>(tilingKeyPara.outputOrder), static_cast<int8_t>(tilingKeyPara.iterOrder),
        inDtype, g_blockDim, stream);
}

torch::Tensor CreateOutputTensor(
    const torch::Tensor& input, const c10::IntArrayRef& oriInputShape, const c10::IntArrayRef& oriWeightShape,
    const c10::IntArrayRef& strides, const c10::IntArrayRef& pads, const c10::IntArrayRef& dilations)
{
    int64_t N = oriInputShape[FORMAT_NCDHW_N_INDEX];
    int64_t D = oriInputShape[FORMAT_NCDHW_D_INDEX];
    int64_t H = oriInputShape[FORMAT_NCDHW_H_INDEX];
    int64_t W = oriInputShape[FORMAT_NCDHW_W_INDEX];
    int64_t Co = oriWeightShape[FORMAT_NCDHW_N_INDEX];
    vector<int64_t> kernelSize = {
        oriWeightShape[FORMAT_NCDHW_D_INDEX], oriWeightShape[FORMAT_NCDHW_H_INDEX], oriWeightShape[FORMAT_NCDHW_W_INDEX]};

    int64_t Do = (D + 2 * pads[ATTRS_D_DIM_IDX_NCDHW] -
                  dilations[ATTRS_D_DIM_IDX_NCDHW] * (kernelSize[ATTRS_D_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_D_DIM_IDX_NCDHW] +
                 1;
    int64_t Ho = (H + 2 * pads[ATTRS_H_DIM_IDX_NCDHW] -
                  dilations[ATTRS_H_DIM_IDX_NCDHW] * (kernelSize[ATTRS_H_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_H_DIM_IDX_NCDHW] +
                 1;
    int64_t Wo = (W + 2 * pads[ATTRS_W_DIM_IDX_NCDHW] -
                  dilations[ATTRS_W_DIM_IDX_NCDHW] * (kernelSize[ATTRS_W_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_W_DIM_IDX_NCDHW] +
                 1;

    TORCH_CHECK(N > 0, "Output of N has to be positive, but got ", N);
    TORCH_CHECK(Co > 0, "Output of C has to be positive, but got ", Co);
    TORCH_CHECK(Do > 0, "Output of D has to be positive, but got ", Do);
    TORCH_CHECK(Ho > 0, "Output of H has to be positive, but got ", Ho);
    TORCH_CHECK(Wo > 0, "Output of W has to be positive, but got ", Wo);
    printf("Output shape: N=%d, Co=%d, Do=%d, Ho=%d, Wo=%d\n",
       static_cast<int>(N),
       static_cast<int>(Co),
       static_cast<int>(Do),
       static_cast<int>(Ho),
       static_cast<int>(Wo));
    return torch::zeros({N, Co, Do, Ho, Wo}, input.options());
}

torch::Tensor Conv3dV2CustomNpu(
    const torch::Tensor& input, const torch::Tensor& weight, const c10::IntArrayRef& strides,
    const c10::IntArrayRef& pads, const c10::IntArrayRef& dilations, const c10::IntArrayRef& oriInputShape,
    const c10::IntArrayRef& oriWeightShape, const c10::optional<torch::Tensor>& bias)
{
    TORCH_CHECK(torch_npu::utils::is_npu(input), "Input tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(weight), "Weight tensor must be on NPU device");
    if (bias.has_value()) {
        TORCH_CHECK(torch_npu::utils::is_npu(bias.value()), "Bias tensor must be on NPU device");
    }
    TORCH_CHECK(input.dim() == NCDHW_DIM, "Input must be 5D (N, C, D, H, W)");
    TORCH_CHECK(weight.dim() == NCDHW_DIM, "Weight must be 5D (N, C, D, H, W)");
    TORCH_CHECK(strides.size() == CONV_ATTRS_DIM, "Stride must have 3 elements");
    TORCH_CHECK(pads.size() == CONV_ATTRS_DIM, "Padding must have 3 elements");
    TORCH_CHECK(dilations.size() == CONV_ATTRS_DIM, "Dilation must have 3 elements");
    TORCH_CHECK(oriInputShape.size() == NCDHW_DIM, "Origin Input must have 5 elements (N, C, D, H, W)");
    TORCH_CHECK(oriWeightShape.size() == NCDHW_DIM, "Origin Weight must have 5 elements (N, C, D, H, W)");

    auto output = CreateOutputTensor(input, oriInputShape, oriWeightShape, strides, pads, dilations);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    vector<int64_t> oriInputShapeList(oriInputShape.begin(), oriInputShape.end());
    vector<int64_t> oriWeightShapeList(oriWeightShape.begin(), oriWeightShape.end());
    vector<int64_t> strideList(strides.begin(), strides.end());
    vector<int64_t> paddingList = {pads[ATTRS_D_DIM_IDX_NCDHW], pads[ATTRS_D_DIM_IDX_NCDHW],
                                   pads[ATTRS_H_DIM_IDX_NCDHW], pads[ATTRS_H_DIM_IDX_NCDHW],
                                   pads[ATTRS_W_DIM_IDX_NCDHW], pads[ATTRS_W_DIM_IDX_NCDHW]};
    vector<int64_t> dilationList(dilations.begin(), dilations.end());
    auto acl_call = [=]() -> int {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "Conv3dV2CustomNpu", [&] {
            Conv3dV2CustomApi<scalar_t>(
                stream, input, weight, output, strideList, paddingList, dilationList, oriInputShapeList,
                oriWeightShapeList, bias);
        });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Conv3dv2", acl_call);
    return output;
}

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "conv3d_v2_custom(Tensor input, Tensor weight, int[3] stride, int[3] padding, int[3] dilation, int[5] "
        "oriInputShape, int[5] oriWeightShape, Tensor? bias) -> Tensor");
}

// Register Ascend implementations for conv3d_v2
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("conv3d_v2_custom", Conv3dV2CustomNpu);
}

}
}