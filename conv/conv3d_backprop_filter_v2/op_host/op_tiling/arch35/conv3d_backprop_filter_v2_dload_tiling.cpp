/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file conv3d_backprop_filter_v2_dload_tiling.cpp
 * \brief
 */

#ifndef CONV3D_BACKPROP_FILTER_V2_DLOAD_TILING_CPP
#define CONV3D_BACKPROP_FILTER_V2_DLOAD_TILING_CPP

#include "conv/conv3d_backprop_filter_v2/op_kernel/arch35/conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_key.h"
#include "op_host/tiling_templates_registry.h"
#include "error_util.h"
#include "op_host/util/math_util.h"
#include "conv3d_backprop_filter_v2_dload_tiling.h"

namespace Ops {
namespace NN {
namespace Conv {
namespace {

struct DLoadShapeCase {
    int32_t batch;
    int32_t co;   // dy C（= filter co）
    int32_t dout; // dy D
    int32_t ci;   // fmap C（= filter ci）
    int32_t di;   // fmap D
    int32_t dhw;  // 立方档（3³/6³：fmap dy 同 D/H/W）
};
// dout/ho/wo = di/hi/wi 对每 case 恒等（3³ 或 6³ 立方），kd/kh/kw=3
constexpr DLoadShapeCase DLOAD_SUPPORT_SHAPE[6] = {
    {8, 384, 6, 384, 6, 6}, // 0020: dy[8,384,6,6,6] fmap[8,384,6,6,6] f[384,384,3,3,3]
    {8, 384, 6, 768, 6, 6}, // 0021: dy[8,384,6,6,6] fmap[8,768,6,6,6] f[384,768,3,3,3]
    {8, 768, 3, 768, 3, 3}, // 0023: dy[8,768,3,3,3] fmap[8,768,3,3,3] f[768,768,3,3,3]
    {4, 384, 6, 384, 6, 6}, // 0041: dy[4,384,6,6,6] fmap[4,384,6,6,6] f[384,384,3,3,3]
    {4, 384, 6, 768, 6, 6}, // 0042: dy[4,384,6,6,6] fmap[4,768,6,6,6] f[384,768,3,3,3]
    {4, 768, 3, 768, 3, 3}, // 0044: dy[4,768,3,3,3] fmap[4,768,3,3,3] f[768,768,3,3,3]
};
} // namespace

bool Conv3DBackpropFilterV2DLoadTiling::CheckFormat()
{
    constexpr size_t Y_INDEX = 2;
    constexpr size_t FILTER_INDEX = 0;
    constexpr size_t OUTPUT_BP_INDEX = 0;

    const auto fmapDesc = context_->GetInputDesc(OUTPUT_BP_INDEX);
    OP_TILING_CHECK(fmapDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "fmap_desc is null"),
                    return false);
    auto fmapFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(fmapDesc->GetStorageFormat()));
    const auto dedyDesc = context_->GetInputDesc(Y_INDEX);
    OP_TILING_CHECK(dedyDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "dedyDesc is null"),
                    return false);
    auto dedyFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(dedyDesc->GetStorageFormat()));
    const auto filterDesc = context_->GetOutputDesc(FILTER_INDEX);
    OP_TILING_CHECK(filterDesc == nullptr, CUBE_INNER_ERR_REPORT("Conv3DBackpropFilterV2", "filterDesc is null"),
                    return false);
    auto filterFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(filterDesc->GetStorageFormat()));

    return fmapFormat == ge::FORMAT_NCDHW && dedyFormat == ge::FORMAT_NCDHW && filterFormat == ge::FORMAT_NCDHW;
}

bool Conv3DBackpropFilterV2DLoadTiling::CheckDLoadDtype()
{
    // 仅放通 hf32
    if (runInfo_.a_dtype != ge::DataType::DT_FLOAT || runInfo_.b_dtype != ge::DataType::DT_FLOAT ||
        runInfo_.c_dtype != ge::DataType::DT_FLOAT) {
        OP_LOGD(opName_, "DLoad tiling only support float");
        return false;
    }
    if (runInfo_.hf32Flag != 1) {
        OP_LOGD(opName_, "DLoad tiling only support hf32");
        return false;
    }
    return true;
}

bool Conv3DBackpropFilterV2DLoadTiling::CheckDLoadAttrs()
{
    // kernel 3³、stride/dilation dhw 全 1、pad 六值全 1、groups=1
    if (runInfo_.kd != DLOAD_KERNEL_SIZE_3 || runInfo_.kh != DLOAD_KERNEL_SIZE_3 ||
        runInfo_.kw != DLOAD_KERNEL_SIZE_3) {
        OP_LOGD(opName_, "DLoad tiling only support 3*3*3 kernel");
        return false;
    }
    if (runInfo_.stride_d != 1 || runInfo_.stride_h != 1 || runInfo_.stride_w != 1 || runInfo_.dilation_d != 1 ||
        runInfo_.dilation_h != 1 || runInfo_.dilation_w != 1) {
        OP_LOGD(opName_, "DLoad tiling only support stride/dilation 1");
        return false;
    }
    if (runInfo_.pad_f != 1 || runInfo_.pad_b != 1 || runInfo_.pad_u != 1 || runInfo_.pad_d != 1 ||
        runInfo_.pad_l != 1 || runInfo_.pad_r != 1) {
        OP_LOGD(opName_, "DLoad tiling only support pad 1");
        return false;
    }
    if (runInfo_.groups != 1) {
        OP_LOGD(opName_, "DLoad tiling only support groups 1");
        return false;
    }
    return true;
}

bool Conv3DBackpropFilterV2DLoadTiling::CheckShape()
{
    // shape 全等比对（fmap=(batch,ci,di,hi,wi)、dy=(batch,co,dout,ho,wo)、filter=(co,ci,kd,kh,kw)——
    // 表内 case 均立方（hi=wi=di、ho=wo=dout），dout 与 di 同值表内给）
    for (const auto& c : DLOAD_SUPPORT_SHAPE) {
        // dy=(batch,co,dout,ho,wo)（表第 1 列）、fmap=(batch,ci,di,hi,wi)（第 2 列），
        // 立方档 dout=ho=wo=di=hi=wi=dhw；filter=(co,ci,3,3,3) 由 co/ci 等值蕴含
        const bool dyEq = runInfo_.batch == c.batch && runInfo_.co == c.co && runInfo_.dout == c.dout &&
                          runInfo_.ho == c.dout && runInfo_.wo == c.dout;
        const bool fmapEq = runInfo_.ci == c.ci && runInfo_.di == c.di && runInfo_.hi == c.di && runInfo_.wi == c.di;
        if (dyEq && fmapEq) {
            return true;
        }
    }
    OP_LOGD(opName_, "DLoad tiling shape mismatch");
    return false;
}

bool Conv3DBackpropFilterV2DLoadTiling::IsCapable()
{
    if (!IsSocVersion91095()) {
        return false;
    }
    if (!CheckFormat()) {
        OP_LOGD(opName_, "current format is not support by DLoad tiling");
        return false;
    }
    if (!CheckDLoadDtype()) {
        return false;
    }
    if (!CheckDLoadAttrs()) {
        return false;
    }
    if (!CheckShape()) {
        return false;
    }
    return true;
}

uint64_t Conv3DBackpropFilterV2DLoadTiling::GetTilingKey() const
{
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TPL_DLOAD, 0, 0, TPL_WINOGRAD_DISABLE, 0);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu] , use DLoad tiling", tilingKey);
    return tilingKey;
}

ge::graphStatus Conv3DBackpropFilterV2DLoadTiling::DoOpTiling()
{
    // shape/attr 直传；块档参数写 blockTiling_ 工作变量，由基类 DoLibApiTiling 统一
    // 提交 dwTiling（含派生与打印）
    SetShapeTiling(tilingData_.dwTiling);
    SetAttrTiling(tilingData_.dwTiling);

    // 固定档：baseK=16（kl0HoWo howo 窗宽）、baseM=128（cout 块宽）、
    // baseN=144（=16 cin × hkwk，mmad N 轴——入口反解 cin=baseN/hkwk=16）
    blockTiling_.blockBaseM = DLOAD_BASE_M;
    blockTiling_.blockBaseN = DLOAD_BASE_N_PER_HKWK * runInfo_.kh * runInfo_.kw;
    blockTiling_.blockBaseK = DLOAD_BASE_K;
    blockTiling_.usedCoreNum = platformInfo_.core_num;
    blockTiling_.singleCoreM = DLOAD_BASE_M;
    blockTiling_.singleCoreN = blockTiling_.blockBaseN;
    blockTiling_.singleCoreK = static_cast<uint64_t>(DLOAD_BASE_K);
    blockTiling_.singleCoreBatchDout = 1;
    blockTiling_.streamkType = 0; // 非 streamK 路径（kernel 侧不读，语义占位防默认 1）
    blockTiling_.splitWo = runInfo_.wo;
    blockTiling_.splitWi = 1;
    blockTiling_.tailWo = 0;
    blockTiling_.tailWi = 0;
    // hf32Flag 显式写（基类 DoOpTiling 同款 runInfo 直传，本类 override 后须自写；
    // SetAttrTiling 只覆盖 stride/pad/dilation/realGroup）；singleCoreCin/Ho/Cout 为
    // DoLibApiTiling 派生量（baseN/hkwk/C0 与 singleCoreK/wo）
    tilingData_.dwTiling.hf32Flag = runInfo_.hf32Flag; // 支持场景恒 1（cube_math_type==3）
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DBackpropFilterV2DLoadTiling::GetWorkspaceSize()
{
    // DLoad 无 NC1HWC0 预转置 / 无切 k 尾块暂存（与 winograd 的结构差异）——workspace 0
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropFilterV2", Conv3DBackpropFilterV2DLoadTiling, 8);
} // namespace Conv
} // namespace NN
} // namespace Ops
#endif // CONV3D_BACKPROP_FILTER_V2_DLOAD_TILING_CPP
