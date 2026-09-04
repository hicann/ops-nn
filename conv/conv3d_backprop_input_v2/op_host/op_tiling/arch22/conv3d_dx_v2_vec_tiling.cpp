/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_dx_v2_vec_tiling.cpp
 * \brief Conv3D backprop input Vector mode tiling implementation.
 *        Registered at priority 2 as an independent template.
 *        Capability: FP16 + ND format filter, non-arch3510.
 *        Unlike Cube path, this template does NOT call GetTbeTiling;
 *        it directly computes tiling via ComputeVecTiling().
 */

#include <algorithm>
#include <log/log.h>
#include <register/op_impl_registry.h>
#include "conv3d_dx_v2_vec_tiling.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"

namespace {
constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t B16_BITS = 4;
constexpr uint32_t FP32_BITS = 3;
const size_t FILTER_INDEX = 1;
const size_t OUT_BACKPROP_INDEX = 2;
} // namespace

namespace Ops {
namespace NN {
namespace Conv {

bool Conv3DDXV2VecTiling::IsCapable()
{
    auto compileInfo = context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>();
    if (compileInfo == nullptr) {
        return false;
    }
    // Vec 模板仅服务于 910B/910_93（npuArch 2201），与 aclnn IsConv3DVecFallbackCase
    // 及二进制部署范围（AddConfig ascend910b/ascend910_93）保持一致；其余代际一律不介入
    // （arch3510 拥有独立的 priority 96~101 tiling 模板）
    if (compileInfo->npuArch != NpuArch::DAV_2201) {
        return false;
    }

    // Vec 模式适用条件：FP16/BF16/FP32 数据类型 + NCDHW 格式 filter。
    // vecDtype_ 已在 GetShapeAttrsInfo() 中从原始输入 desc 读取，避免 BF16 被
    // SetRunInfoToV2 归一到 FP16 后无法与 FP16 区分。
    bool isVecDtype = (vecDtype_ == ge::DT_FLOAT16 || vecDtype_ == ge::DT_BF16 || vecDtype_ == ge::DT_FLOAT);
    auto filterDesc = context_->GetInputDesc(FILTER_INDEX);
    bool isNdFormat = (runInfo_.filterFormat == ge::FORMAT_NCDHW && filterDesc != nullptr &&
                       filterDesc->GetStorageFormat() == ge::FORMAT_NCDHW);
    return isVecDtype && isNdFormat;
}

ge::graphStatus Conv3DDXV2VecTiling::GetShapeAttrsInfo()
{
    // arch3510 直接返回，IsCapable 会返回 false 跳过本模板
    auto compileInfo = context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>();
    if (compileInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "GetCompileInfo returned nullptr in VecTiling");
        return ge::GRAPH_FAILED;
    }
    if (compileInfo->npuArch == NpuArch::DAV_3510) {
        return ge::GRAPH_SUCCESS;
    }

    // 复用基类的 dtype 校验逻辑
    if (Conv3DBackpropInputV2Tiling::GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 填充 runInfo_，供 IsCapable 和 ComputeVecTiling 使用
    // 注意：Vec 模式不调用 GetTbeTiling，无需 Cube 的 tiling 参数
    if (!SetRunInfoToV2(context_, runInfo_, opType_)) {
        OP_LOGE(context_->GetNodeName(), "SetRunInfoToV2 failed in VecTiling");
        return ge::GRAPH_FAILED;
    }

    blockSize_ = BYTE_BLOCK / runInfo_.a_dtype_bytes;
    dtypeByte_ = runInfo_.a_dtype_bytes;

    // 记录原始 out_backprop dtype，用于 kernel 类型分发和 c0Bits 选择。
    auto outBackpropDesc = context_->GetInputDesc(OUT_BACKPROP_INDEX);
    vecDtype_ = (outBackpropDesc != nullptr) ? outBackpropDesc->GetDataType() : ge::DT_FLOAT16;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2VecTiling::DoOpTiling()
{
    // Vec 模式无需 Cube 的 tiling 计算，直接置标志
    // loadB2Condition_ = 0 保证 Tiling Key 合法
    useVecMode_ = true;
    loadB2Condition_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2VecTiling::DoLibApiTiling()
{
    // ComputeVecTiling 会设置 useVecMode_=true、loadB2Condition_=0
    // 并填充 tilingData_ 的所有 Vec 专用字段
    return ComputeVecTiling();
}

void Conv3DDXV2VecTiling::FillShapeFields()
{
    auto& dx = tilingData_.conv3DDxTiling;
    // 阶段A：从 runInfo_ 填充基础 shape 字段
    dx.batch = runInfo_.batch_n;
    dx.cin = runInfo_.dedx_cin;
    dx.cout = runInfo_.dedy_cout;
    dx.cout1 = runInfo_.dedy_cout1;
    dx.cin1 = runInfo_.dedx_cin1;
    dx.cout1G = runInfo_.dedy_cout1_g;
    dx.cin1G = runInfo_.dedx_cin1_g;
    dx.c0 = blockSize_;
    dx.c0Bits = (dtypeByte_ == static_cast<uint32_t>(ge::GetSizeByDataType(ge::DT_FLOAT))) ? FP32_BITS : B16_BITS;
    dx.vecDtype = static_cast<uint8_t>(
        vecDtype_ == ge::DT_BF16 ? Conv3DDxVecDtype::BF16 :
                                   (vecDtype_ == ge::DT_FLOAT ? Conv3DDxVecDtype::FP32 : Conv3DDxVecDtype::FP16));
    dx.ho = runInfo_.dedy_h;
    dx.wo = runInfo_.dedy_w;
    dx.dout = runInfo_.dedy_d;
    dx.di = runInfo_.dedx_d;
    dx.hi = runInfo_.dedx_h;
    dx.wi = runInfo_.dedx_w;
    dx.hk = runInfo_.kernel_h;
    dx.wk = runInfo_.kernel_w;
    dx.dk = runInfo_.kernel_d;

    dx.group = runInfo_.groups; // vec kernel 按真实 groups 做 weight 寻址；real_g 会因 multiple_extend 合并分组
    OP_LOGI(opName_, "vec dx tiling: groups=%d, real_g=%d, cin=%d, cout=%d", runInfo_.groups, runInfo_.real_g,
            runInfo_.dedx_cin, runInfo_.dedy_cout);
    dx.strideH = runInfo_.stride_h;
    dx.strideW = runInfo_.stride_w;
    dx.strideD = runInfo_.stride_d;
    dx.padFront = runInfo_.pad_h;
    dx.padBack = runInfo_.pad_t;
    dx.padUp = runInfo_.pad_u;
    dx.padDown = runInfo_.pad_d;
    dx.padLeft = runInfo_.pad_l;
    dx.padRight = runInfo_.pad_r;
    dx.dilationH = runInfo_.dilation_h;
    dx.dilationW = runInfo_.dilation_w;
    dx.dilationD = runInfo_.dilation_d;
}

void Conv3DDXV2VecTiling::CalcReversePadding()
{
    auto& dx = tilingData_.conv3DDxTiling;
    // 阶段B：计算反向卷积 padding
    int32_t kernelD = static_cast<int32_t>(dx.dk);
    int32_t kernelH = static_cast<int32_t>(dx.hk);
    int32_t kernelW = static_cast<int32_t>(dx.wk);
    int32_t dilationD = static_cast<int32_t>(dx.dilationD);
    int32_t dilationH = static_cast<int32_t>(dx.dilationH);
    int32_t dilationW = static_cast<int32_t>(dx.dilationW);
    int32_t strideD = static_cast<int32_t>(dx.strideD);
    int32_t strideH = static_cast<int32_t>(dx.strideH);
    int32_t strideW = static_cast<int32_t>(dx.strideW);
    int32_t padFront = static_cast<int32_t>(dx.padFront);
    int32_t padUp = static_cast<int32_t>(dx.padUp);
    int32_t padLeft = static_cast<int32_t>(dx.padLeft);

    dx.padHDx = (kernelD - 1) * dilationD - padFront;
    dx.padUDx = (kernelH - 1) * dilationH - padUp;
    dx.padLDx = (kernelW - 1) * dilationW - padLeft;

    dx.dilatedHk = static_cast<uint32_t>((kernelH - 1) * dilationH + 1);
    dx.dilatedWk = static_cast<uint32_t>((kernelW - 1) * dilationW + 1);
}

ge::graphStatus Conv3DDXV2VecTiling::CalcUbStrategy()
{
    auto& dx = tilingData_.conv3DDxTiling;
    // 阶段C：对齐参数 + UB 策略
    dx.dtypeBytes = static_cast<uint8_t>(dtypeByte_);
    dx.dataPerBlock = 32 / dtypeByte_;
    dx.alignedDilatedW = ((dx.dilatedWk + dx.dataPerBlock - 1) / dx.dataPerBlock) * dx.dataPerBlock;

    auto platform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    CalcB16UbBudget(ubSize);
    CalcFp32UbBudget(ubSize);
    return CheckWeightUbBudget(ubSize);
}

void Conv3DDXV2VecTiling::CalcB16UbBudget(uint64_t ubSize)
{
    auto& dx = tilingData_.conv3DDxTiling;
    // BF16/FP16 vector path UB budget check（与 kernel InitBuffer 一致的口径）：
    //   行缓冲 rowAcc/prod(FP32) + gradRow/outH(BF16) = alignedWi*12B；
    //   strideW>1 相位分解路径只分配类缓冲（≈alignedWi/strideW*12B）；
    //   weightDilated 按 dilatedHk*alignedDilatedW（不能用 hk，dilationH>1 会低估）。
    // 超限不再 GRAPH_FAILED：降级为标量路径（vecScalarOnly=1，kernel 只保留
    // weightDilated 缓冲），保证大 shape 用例可正常执行而不在 tiling 处被拦截。
    if (vecDtype_ == ge::DT_BF16 || vecDtype_ == ge::DT_FLOAT16) {
        const uint32_t alignedWi = ((static_cast<uint32_t>(dx.wi) + dx.dataPerBlock - 1) / dx.dataPerBlock) *
                                   dx.dataPerBlock;
        const uint32_t sW = static_cast<uint32_t>(dx.strideW > 1 ? dx.strideW : 1);
        const uint32_t classAlignedWi = (sW > 1) ?
                                            (((static_cast<uint32_t>(dx.wi) + sW - 1) / sW + dx.dataPerBlock - 1) /
                                             dx.dataPerBlock) *
                                                dx.dataPerBlock :
                                            0;
        uint64_t bf16UbNeed = static_cast<uint64_t>(alignedWi) * 12 + static_cast<uint64_t>(classAlignedWi) * 12 +
                              static_cast<uint64_t>(dx.dilatedHk) * dx.alignedDilatedW * sizeof(uint16_t);
        if (bf16UbNeed > ubSize / 2) {
            OP_LOGI(opName_,
                    "vec BF16/FP16 row buffer %uKB exceeds UB budget %uKB (wi=%u, strideW=%u, "
                    "dilatedHk=%u): degrade to scalar-only path",
                    static_cast<uint32_t>(bf16UbNeed / 1024), static_cast<uint32_t>((ubSize / 2) / 1024), dx.wi,
                    dx.strideW, dx.dilatedHk);
            dx.vecScalarOnly = 1;
            // rowAcc+outH（按 alignedWi*8B 保守核算）+ weightDilated 放得下 UB 时，
            // 降级到 ComputeRowScalarAcc（FP32 累加，精度口径与向量路径一致）；
            // 极端大 wi 连 rowAcc+outH 都放不下时才退回逐点舍入的纯标量 ComputeRow。
            dx.useScalarAcc = (static_cast<uint64_t>(alignedWi) * 8 +
                                   static_cast<uint64_t>(dx.dilatedHk) * dx.alignedDilatedW * sizeof(uint16_t) <=
                               ubSize / 2) ?
                                  1 :
                                  0;
        }
    }
}

void Conv3DDXV2VecTiling::CalcFp32UbBudget(uint64_t ubSize)
{
    auto& dx = tilingData_.conv3DDxTiling;
    // FP32 vector path UB budget check（与 kernel InitBuffer 一致的口径）：
    //   向量快路径（CanUseBf16VecRow 同款条件）行缓冲 rowAcc/prod/gradRow/outH = alignedWi*16B；
    //   strideW>1 或 pad 超限时降级 ComputeRowScalarAcc，只需 rowAcc/outH = alignedWi*8B；
    //   weightDilated 按 dilatedHk*alignedDilatedW*4B。超限降级为纯标量 ComputeRow。
    if (vecDtype_ == ge::DT_FLOAT) {
        const uint32_t alignedWi = ((static_cast<uint32_t>(dx.wi) + dx.dataPerBlock - 1) / dx.dataPerBlock) *
                                   dx.dataPerBlock;
        const int64_t rightPadMax = static_cast<int64_t>(dx.wi) - static_cast<int64_t>(dx.wo) +
                                    (static_cast<int64_t>(dx.wk) - 1) * dx.dilationW - dx.padLDx;
        const bool canVecRow = dx.strideW == 1 && dx.padLDx >= 0 && dx.padLDx <= 255 && dx.dilatedWk <= 255 &&
                               dx.wi > 0 && dx.wo > 0 && static_cast<int64_t>(dx.padLDx) * sizeof(float) <= 32 &&
                               (rightPadMax <= 0 || rightPadMax * sizeof(float) <= 32) && alignedWi <= 0xFFFFU;
        uint64_t fp32UbNeed = static_cast<uint64_t>(alignedWi) * (canVecRow ? 16 : 8) +
                              static_cast<uint64_t>(dx.dilatedHk) * dx.alignedDilatedW * sizeof(float);
        if (fp32UbNeed > ubSize / 2) {
            OP_LOGI(opName_,
                    "vec FP32 row buffer %uKB exceeds UB budget %uKB (wi=%u, strideW=%u, "
                    "dilatedHk=%u): degrade to scalar-only path",
                    static_cast<uint32_t>(fp32UbNeed / 1024), static_cast<uint32_t>((ubSize / 2) / 1024), dx.wi,
                    dx.strideW, dx.dilatedHk);
            dx.vecScalarOnly = 1;
        }
    }
}

ge::graphStatus Conv3DDXV2VecTiling::CheckWeightUbBudget(uint64_t ubSize)
{
    auto& dx = tilingData_.conv3DDxTiling;
    // 兜底路径（vecScalarOnly=1 且 useScalarAcc=0）下 kernel 只分配 weightDilated：
    // 极端 kernel×dilation（dilatedHk*alignedDilatedW 超 UB 预算）会让 InitBuffer 越界，
    // 此处 fail-stop 提前暴露，比运行时 UB 溢出更可诊断。
    if (dx.vecScalarOnly != 0 && dx.useScalarAcc == 0) {
        const uint64_t weightUbNeed = static_cast<uint64_t>(dx.dilatedHk) * dx.alignedDilatedW * dtypeByte_;
        if (weightUbNeed > ubSize / 2) {
            OP_LOGE(opName_,
                    "vec weightDilated buffer %uKB exceeds UB budget %uKB (dilatedHk=%u, "
                    "alignedDilatedW=%u): fail-stop, cannot degrade safely",
                    static_cast<uint32_t>(weightUbNeed / 1024), static_cast<uint32_t>((ubSize / 2) / 1024),
                    dx.dilatedHk, dx.alignedDilatedW);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

void Conv3DDXV2VecTiling::SetupMultiCorePartition()
{
    auto& dx = tilingData_.conv3DDxTiling;
    // 阶段D：多核分区（Phase1）+ 重置 Cube 字段
    // 归约维(co,dk)完整保留在单元内，按 gradInput 的 (n,ci,di,hi) 行分区，无竞争
    int32_t archCores = context_->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->core_num;
    if (archCores <= 0) {
        archCores = 1;
    }
    const uint64_t totalRows = static_cast<uint64_t>(dx.batch) * dx.cin * dx.di * dx.hi;
    const uint32_t usedCores = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(archCores), std::max<uint64_t>(totalRows, 1ULL)));
    tilingData_.params.coreNum = usedCores;
    context_->SetBlockDim(usedCores);
    tilingData_.params.batchDim = 1;
    tilingData_.params.groupDim = 1;
    tilingData_.params.mDim = 1;
    tilingData_.params.kDim = 1;
    tilingData_.params.nDim = 1;
    tilingData_.params.dDim = 1;

    dx.baseM = 1;
    dx.baseK = 1;
    dx.baseN = 1;
    dx.baseD = 1;
    dx.baseBatch = 1;
    dx.baseGroup = 1;
    dx.stepM = 1;
    dx.stepN = 1;
    dx.stepKa = 1;
    dx.stepKb = 1;
    dx.stepBatch = 1;
    dx.stepGroup = 1;
    dx.al0Pbuffer = 1;
    dx.bl0Pbuffer = 1;
    dx.cl0Pbuffer = 1;
    dx.al1Pbuffer = 1;
    dx.bl1Pbuffer = 1;
    dx.iterateOrder = 0;
    dx.singleCoreBatch = dx.batch;
    dx.singleCoreCout = dx.cout;
    dx.singleCoreCout1 = dx.cout1;
    dx.singleCoreCin = dx.cin;
    dx.singleCoreCin1 = dx.cin1;
    dx.singleCoreDin = dx.di;
    dx.singleCoreM = static_cast<uint64_t>(dx.hi) * dx.wi;
    dx.singleCoreHo = 1;
}

ge::graphStatus Conv3DDXV2VecTiling::ComputeVecTiling()
{
    opName_ = context_->GetNodeName();
    auto& dx = tilingData_.conv3DDxTiling;

    FillShapeFields();
    CalcReversePadding();
    if (CalcUbStrategy() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetupMultiCorePartition();

    useVecMode_ = true;
    loadB2Condition_ = 0;

    OP_LOGD(opName_,
            "vec tiling computed: batch=%u, dout=%u, ho=%u, wo=%u, cin=%u, di=%u, hi=%u, wi=%u, "
            "dk=%u, hk=%u, wk=%u, padHDx=%d, padUDx=%d, padLDx=%d",
            dx.batch, dx.dout, dx.ho, dx.wo, dx.cin, dx.di, dx.hi, dx.wi, dx.dk, dx.hk, dx.wk, dx.padHDx, dx.padUDx,
            dx.padLDx);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3DDXV2VecTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    // Vec 模式 workspace 用于存放展开后的 gradOutput。
    // 注意：kernel 侧 GetUserWorkspace 返回的是 workspace 基址 + RESERVED_WORKSPACE(2201为16MB)，
    // 因此声明总量必须包含系统保留区，否则用户区指针越界（此前只声明净需求导致 AIV 写 workspace 即异常）。
    constexpr size_t kReservedWorkspace = 16 * 1024 * 1024; // 与 base tiling 的 WORKSIZE 惯例一致
    // workspace 存 dilatedWk 份错位副本（replica）：tap dw 读第 dw 份行首（对齐），
    // 错位在写入侧消化，向量读取无需 lane 偏移
    // vec 路径直接从 gradOutput GM 读取，无 workspace 展开；仅预留 RESERVED 供 GetUserWorkspace 返回非空
    size_t userNeed = 0;
    workspaces[0] = kReservedWorkspace + userNeed;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropInputV2", Conv3DDXV2VecTiling, 2);

} // namespace Conv
} // namespace NN
} // namespace Ops
