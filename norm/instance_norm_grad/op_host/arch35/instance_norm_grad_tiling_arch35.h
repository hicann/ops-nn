/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file instance_norm_grad_tiling_arch35.h
 * \brief RegBase (arch35) tiling class for InstanceNormGrad.
 */
#pragma once

#include "instance_norm_grad_tiling.h"
using namespace Ops::NN::Optiling;

namespace optiling {
class InstanceNormGradRegBaseTiling : public TilingBaseClass {
public:
    explicit InstanceNormGradRegBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus InputCheck();
    ge::graphStatus ParamsCheck();
    ge::graphStatus CheckTensorDtype(const gert::CompileTimeTensorDesc* desc, const char* name) const;
    ge::graphStatus BlockTiling();
    ge::graphStatus UbTiling();
    void SetTilingData();
    void PrintTilingData() const;
    uint32_t GetTypeSize(ge::DataType dtypeStr) const;

private:
    static constexpr uint32_t DOUBLE_BUFFER = 2;
    static constexpr uint32_t UB_COPIES_3 = 3; // x, dy, pd_x flowing buffers
    // stage1 中按 fp32 参数长度分配的缓冲个数,当前 11 个:
    //   mean, gamma, rstd, pdVar, pdMean, accDgamma, accDbeta,
    //   cDgamma, cDbeta, cPdVar, cPdMean(后四个为 Kahan 补偿,跨 M-tile 持久化)
    // var 不单独占缓冲:载入后直接在 rstdBuf 上原地转成 rstd。
    // 另有 1 份输入 dtype 的临时缓冲 tmpParamBuf,按 tTypeBytes_ 单独计(见 Stage1ParamBytes)。
    // 缓冲字节数由 host 算定后经 tilingData 下发,内核不再自行推导尺寸。
    static constexpr uint32_t PARAM_BUFFERS = 11;
    // 必须与 op_kernel/arch35/instance_norm_grad_base.h 里 Stage2Process 的 float 缓冲个数一致。
    // 当前 6 个:s2InQue(双缓冲计 2)、accDg、accDb、cDg、cDb(后两个为跨 N 合并的 Kahan 补偿);
    // 另有 1 份输出 dtype 缓冲 s2OutBuf,按 tTypeBytes_ 单独计。
    static constexpr uint32_t STAGE2_BUFFERS_F32 = 6;
    static constexpr uint32_t WORKSPACE_COPIES = 2; // dgamma + dbeta partial sums
    static constexpr uint32_t FLOAT_DTYPE_BYTES = 4;
    static constexpr uint32_t FLOAT16_DTYPE_BYTES = 2;
    static constexpr int64_t MODE_FULL_LOAD = 100;
    static constexpr int64_t MODE_RECOMPUTE = 300;
    static constexpr uint32_t MIN_BLOCK_SIZE = 512;

    const char* opName = "InstanceNormGrad";
    InstanceNormGradTilingData tilingData;

    uint64_t ubSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t sysWorkspaceSize_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t vectorLen_ = 0;         // fp32 lanes per VL
    uint32_t paramBufBytes_ = 0;     // 下发:每个 fp32 参数缓冲字节数
    uint32_t tmpParamBufBytes_ = 0;  // 下发:输入 dtype 临时参数缓冲字节数
    uint32_t tileBytes_ = 0;         // 下发:每个流水缓冲字节数
    uint32_t stage2BufBytes_ = 0;    // 下发:stage2 每个 fp32 缓冲字节数
    uint32_t stage2OutBufBytes_ = 0; // 下发:stage2 输出 dtype 缓冲字节数

    // 下面两个函数是「内核实际会分配多少」的唯一口径,与 op_kernel/arch35/instance_norm_grad_base.h
    // 的 InitStage1Buffers 一一对应:C 对齐到向量长度(不是 block),流水缓冲额外补一个向量长度。
    int64_t Stage1ParamBytes(int64_t cTile) const;
    int64_t FlowTileBytes(int64_t cTile, int64_t mRows) const;

    ge::DataType dtype_ = ge::DT_UNDEFINED;
    uint32_t tTypeBytes_ = 0;

    int64_t N_ = 0;
    int64_t C_ = 0;
    int64_t M_ = 1;
    int64_t cTile_ = 0;
    int64_t cTileNum_ = 1;
    int64_t taskNum_ = 0;
    uint32_t taskNumPerCore_ = 0;
    uint32_t taskNumPerTailCore_ = 0;
    uint32_t tailCore_ = 0;
    uint32_t stage1CoreUsed_ = 0;
    uint32_t modeKey_ = MODE_FULL_LOAD;
    uint32_t mUbTile_ = 0;
    uint32_t mUbIterNum_ = 1;
    uint32_t mUbTailNum_ = 0;

    int64_t reduceNCnt_ = 0;
    int64_t workSpaceSize_ = 0;
    uint32_t stage2CoreUsed_ = 0;
    int64_t cBlockFactor_ = 0;
    int64_t cTailBlockFactor_ = 0;
    uint32_t stage2SubCap_ = 0; // stage2 每轮处理的通道数,由 ubSize_ 算出
};
} // namespace optiling
