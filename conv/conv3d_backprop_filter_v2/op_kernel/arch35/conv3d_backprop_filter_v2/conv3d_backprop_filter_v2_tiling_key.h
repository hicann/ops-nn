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
 * \file conv3d_backprop_filter_v2_tiling_key.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_V2_TILING_KEY_ARCH35_H
#define CONV3D_BACKPROP_FILTER_V2_TILING_KEY_ARCH35_H

#include "ascendc/host_api/tiling/template_argument.h"

// 原两槽位（STREAM_K/MN_STREAM_K）合并由 host 统一编码为 STREAM_K，调度差异由运行期
// tilingData.streamkType 区分，kernel 分派同一实现。MN_STREAM_K 为历史保留槽位：host 不产出、
#define TPL_STREAM_K 1
#define TPL_MN_STREAM_K 2

// DLoad 模板（原 TPL_FMAP_RESIDENT 改名）。宏值不参与编码（取 LIST 位置索引），
// host/kernel 两侧 LIST 顺序必须保持一致。
#define TPL_DLOAD 3
#define TPL_WINOGRAD_DISABLE 0
#define TPL_WINOGRAD_SINGLE_SHAPE_TILE_1 1
#define TPL_WINOGRAD_SINGLE_SHAPE_TILE_2 2
#define TPL_WINOGRAD_RESIDENT_FMAP 0
#define TPL_WINOGRAD_RESIDENT_DY 1

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#define ASCENDC_TPL_MIX_AIC_VALUE ASCENDC_TPL_MIX_AIC_1_2
#else
#define ASCENDC_TPL_MIX_AIC_VALUE ASCENDC_TPL_MIX_AIC_1_1
#endif

// 模板参数
ASCENDC_TPL_ARGS_DECL(Conv3dBackPropFilterV2,
                      ASCENDC_TPL_UINT_DECL(conv3DDWTemplateId, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, TPL_STREAM_K,
                                            TPL_MN_STREAM_K, TPL_DLOAD), // LIST模式, 穷举；尾部 append 保序，
                      // MN_STREAM_K(索引1) 为历史保留槽位：不产出 key、不编译实例，仅为 DLoad 索引占位
                      ASCENDC_TPL_BOOL_DECL(isSplitKernelHW, 0, 1), ASCENDC_TPL_BOOL_DECL(groupEnlarge, 0, 1),
                      ASCENDC_TPL_UINT_DECL(winogradTilingFlag, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
                                            TPL_WINOGRAD_DISABLE, TPL_WINOGRAD_SINGLE_SHAPE_TILE_1,
                                            TPL_WINOGRAD_SINGLE_SHAPE_TILE_2),
                      ASCENDC_TPL_BOOL_DECL(winogradResidentFlag, 0, 1));

// 模板参数组合
// 用于调用GET_TPL_TILING_KEY获取TilingKey时，接口内部校验TilingKey是否合法
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, TPL_WINOGRAD_DISABLE),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 1), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, TPL_WINOGRAD_DISABLE),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)),

    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 1),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, TPL_WINOGRAD_DISABLE),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 1), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 1),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, TPL_WINOGRAD_DISABLE),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)),
    // winograd tiling key
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST,
                                              TPL_WINOGRAD_SINGLE_SHAPE_TILE_1),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, TPL_WINOGRAD_RESIDENT_FMAP)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST,
                                              TPL_WINOGRAD_SINGLE_SHAPE_TILE_2),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, TPL_WINOGRAD_RESIDENT_FMAP)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST,
                                              TPL_WINOGRAD_SINGLE_SHAPE_TILE_1),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, TPL_WINOGRAD_RESIDENT_DY)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_STREAM_K),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST,
                                              TPL_WINOGRAD_SINGLE_SHAPE_TILE_2),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, TPL_WINOGRAD_RESIDENT_DY)),
    // DLoad tiling key（原 fmap_resident 槽位改名——DLoad 主线模板，priority=1 host 模板）
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_VALUE),
                         ASCENDC_TPL_UINT_SEL(conv3DDWTemplateId, ASCENDC_TPL_UI_LIST, TPL_DLOAD),
                         ASCENDC_TPL_BOOL_SEL(isSplitKernelHW, 0), ASCENDC_TPL_BOOL_SEL(groupEnlarge, 0),
                         ASCENDC_TPL_UINT_SEL(winogradTilingFlag, ASCENDC_TPL_UI_LIST, TPL_WINOGRAD_DISABLE),
                         ASCENDC_TPL_BOOL_SEL(winogradResidentFlag, 0)));

#endif // CONV3D_BACKPROP_FILTER_V2_TILING_KEY_ARCH35_H
