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
 * \file sparse_tensor_dense_mat_mul_tiling_def.h
 * \brief TilingData def for op SparseTensorDenseMatMul
 */

#ifndef __SPARSE_TENSOR_DENSE_MAT_MUL_TILING_DEF_H__
#define __SPARSE_TENSOR_DENSE_MAT_MUL_TILING_DEF_H__

// 所有模板共用同一套TilingData
struct SparseTensorDenseMatMulTilingData {
    /* 
     *  阶段1. InitOutput或Workspace：把输出y所在的位置或是workspace的GM空间清零，此时elemNum指的是y里面的所有元素个数
     *  这一阶段要在SIMT之前、pipe->InitBuffer之前，也就是Init()内执行，且前后要处理好同步SyncAll，不考虑DOUBLE_BUFFER
     */
    int64_t initAndOutUsedCoreNum;              // Init和Output阶段用到的核心数
    int64_t initAndOutFormerCoreElemNum;        // 非尾核，1.Init阶段调用InitGlobalMemory的元素个数（即参数size）；2.Output阶段要处理的元素个数
    int64_t initAndOutTailCoreElemNum;          // 尾核，1.Init阶段调用InitGlobalMemory的元素个数；2.Output阶段要处理的元素个数

    /*
     *  阶段2，Compute：SIMT计算，此时elemNum指的是要计算乘法（总共nnz*p个）的元素对的个数
     */ 
    int64_t computeUsedCoreNum;                 // 计算阶段用到的核心数
    int64_t computePerCoreThreadNum;            // 计算阶段每核起的线程数（暂且写死等于LAUNCH_BOUND_THREAD_NUM）
    int64_t computeTotalElemNum;                // 要计算的所有元素的个数，等于nnz*p，其中nnz为稀疏矩阵x1的非0元素个数，p为矩阵x2（转置后）的列数，也即输出y矩阵的列数
    int64_t computeNnz;                         // nnz为输入x1_indices/x1_values的长度（非0元素个数）
    int64_t computeM;                           // 输入x1的shape为[m,n]或[n,m]
    int64_t computeN;                           // 输入x2的shape为[n,p]或[p,n]
    int64_t computeP;                           // 取决于x1和x2是否为adjoint

    /* 
     *  阶段3，Output（只有B16模板用到）：把在workspace的yWorkspace（此时为B32类型）转换为B16类型，再写回输出y的位置
     *  此时计算UB处理的元素个数，都必须以B32为基本大小单位
     *  拷入拷出使用同一个yQue_即TQueBind类型，开启DOUBLE_BUFFER，注意处理同步问题
     */
    int64_t outFormerCoreUbFactor;              // 非尾核，前面的一次循环要处理多少元素，即Ub一次能装下处理的元素个数，UbFactor都必须是对齐32字节后的元素个数!!
    int64_t outFormerCoreUbTailFactor;          // 非尾核，最后一次循环要处理多少元素（<=outFormerCoreUbFactor），UbFactor都必须是对齐32字节后的元素个数!!
    int64_t outFormerCoreUbLoopTimes;           // 非尾核，要多少次循环能把该核负责的元素全处理完（每次循环处理UbFactor个元素）
    int64_t outTailCoreUbFactor;                // 尾核，前面的一次循环处理多少元素，UbFactor都必须是对齐32字节后的元素个数!!
    int64_t outTailCoreUbTailFactor;            // 尾核，最后一次循环处理多少元素，UbFactor都必须是对齐32字节后的元素个数!!
    int64_t outTailCoreUbLoopTimes;             // 尾核，要多少次循环
};

#endif //__SPARSE_TENSOR_DENSE_MAT_MUL_TILING_DEF_H__