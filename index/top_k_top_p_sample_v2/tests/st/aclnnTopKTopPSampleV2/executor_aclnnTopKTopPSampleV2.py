#!/usr/bin/env python3
# -- coding: utf-8 --
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""
top_k_top_p_sample_v2
"""
import torch

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import numpy as np
import torch
import torch.nn.functional as F
import random

@register("function_aclnn_top_k_top_p_sample_v2")
class FunctionTopKTopPSampleV2Api(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        logits = input_data.kwargs['logits']
        topK = input_data.kwargs['top_k']
        topP = input_data.kwargs['top_p']
        q = input_data.kwargs['q']
        min_ps = input_data.kwargs['min_ps']
        eps = input_data.kwargs['eps']
        is_need_logits = input_data.kwargs['is_need_logits']
        guess_k = input_data.kwargs['top_k_guess']
        k_max = input_data.kwargs['ks_max']
        input_is_logits = input_data.kwargs['input_is_logits']
        is_need_sample_result = input_data.kwargs['is_need_sample_result']

        TOP_P_MASKED_Logits = True   # gather origin input logits with masked sorted indices 
        USE_FAST_PROBS = True       # skip softmax before the post_sample stage to boost computation

        USE_PRINT = 0 # 打印开关

        ALL_P_MAX = 1.0
        FLT_NEG_INF = float('-inf')


        def mySoftmaxAndSort(x, dim=-1):
            # 处理负数维度
            if dim < 0:
                dim = x.dim() + dim
            
            # 减去最大值以提高数值稳定性
            max_vals = torch.max(x, dim=dim, keepdim=True)[0]
            shifted = x - max_vals
            
            # 计算指数
            exp_vals = torch.exp(shifted)
            
            # 计算Softmax
            softmax_output = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
            
            # 按概率从大到小排序
            sorted_probs, sorted_indices = torch.sort(softmax_output, dim=dim, descending=True)
            
            return sorted_probs, sorted_indices

        def onlySoftmax(x, dim=-1):
            # 处理负数维度
            if dim < 0:
                dim = x.dim() + dim
            # print(f"源logits = {x}")
            # 减去最大值以提高数值稳定性
            max_vals = torch.max(x, dim=dim, keepdim=True)[0]
            # print("max_vals = %f", max_vals)
            shifted = x - max_vals
            # print(f"shifted={shifted}")
            # 计算指数
            exp_vals = torch.exp(shifted)
            # print(f"exp_vals={exp_vals}")
            # 计算Softmax
            softmax_output = exp_vals / torch.sum(exp_vals, dim=dim, keepdim=True)
            # print("topKExpSum = %f", torch.sum(exp_vals, dim=dim, keepdim=True))
            return softmax_output

        batch_size, vocab_size = logits.shape
        if USE_PRINT:
            print(f"logits",logits)
            print(f"logits.shape",logits.shape)
        # 初始化结果张量
        rs_index = torch.zeros(batch_size, dtype=torch.long)
        logits_idx = torch.zeros((batch_size, vocab_size), dtype=torch.long)
        logits_sort_masked = torch.zeros((batch_size, vocab_size), dtype=torch.float32)


        # 根据是否需要 logits 初始化 rs_value
        if is_need_logits:
            if input_is_logits:    # pred_top_p_mode == "softmax":
                # 输入logits未归一化，先归一化再做采样，则无效概率初始化为全-INF，表示所有位置都未通过筛选
                rs_value = torch.ones((batch_size, vocab_size), dtype=torch.float32) * FLT_NEG_INF
            else:
                # 输入logits已经归一化，就不需要再进行归一化以免梯度消失，则无效概率初始化为全0，表示所有位置都未通过筛选
                rs_value = torch.zeros((batch_size, vocab_size), dtype=torch.float32)
        else:
            rs_value = torch.empty(0)

        # compute golden 
        for i in range(batch_size):
            # 保存原始logits，用于后续填充rs_value
            original_logits = logits[i].float()
    
            # TopKSample switch
            k_val = topK[i].item()  # 获取标量值
            top_ks_max = min(k_max, vocab_size)
            use_top_k = (k_val >=1 and k_val<=top_ks_max)

            # topPSample switch
            p = topP[i].item()
            use_top_p = p<ALL_P_MAX
            
            # sort input logits firstly （降序）
            topk_logits, topk_indices = torch.sort(original_logits, dim=-1, descending=True, stable=True)

            # topK
            if use_top_k:
                # 对 sorted_logits 取前 k 个
                k_val = min(k_val, vocab_size)  # 确保不超过词汇表大小
                topk_logits = topk_logits[:k_val]
                topk_indices = topk_indices[:k_val]
                
            if USE_PRINT:
                print(f'[use_top_k, use_top_p] = {use_top_k}, {use_top_p}')
                print(f'topk_logits = {topk_logits}')
                print(f'topk_indices = {topk_indices}')

            if input_is_logits:
                # 先归一化再做采样，计算排序后logits的softmax，仅用于topP
                topk_probs = onlySoftmax(topk_logits, dim=-1)
            else:
                # immediately cumsum on sorted logit probs
                topk_probs = topk_logits

            # 根据 p 和 ALL_P_MAX 的关系决定是否使用 Top-P
            if use_top_p:
                # 判断是否使用guessK逻辑：当没有明确topK但有topP时
                use_guess_k = not use_top_k
                # 对概率排序以便计算累积概率
                sorted_probs, sorted_probs_indices = torch.sort(topk_probs, dim=-1, descending=True, stable=True) #TODO: 为什么还需要进行排序
                if p>0:
                    # common top-p sample using cumsum prob filter
                    probs_sum = sorted_probs.cumsum(dim=-1)
                    top_p_mask = (probs_sum - sorted_probs) >= p
                    if USE_PRINT:
                        print(f'probs_sum={probs_sum}')
                else:
                    # reserve only one token with max prob if p<=0
                    top_p_mask = [True] * sorted_probs.numel()
                    top_p_mask[0] = False

                top_p_mask = torch.tensor(top_p_mask)
                top_p_sel = ~top_p_mask

                # 获取通过筛选的token在排序后概率中的索引
                selected_probs_indices = sorted_probs_indices[top_p_sel]    # local indices    
                if USE_PRINT:
                    print(f'\033[93m[use_top_k, use_top_p] = {use_top_k}, {use_top_p}\033[0m')
                    print(f'sorted_logits={sorted_probs}')
                    print(f'top_p_mask={top_p_mask}')
                    print(f'top_p_sel={top_p_sel}')
                    print(f'sorted_probs_indices={sorted_probs_indices}')
                    print(f'selected_probs_indices={selected_probs_indices}')

                # 映射回原始索引
                if USE_FAST_PROBS:
                    # 统一使用截断的top_p采样结果
                    selected_indices = topk_indices[selected_probs_indices]  
                    selected_logits = sorted_probs[top_p_sel]   # Logits
                else:
                    selected_indices = topk_indices[selected_probs_indices]
                    selected_logits = topk_logits[selected_probs_indices] # 待定

                # 获取非掩码部分的数量（即保留的token数量）
                false_count = (top_p_sel>0).sum().item()
            else:
                # 不使用 Top-P，继承top-P的输入
                selected_indices = topk_indices  # 所有通过TopK筛选的token
                selected_logits = topk_probs
                false_count = topk_probs.numel()
                top_p_sel = [True] * false_count
                top_p_sel = torch.tensor(top_p_sel)

            if USE_PRINT:
                print(f'\033[93m[use_top_k, use_top_p] = {use_top_k}, {use_top_p}\033[0m')
                print(f"Selected ele num = {false_count}")
                print(f"topPNum = {false_count}")
                print(f'selected_logits={selected_logits}')
                print(f'selected_indices={selected_indices}')
            
            if p <= 0 and input_is_logits: # kernel侧p小于零时取最大值，并且概率为1
                selected_logits[0] = 1

            # Saliency filtering using min_p，按核函数实现保存中间结果
            if min_ps != None:
                min_p = min_ps[i].item()
            else:
                min_p = -1 # 跳过minp采样

            if not use_top_k and not use_top_p and min_p < 1: # 没做过top_k、top_p时，kernel中未进行排序
                selected_indices = torch.arange(len(original_logits))
                if input_is_logits:
                    selected_logits = onlySoftmax(original_logits, dim = -1)
                else:
                    selected_logits = original_logits

            if min_p<=0:
                # keep all logits inherit from previous stage
                min_p_sel = [True] * false_count    # ones mask matching inherited input
            elif min_p<1:
                # ensure min_p_thd ALWAYS computed with the max logit of current batch
                min_p_thd = torch.max(selected_logits) * min_p
                sel_prob_mask = selected_logits >= min_p_thd
                min_p_sel = [a and b for a,b in zip(top_p_sel, sel_prob_mask)]
            else:
                # reserve only 1 max token for current batch
                min_p_sel = [False] * false_count
                min_p_sel[0] = True

            min_p_sel = torch.tensor(min_p_sel)
            # if not use_top_k and not use_top_p: # 当不使能topK、topP时，对logits归一化结果直接进行minP采样

            
            selected_logits = selected_logits[min_p_sel]
            selected_indices = selected_indices[min_p_sel]
            false_count = selected_logits.numel()
            if USE_PRINT:
                print(f'min_p={min_p}, min_p_sel={min_p_sel}')
                print(f'min_p_selected_logits={selected_logits}')
                print(f'min_p_selected_indices={selected_indices}')

            # Metric consistency
            if USE_FAST_PROBS:
                # 更快的计算，直接使用采样后的结果，仅确保它们都已归一化，但不确保当前batch的采样结果概率之和为1
                selected_probs = selected_logits
            else:
                if input_is_logits:
                    # 计算筛选后logits的softmax (0910 vllm golden)
                    selected_probs = onlySoftmax(selected_logits, dim=-1)
                else:
                    # 已经归一化，则使用直接值以免梯度消失 (sglang)
                    selected_probs = selected_logits

            if USE_PRINT:
                print(f'\033[93mUSE_FAST_PROBS={USE_FAST_PROBS}, input_is_logits={input_is_logits}\033[0m')
                print(f'selected_probs={selected_probs}')

            # Post sample 

            if q is not None: # qsample
                q_i = q[i, :false_count]
                q_sample = selected_probs / (q_i.abs() + eps)
                probs_index = q_sample.argmax(dim=0).view(-1)

                if USE_PRINT:
                    sorted_qsample, sorted_q_indices = torch.sort(q_sample, dim=-1, descending=True, stable=True)
                    print(f"q_sample={q_sample}")
                    print(f"Sorted Qsample logits of row{i} = {sorted_qsample}") 
                    print(f"Sorted Qsample indices of row{i} = {sorted_q_indices}") 
            elif not is_need_sample_result: # 直接在选择概率中取最大值
                probs_index = selected_probs.argmax(dim=0).view(-1)

            # 使用probs_index直接获取对应的原始索引
            if (is_need_sample_result):
                logits_sort_masked[i, :len(selected_logits)] = selected_probs
                logits_idx[i, :len(selected_indices)] = selected_indices
                # rs_index = None
            else:
                golden_index = selected_indices[probs_index].squeeze(0)
                if USE_PRINT:
                    print(f"golden_index[{i}] = {golden_index}, probs_index={probs_index}")
                rs_index[i] = golden_index

            if is_need_logits:
                # 设置所有通过筛选的token位置的值为原始logits值
                rs_value[i, selected_indices] = original_logits[selected_indices]
                #rs_value[rs_value==0] = FLT_NEG_INF

        if (is_need_sample_result):
            logits_sort_masked = logits_sort_masked
            logits_idx = logits_idx
            rs_index = None
        else:
            logits_sort_masked = None
            logits_idx = None
            rs_index = rs_index
        
        rs_value = rs_value

        if rs_index is None:
            rs_index = torch.tensor([], dtype=torch.int64)
        if rs_value is None:
            rs_value = torch.tensor([], dtype=torch.float32)
        if logits_idx is None:
            logits_idx = torch.tensor([], dtype=torch.int64)
        if logits_sort_masked is None:
            logits_sort_masked = torch.tensor([], dtype=torch.float32)   
        if USE_PRINT:
            print(f"[top_k_top_p_sample_golden] rs_index={rs_index}")
            print(f"[top_k_top_p_sample_golden] rs_value={rs_value}")
            print(f"[top_k_top_p_sample_golden] logits_idx={logits_idx}")   
            print(f"[top_k_top_p_sample_golden] logits_sort_masked={logits_sort_masked}") 

        return rs_index, rs_value, logits_idx, logits_sort_masked

@register("function_pyaclnn_top_k_top_p_sample_v2")
class PyAclnnTopKTopPSampleV2(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        import random
        import ctypes
        from atk.tasks.backends.lib_interface.acl_wrapper import nnopbase, AclDataType, AclFormat, AclTensor
        input_args, output_packages = super().init_by_input_data(input_data)

        AclTensorPtr = ctypes.POINTER(AclTensor)  # tensor指针类型
        null_void_ptr = ctypes.c_void_p(None)  # 声明一个空指针
        null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)  # 把这个空指针类型转换为tensor指针类型

        if type(input_args[3]).__name__ == "c_void_p":
            input_args[3] = null_tensor_ptr  # 赋值给q

        if str(input_args[10]) == "c_bool(True)":
            batch_size = input_data.kwargs['logits'].shape[0]
            vocab_size = input_data.kwargs['logits'].shape[1]
            logits_idx = torch.zeros((batch_size, vocab_size), dtype=torch.long)
            
            output_packages[2] = self.torch_tensor_to_acl(logits_idx, AclFormat.ACL_FORMAT_ND)
            print(output_packages[2])
            input_args[13] = output_packages[2]
        return input_args, output_packages

    def after_call(self, output_packages):
        rs_index, rs_value, logits_idx, logits_sort_masked = super().after_call(output_packages)
        print("output_packages[2] end", logits_idx)
        return rs_index, rs_value, logits_idx, logits_sort_masked
