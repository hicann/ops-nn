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

import torch
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


@register("ascend_aclnn_ctc_loss")
class CtcLoss(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        def exec_cpu(log_probs_ori, targets_input, input_lengths, target_lengths, blank, reduction, zero_infinity,
                     is_after_910=True):
            log_probs = log_probs_ori
            targets = targets_input.astype(np.int64)
            if len(log_probs_ori.shape) == 2:
                T = log_probs_ori.shape[0]
                C = log_probs_ori.shape[1]
                log_probs = log_probs_ori.reshape([T, 1, C])

            def is_aicore_support(log_probs, targets, input_lengths, target_lengths, is_after_910=True):
                BLOCK = 8
                max_target_length = np.max(target_lengths)
                output_size = 2 * max_target_length + 1
                output_size_up = (output_size + BLOCK - 1) // BLOCK * BLOCK

                SYMBOL_SET_INDEX = 2
                BATCH_INDEX = 1
                TIME_INDEX = 0
                LABEL_INDEX = 2
                C_UB_NUM = 5
                MAX_LABEL_LEN = 1000
                ONE_FLOAT = 1.0
                C_LOOP_BLOCK = 64
                C_LOOP_SUM_BLOCK = 8
                FP32_BYTES = 4
                TARGETS_UB_NUM = 24
                max_label = (output_size_up - 1) // 2
                SMALL_UB_SIZE = 192 * 1024
                UB_SIZE = 256 * 1024
                RESERVED_UB_SIZE = 2 * 1024
                if (max_label > MAX_LABEL_LEN):
                    return False

                log_probs_shape = log_probs.shape
                symble_set = log_probs_shape[SYMBOL_SET_INDEX]
                batch_size = log_probs_shape[BATCH_INDEX]
                time_step = log_probs_shape[TIME_INDEX]
                targets_dsize = 4 if targets.dtype == np.int32 else 8
                input_lengths_dsize = 4 if input_lengths.dtype == np.int32 else 8
                all_data_size = int(symble_set * (
                            C_UB_NUM + ONE_FLOAT / C_LOOP_BLOCK + ONE_FLOAT / C_LOOP_SUM_BLOCK) * FP32_BYTES + time_step * input_lengths_dsize + max_label * (
                                                TARGETS_UB_NUM * FP32_BYTES + targets_dsize))
                ub_size = SMALL_UB_SIZE if is_after_910 else UB_SIZE
                availableUbSize = ub_size - RESERVED_UB_SIZE
                if (all_data_size >= availableUbSize):
                    return False

                if (batch_size * (targets_dsize + FP32_BYTES) >= availableUbSize):
                    return False

                if log_probs.dtype == np.float32:
                    return True
                else:
                    return False

            def count_trace(s_i, targets, max_target_length):
                output_size = 2 * max_target_length + 1
                s_inc = [0 for i in range(output_size)]
                e_inc = [0 for i in range(output_size)]

                one_step = 1
                two_step = 2
                repeats = 0
                idx_counter = 1
                s_inc[0] = one_step
                if (s_i >= 1):
                    for idx in range(1, s_i):
                        left = targets[idx - 1]
                        right = targets[idx]
                        if (left == right):
                            s_inc[idx_counter] = one_step
                            e_inc[idx_counter - 1] = one_step

                            s_inc[idx_counter + 1] = one_step
                            e_inc[idx_counter] = one_step

                            idx_counter = idx_counter + two_step
                            repeats = repeats + 1
                        else:
                            s_inc[idx_counter] = two_step
                            e_inc[idx_counter - 1] = two_step
                            idx_counter = idx_counter + 1
                e_inc[idx_counter - 1] = one_step
                return repeats, s_inc, e_inc

            def gen_log_alpha_golden(log_probs, targets, input_lengths, target_lengths, blank, is_after_910=True):
                ori_targets_dim = len(targets.shape)
                BLOCK = 8
                inf = 3.4e38
                neg_inf = -inf
                if is_after_910:
                    inf = float("inf")
                    neg_inf = float("-inf")
                T = log_probs.shape[0]
                N = log_probs.shape[1]
                C = log_probs.shape[2]
                C_BLOCK = (C + BLOCK - 1) // BLOCK * BLOCK
                targets_offset = [0]
                max_target_length = np.max(target_lengths)
                output_size = 2 * max_target_length + 1
                output_size_up = (output_size + BLOCK - 1) // BLOCK * BLOCK
                log_alpha = np.full([N * T * output_size_up], 0, dtype=log_probs.dtype)
                if len(targets.shape) == 1:
                    for i in range(N):
                        targets_offset.append(targets_offset[i] + target_lengths[i])
                else:
                    S = targets.shape[1]
                    for i in range(N):
                        targets_offset.append(targets_offset[i] + S)
                targets = targets.reshape(-1)
                targets_per_block = 8 if str(targets.dtype) == "int32" else 4
                targets_block = (max_target_length + targets_per_block - 1) // targets_per_block * targets_per_block
                targets_ub = np.full([targets_block], 0, dtype=targets.dtype)
                lamda1_ub = np.array([0 for i in range(output_size_up)], dtype=log_probs.dtype)
                lamda2_ub = np.array([0 for i in range(output_size_up)], dtype=log_probs.dtype)
                lamda3_ub = np.array([0 for i in range(output_size_up)], dtype=log_probs.dtype)

                for batch in range(N):
                    t_i = input_lengths[batch]
                    s_i = target_lengths[batch]
                    if t_i >= s_i:
                        output_dst = 0
                        output_src = output_size_up
                        log_alpha_ub = np.full([2 * output_size_up], neg_inf, dtype=log_probs.dtype)
                        for i in range(output_size_up):
                            log_alpha_ub[output_dst + i] = neg_inf
                        current_log_probs_ub = np.array([0 for i in range(output_size_up)], dtype=log_probs.dtype)
                        if ori_targets_dim == 2:
                            for tui in range(max_target_length):
                                targets_ub[tui] = targets[targets_offset[batch] + tui]
                        else:
                            for tui in range(s_i):
                                targets_ub[tui] = targets[targets_offset[batch] + tui]
                        repeats, s_inc, e_inc = count_trace(s_i, targets_ub, max_target_length)
                        log_alpha_ub[output_dst] = log_probs[0][batch][blank]
                        current_target = targets_ub[0]
                        if current_target >= 0:
                            log_alpha_ub[output_dst + 1] = log_probs[0][batch][current_target]
                        else:
                            log_alpha_ub[output_dst + 1] = log_probs[0][batch][blank]

                        if repeats < (t_i - s_i):
                            start = 0
                        else:
                            start = 1
                        end = 2

                        if t_i >= 1:
                            for t in range(1, t_i):
                                log_probs_ub = log_probs[t][batch]
                                log_probs_ub = np.pad(log_probs_ub, (0, C_BLOCK - log_probs_ub.shape[0]),
                                                      constant_values=0)
                                for li in range(output_size_up):
                                    log_alpha_ub[output_src + li] = neg_inf

                                remain = s_i + repeats - t_i + t
                                if remain >= 0:
                                    tmp = s_inc[remain]
                                    start = start + tmp
                                start_loop = start

                                if t <= (s_i + repeats):
                                    tmp = e_inc[t - 1]
                                    end = end + tmp

                                if start_loop == 0:
                                    a_tmp = log_alpha_ub[output_dst]
                                    b_tmp = log_probs_ub[blank]
                                    log_alpha_ub[output_src] = a_tmp + b_tmp
                                    start_loop = 1

                                for s in range(start_loop, end):
                                    if s % 2 == 0:
                                        current_target = blank
                                    else:
                                        current_target = targets_ub[s // 2]
                                    offset = s - start_loop

                                    if current_target >= 0:
                                        current_log_probs_ub[offset] = log_probs_ub[current_target]
                                    else:
                                        current_log_probs_ub[offset] = log_probs_ub[0]
                                    lamda1_ub[offset] = log_alpha_ub[output_dst + s]
                                    lamda2_ub[offset] = log_alpha_ub[output_dst + s - 1]

                                    if (s % 2 != 0) and (s != 1):
                                        next_target = targets_ub[s // 2 - 1]
                                        if current_target != next_target:
                                            lamda3_ub[offset] = log_alpha_ub[output_dst + s - 2]
                                        else:
                                            lamda3_ub[offset] = neg_inf
                                    else:
                                        lamda3_ub[offset] = neg_inf

                                lamax_ub = np.maximum(lamda1_ub, lamda2_ub)
                                lamax_ub = np.maximum(lamax_ub, lamda3_ub)

                                lamda1_ub = lamda1_ub - lamax_ub
                                lamda2_ub = lamda2_ub - lamax_ub
                                lamda3_ub = lamda3_ub - lamax_ub

                                lamda1_ub = np.exp(lamda1_ub)
                                lamda2_ub = np.exp(lamda2_ub)
                                lamda3_ub = np.exp(lamda3_ub)

                                exp_add_ub = lamda1_ub + lamda2_ub + lamda3_ub

                                ln_add_ub = np.log(exp_add_ub)

                                compute_tmp_ub = ln_add_ub + lamax_ub

                                compute_tmp_ub = compute_tmp_ub + current_log_probs_ub

                                for s in range(start_loop, end):
                                    log_alpha_ub[output_src + s] = compute_tmp_ub[s - start_loop]

                                for s in range(output_size):
                                    log_alpha[batch * T * output_size_up + (t - 1) * output_size + s] = log_alpha_ub[
                                        output_dst + s]

                                output_src = output_dst
                                output_dst = output_size_up - output_src

                        if ((batch * T * output_size_up + (t_i - 1) * output_size) >= 0):
                            for s in range(output_size):
                                log_alpha[batch * T * output_size_up + (t_i - 1) * output_size + s] = log_alpha_ub[
                                    output_dst + s]
                    else:
                        pass

                return log_alpha.reshape(N, T, output_size_up)

            def gen_log_alpha_golden_aicpu(log_probs, targets, input_lengths, target_lengths, blank, is_after_910=True):
                ori_targets_dim = len(targets.shape)
                BLOCK = 8
                inf = 3.4e38
                neg_inf = -inf
                if is_after_910:
                    inf = float("inf")
                    neg_inf = float("-inf")

                T = log_probs.shape[0]
                N = log_probs.shape[1]
                C = log_probs.shape[2]
                C_BLOCK = (C + BLOCK - 1) // BLOCK * BLOCK
                targets_offset = [0]
                max_target_length = np.max(target_lengths)
                output_size = 2 * max_target_length + 1
                output_size_up = (output_size + BLOCK - 1) // BLOCK * BLOCK
                log_alpha = np.full([N, T, output_size_up], 0, dtype=log_probs.dtype)
                log_alpha_offset0 = T * output_size
                log_alpha_offset1 = output_size
                if len(targets.shape) == 1:
                    for i in range(N):
                        targets_offset.append(targets_offset[i] + target_lengths[i])
                else:
                    S = targets.shape[1]
                    for i in range(N):
                        targets_offset.append(targets_offset[i] + S)
                targets = targets.reshape(-1)

                def get_target_prime(targets, offset, idx, blank):
                    interval = 2
                    target_offset = offset + (idx // interval)
                    if (idx % interval == 0):
                        return blank
                    elif (target_offset < 0):
                        return blank
                    else:
                        return targets[target_offset]

                for batch in range(N):
                    for i in range(output_size_up):
                        log_alpha[batch, 0, i] = neg_inf
                    input_length = input_lengths[batch]
                    target_length = target_lengths[batch]
                    offset = targets_offset[batch]
                    log_alpha[batch][0][0] = log_probs[0][batch][blank]
                    if target_length > 0:
                        current_target = get_target_prime(targets, offset, 1, blank)
                        if current_target < 0:
                            log_alpha[batch][0][1] = log_probs[0][batch][blank]
                        else:
                            log_alpha[batch][0][1] = log_probs[0][batch][current_target]
                    if target_length > max_target_length:
                        continue
                    for t in range(1, input_length):
                        for s in range(2 * target_length + 1):
                            current_target_prime = get_target_prime(targets, offset, s, blank)
                            la1 = log_alpha[batch][t - 1][s]
                            lamax = la1
                            if (s > 0):
                                la2 = log_alpha[batch][t - 1][s - 1]
                                lamax = np.maximum(la2, lamax)
                            else:
                                la2 = neg_inf
                            if ((s > 1) and (get_target_prime(targets, offset, s - 2, blank) != current_target_prime)):
                                la3 = log_alpha[batch][t - 1][s - 2]
                                lamax = np.maximum(la3, lamax)
                            else:
                                la3 = neg_inf
                            if (lamax == neg_inf):
                                lamax = 0
                            if current_target_prime < 0:
                                log_alpha[batch][t][s] = np.log(
                                    np.exp(la1 - lamax) + np.exp(la2 - lamax) + np.exp(la3 - lamax)) + lamax + \
                                                         log_probs[t][batch][blank]
                            else:
                                log_alpha[batch][t][s] = np.log(
                                    np.exp(la1 - lamax) + np.exp(la2 - lamax) + np.exp(la3 - lamax)) + lamax + \
                                                         log_probs[t][batch][current_target_prime]
                return log_alpha.reshape(N, T, output_size_up)

            if is_aicore_support(log_probs, targets, input_lengths, target_lengths, is_after_910):
                log_alpha = gen_log_alpha_golden(log_probs, targets, input_lengths, target_lengths, blank, is_after_910)
            else:
                log_alpha = gen_log_alpha_golden_aicpu(log_probs, targets, input_lengths, target_lengths, blank,
                                                       is_after_910)

            log_probs = torch.from_numpy(log_probs_ori)
            targets = torch.from_numpy(targets)
            input_lengths = torch.from_numpy(input_lengths)
            target_lengths = torch.from_numpy(target_lengths)
            ctc_loss = torch.nn.CTCLoss(blank, reduction, zero_infinity)
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            return [loss, torch.from_numpy(log_alpha)]

        logProbs = input_data.kwargs["logProbs"]
        targets = input_data.kwargs["targets"]
        inputLengths = input_data.kwargs["inputLengths"]
        targetlengths = input_data.kwargs["targetlengths"]
        blank = input_data.kwargs["blank"]
        zeroInfinity = input_data.kwargs["zeroInfinity"]
        output_cpu = exec_cpu(logProbs.numpy(), targets.numpy(), np.array(inputLengths), np.array(targetlengths),
                              blank, "none", zeroInfinity, is_after_910=True)
        return output_cpu[0], output_cpu[1]
