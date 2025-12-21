#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import os
import numpy as np
import stat
import torch

OPEN_FILE_MODES_640 = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
WRITE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

def write_file(shape, input):
    if "vocab_parallel_logits" in input:
        vocab_parallel_logits = np.random.randint(1, 255, shape).astype(np.float32)
        vocab_parallel_logits.tofile("./vocab_parallel_logits.bin")
    if "logits_max" in input:
        logits_max = np.random.random(shape).astype(np.float32)
        logits_max.tofile("./logits_max.bin")
    if "sum_exp_logits" in input:
        sum_exp_logits = np.random.random(shape).astype(np.float32)
        sum_exp_logits.tofile("./sum_exp_logits.bin")
    if "predicted_logits" in input:
        predicted_logits = np.random.random(shape).astype(np.float32)
        predicted_logits.tofile("./predicted_logits.bin")
    if "input" in input:
        input_np = np.random.random(shape).astype(np.float32)
        input_np.tofile("./input.bin")
    if "weight" in input:
        weight = np.random.random(shape).astype(np.float32)
        weight.tofile("./weight.bin")
    

def gen_tiling():
    formerbtCountLen = 22
    latterbtCountLen = 21
    formerbtCountTime = 0
    latterbtCountTime = 0
    formerCoreNum = 16
    formerCoreReservedbtNum = 22
    latterCoreReservedbtNum = 21
    singleCalculationQuantity = 0
    singleCalculationReservedQuantity = 1096
    elementsNumber = 16000
    vLen = 4096
    tiling = (np.array(i, dtype=np.uint32) for i in (formerbtCountLen, latterbtCountLen, formerbtCountTime, latterbtCountTime,
                                                    formerCoreNum, formerCoreReservedbtNum, latterCoreReservedbtNum, singleCalculationQuantity, 
                                                    singleCalculationReservedQuantity, elementsNumber, vLen
                                                    ))
    tiling_data = b''.join(x.tobytes() for x in tiling)

    with os.fdopen(os.open('./tiling.bin', WRITE_FILE_FLAGS, OPEN_FILE_MODES_640), 'wb') as f:
        f.write(tiling_data)

if __name__ == "__main__":
    vocab_parallel_logits_shape = [1024, 4096] # NHWC
    logits_max_shape = [1024]
    sum_exp_logits_shape = [1024]
    predicted_logits_shape = [1024]
    input_shape = [1024]
    weight_shape = [1024]
    write_file(vocab_parallel_logits_shape, "vocab_parallel_logits_shape")
    write_file(logits_max_shape, "logits_max_shape")
    write_file(sum_exp_logits_shape, "sum_exp_logits_shape")
    write_file(predicted_logits_shape, "predicted_logits_shape")
    write_file(input_shape, "input_shape")
    write_file(weight_shape, "weight_shape")
    gen_tiling()