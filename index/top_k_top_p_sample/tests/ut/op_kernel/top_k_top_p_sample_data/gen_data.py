#!/usr/bin/env python3
# coding: utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
import os
import numpy as np
import tensorflow as tf

def gen_top_k_top_p_sample_data(batch, voc_size, logitsLeft, logitsRight, topKsleft, topksRight, topPsleft, topPsRight, dtype):
    logits = np.random.uniform(logitsLeft, logitsRight, [int(batch), int(voc_size)]).astype(dtype)
    topKs = np.random.uniform(topKsleft, topksRight, int(batch)).astype(np.int32)
    topPs = np.random.uniform(topPsleft, topPsRight, int(batch)).astype(dtype)

    logits.tofile("logits.bin")
    topKs.tofile("topKs.bin")
    topPs.tofile("topPs.bin")

DTYPE_MAP = {
    0: tf.bfloat16.as_numpy_dtype,#bfloat16
    1: np.float16 #float16
}

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    batch, voc_size, logitsLeft, logitsRight, topKsleft, topksRight, topPsleft, topPsRight, dataType = [int(p) for p in sys.argv[1:]]
    dtype = DTYPE_MAP.get(dataType)
    gen_top_k_top_p_sample_data(batch, voc_size, logitsLeft, logitsRight, topKsleft, topksRight, topPsleft, topPsRight, dtype)

