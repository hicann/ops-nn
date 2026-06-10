# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import ast
import numpy
import sys
import torch


def verify_result(y_type, y_loss, mask_type, mask_loss):
    y_type = getattr(torch, y_type)
    y_loss = ast.literal_eval(y_loss)
    mask_type = getattr(torch, mask_type)
    mask_loss = ast.literal_eval(mask_loss)

    y = torch.from_numpy(numpy.fromfile('y.bin', numpy.uint8)).view(y_type)
    golden_y = torch.from_numpy(numpy.fromfile('golden.y.bin', numpy.uint8)).view(y_type)
    mask = torch.from_numpy(numpy.fromfile('mask.bin', numpy.uint8)).view(mask_type)
    golden_mask = torch.from_numpy(numpy.fromfile('golden.mask.bin', numpy.uint8)).view(mask_type)

    if y_loss == 0:
        y_pass = torch.equal(y, golden_y)
    else:
        y = y.float()
        golden_y = golden_y.float()
        abs_err = (y - golden_y).abs()
        rel_err = abs_err / golden_y.abs()
        elem_bad = ~((abs_err <= y_loss) | (rel_err <= y_loss))
        bad_ratio = elem_bad.sum().item()
        y_pass = bad_ratio <= y_loss
    if mask_loss == 0:
        mask_pass = torch.equal(mask, golden_mask)
    else:
        mask = mask.float()
        golden_mask = golden_mask.float()
        abs_err = (mask - golden_mask).abs()
        rel_err = abs_err / golden_mask.abs()
        elem_bad = ~((abs_err <= mask_loss) | (rel_err <= mask_loss))
        bad_ratio = elem_bad.sum().item()
        mask_pass = bad_ratio <= mask_loss

    return y_pass and mask_pass


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: python compare_data.py y_type y_loss mask_type mask_loss')
        exit(1)
    if not verify_result(*sys.argv[1:]):
        exit(1)
