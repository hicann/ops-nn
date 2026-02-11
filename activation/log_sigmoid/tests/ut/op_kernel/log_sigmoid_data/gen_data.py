import sys
import numpy as np
import torch
from bfloat16 import bfloat16


def gen_golden_data(x_shape, x_dtype):
    x_shape = eval(x_shape)

    if x_dtype == "bfloat16":
        x = np.random.uniform(-1, 1, x_shape).astype(bfloat16)
    else:
        x = np.random.uniform(-1, 1, x_shape).astype(x_dtype)

    x.tofile("./x.bin")

    # Write your compute here
    if x_dtype == "bfloat16":
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        y = torch.nn.LogSigmoid()(x)
        y = y.numpy()
        y = y.astype(bfloat16)
    elif x_dtype == "float16":
        x = x.astype(np.float32)
        y = torch.nn.LogSigmoid()(torch.from_numpy(x)).numpy()
        y = y.astype(np.float16)
    else:
        y = torch.nn.LogSigmoid()(torch.from_numpy(x)).numpy()
    # End compute

    y.tofile("./golden_y.bin")


if __name__ == "__main__":
    gen_golden_data(*sys.argv[1:])