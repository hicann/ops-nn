import sys

import numpy as np
from bfloat16 import bfloat16

def compare_data(y_dtype):
    y = np.fromfile("./y.bin", y_dtype)
    golden_y = np.fromfile("./golden_y.bin", y_dtype)

    check_y = np.allclose(y, golden_y, 0.005, 0.005)
    if not check_y:
        raise RuntimeError("y compare failed")

if __name__ == "__main__":
    compare_data(*sys.argv[1:])