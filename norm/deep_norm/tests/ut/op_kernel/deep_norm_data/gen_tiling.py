import numpy as np
import sys

case0_params = [2, 256, 2, 64, 12288, 1, 1]

params_info = {
    "case0": case0_params,
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"

    base_params = np.array(params_list, dtype=np.uint32)

    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()