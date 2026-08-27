# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import math

__golden__ = {"kernel": {"max_pool3d_grad": "max_pool3d_grad_golden"}}


class MaxPool3DGradGoldenGpuClient:
    def __init__(self, server_ip, server_port=8888):
        self.server_ip = server_ip
        self.server_port = server_port
        self._deps_loaded = False
        self._load_dependencies()

    def _load_dependencies(self):
        if not self._deps_loaded:
            global socket, pickle, struct, torch, np
            import socket
            import pickle
            import struct
            import torch
            import numpy as np

            self._deps_loaded = True

    def _recv_all(self, sock, n):
        data = b""
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def _send_msg(self, sock, msg):
        msg = pickle.dumps(msg)
        msg = struct.pack(">I", len(msg)) + msg
        sock.sendall(msg)

    def _recv_msg(self, sock):
        raw_msglen = self._recv_all(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack(">I", raw_msglen)[0]
        return pickle.loads(self._recv_all(sock, msglen))

    def compute_on_gpu(
        self,
        input_orig_x,
        input_orig_y,
        input_grads,
        attr_ksize,
        attr_strides,
        attr_padding,
        attr_pads,
        attr_dilation,
        attr_ceil_mode,
        attr_data_format,
    ):
        request = {
            "input_orig_x": input_orig_x,
            "input_orig_y": input_orig_y,
            "input_grads": input_grads,
            "attr_ksize": attr_ksize,
            "attr_strides": attr_strides,
            "attr_padding": attr_padding,
            "attr_pads": attr_pads,
            "attr_dilation": attr_dilation,
            "attr_ceil_mode": attr_ceil_mode,
            "attr_data_format": attr_data_format,
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(300)
                s.connect((self.server_ip, self.server_port))
                self._send_msg(s, request)
                result = self._recv_msg(s)
                return result
        except Exception as e:
            print(f"连接错误: {e}")


def handle_computation_torch(data):
    import torch
    import torch.nn.functional as F

    input_orig_x = data["input_orig_x"]
    input_grads = data["input_grads"]
    attr_ksize = data["attr_ksize"]
    attr_strides = data["attr_strides"]
    attr_data_format = data["attr_data_format"]

    x = torch.tensor(input_orig_x, requires_grad=True)
    if attr_data_format == "NDHWC":
        x_t = x.permute(0, 4, 1, 2, 3).float()
        kernel_size = (attr_ksize[1], attr_ksize[2], attr_ksize[3])
        strides = (attr_strides[1], attr_strides[2], attr_strides[3])
    else:
        x_t = x.float()
        kernel_size = (attr_ksize[2], attr_ksize[3], attr_ksize[4])
        strides = (attr_strides[2], attr_strides[3], attr_strides[4])

    x_t = x_t.detach().requires_grad_(True)
    out = F.max_pool3d(x_t, kernel_size=kernel_size, strides=strides)

    grads = torch.tensor(input_grads)
    if attr_data_format == "NDHWC":
        grads = grads.permute(0, 4, 1, 2, 3).float()
    else:
        grads = grads.float()

    out.backward(grads)
    dx = x_t.grad_output

    if attr_data_format == "NDHWC":
        dx = dx.permute(0, 2, 3, 4, 1)

    return dx.numpy()


def get_out_shape(
    in_width, pad_left, pad_right, kw, dilation, strides, ceil_mode=False
):
    out_width = 0
    if ceil_mode:
        out_width = (
            in_width + pad_left + pad_right - (dilation * (kw - 1) + 1) + strides - 1
        ) // strides + 1
        if (out_width - 1) * strides >= in_width + pad_left:
            out_width = out_width - 1
    else:
        out_width = (
            in_width + pad_left + pad_right - (dilation * (kw - 1) + 1)
        ) // strides + 1
    return out_width


def get_pad_for_same(out_width, strides, kw, dilation, in_width):
    pad_need = max((out_width - 1) * strides + ((kw - 1) * dilation + 1) - in_width, 0)
    pad_left = pad_need // 2
    pad_right = pad_need - pad_left
    flag = False
    if pad_need % 2 == 0:
        flag = True
    return flag, pad_left, pad_right


def normalize_padding(padding):
    if isinstance(padding, int):
        return (padding, padding, padding, padding, padding, padding)
    elif len(padding) == 1:
        return (padding[0], padding[0], padding[0], padding[0], padding[0], padding[0])
    elif len(padding) == 3:
        return (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2])
    elif len(padding) == 6:
        return padding
    else:
        raise ValueError(
            f"padding should be int, or tuple of 1, 3, or 6 ints, got {padding}"
        )


def max_pool_3d_forward(
    x, kernel_size, grad_out, stride=None, padding=0, dilation=1, ceil_mode=False
):
    if stride is None:
        stride = kernel_size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    padding = normalize_padding(padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    pD_front, pD_back, pH_front, pH_back, pW_front, pW_back = padding
    dD, dH, dW = dilation

    if x.ndim == 4:
        x = x[np.newaxis, :]

    N, C, D_in, H_in, W_in = x.shape

    total_pD = pD_front + pD_back
    total_pH = pH_front + pH_back
    total_pW = pW_front + pW_back

    padded_D = D_in + total_pD
    padded_H = H_in + total_pH
    padded_W = W_in + total_pW

    D_out = grad_out.shape[2]
    H_out = grad_out.shape[3]
    W_out = grad_out.shape[4]

    output = np.full((N, C, D_out, H_out, W_out), -np.inf, dtype=x.dtype)
    indices = np.zeros((N, C, D_out, H_out, W_out, 3), dtype=np.int32)
    indices[:] = -1

    for n in range(N):
        for c in range(C):
            for d_out in range(D_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        d_start = d_out * sD
                        h_start = h_out * sH
                        w_start = w_out * sW

                        max_val = -np.inf
                        max_d, max_h, max_w = -1, -1, -1

                        for kd in range(kD):
                            for kh in range(kH):
                                for kw in range(kW):
                                    d_pos = d_start + kd * dD
                                    h_pos = h_start + kh * dH
                                    w_pos = w_start + kw * dW

                                    if (
                                        d_pos < padded_D
                                        and h_pos < padded_H
                                        and w_pos < padded_W
                                    ):
                                        orig_d = d_pos - pD_front
                                        orig_h = h_pos - pH_front
                                        orig_w = w_pos - pW_front

                                        if (
                                            0 <= orig_d < D_in
                                            and 0 <= orig_h < H_in
                                            and 0 <= orig_w < W_in
                                        ):
                                            val = x[n, c, orig_d, orig_h, orig_w]
                                            if val > max_val or math.isnan(val):
                                                max_val = val
                                                max_d, max_h, max_w = (
                                                    orig_d,
                                                    orig_h,
                                                    orig_w,
                                                )

                        output[n, c, d_out, h_out, w_out] = max_val
                        indices[n, c, d_out, h_out, w_out] = [max_d, max_h, max_w]

    return output, indices


def max_pool_3d_backward(grad_output, indices, input_shape, padding):
    padding = normalize_padding(padding)

    if len(input_shape) == 4:
        input_shape = (1,) + input_shape

    N, C, D_in, H_in, W_in = input_shape
    grad_input = np.zeros(input_shape, dtype=grad_output.dtype)
    N_out, C_out, D_out, H_out, W_out = grad_output.shape

    for n in range(N_out):
        for c in range(C_out):
            for d_out in range(D_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        max_d, max_h, max_w = indices[n, c, d_out, h_out, w_out]
                        if max_d >= 0 and max_h >= 0 and max_w >= 0:
                            grad_input[n, c, max_d, max_h, max_w] += grad_output[
                                n, c, d_out, h_out, w_out
                            ]

    return grad_input


def max_pool_3d_grad(
    x, grad_output, kernel_size, strides, padding=0, dilation=[1, 1, 1], ceil_mode=False
):
    output, indices = max_pool_3d_forward(
        x, kernel_size, grad_output, strides, padding, dilation, ceil_mode
    )
    grad_input = max_pool_3d_backward(grad_output, indices, x.shape, padding)

    return grad_input


def max_pool3d_grad_GPU(x, grad, argmax, ksize, strides, padding, pads, data_format):
    input_orig_x = x
    input_orig_y = grad
    input_grads = argmax
    attr_ksize = ksize
    attr_strides = strides
    attr_padding = padding
    attr_pads = pads
    attr_dilation = [1, 1, 1, 1, 1]
    attr_ceil_mode = 0
    attr_data_format = data_format
    GPU_SERVER_IP = "127.0.0.1"
    GPU_SERVER_PORT = 8888
    client = MaxPool3DGradGoldenGpuClient(GPU_SERVER_IP, GPU_SERVER_PORT)
    t_dx = client.compute_on_gpu(
        input_orig_x=input_orig_x,
        input_orig_y=input_orig_y,
        input_grads=input_grads,
        attr_ksize=attr_ksize,
        attr_strides=attr_strides,
        attr_padding=attr_padding,
        attr_pads=attr_pads,
        attr_dilation=attr_dilation,
        attr_ceil_mode=attr_ceil_mode,
        attr_data_format=attr_data_format,
    )

    return t_dx


def max_pool3d_grad_golden(
    x, grad, argmax, ksize, strides, padding, pads, data_format, **kwargs
):
    inputx = x
    grad_out = argmax
    ksize = ksize
    strides = strides
    padding_mode = padding
    data_format = data_format
    ceil_mode = False
    pads = [0]
    dilation = [1]
    input_dtype = inputx.dtype

    if "float16" in str(input_dtype):
        inputx = inputx.astype(np.float32)
        grad_out = grad_out.astype(np.float32)

    if len(pads) == 1:
        pads = [pads[0], pads[0], pads[0], pads[0], pads[0], pads[0]]
    elif len(pads) == 3:
        pads = [pads[0], pads[0], pads[1], pads[1], pads[2], pads[2]]

    if len(strides) == 1:
        strides = [strides[0], strides[0], strides[0]]

    if len(ksize) == 1:
        ksize = [ksize[0], ksize[0], ksize[0]]

    if len(dilation) == 1:
        dilation = [dilation[0], dilation[0], dilation[0]]

    if data_format.lower() not in ["ncdhw", "ndhwc"]:
        raise Exception("MaxPool3DGrad only support NDHWC and NCDHW")

    if data_format == "NDHWC":
        inputx = np.transpose(inputx, (0, 4, 1, 2, 3))
        grad_out = np.transpose(grad_out, (0, 4, 1, 2, 3))
        if len(ksize) == 5:
            ksize = [ksize[1], ksize[2], ksize[3]]
        if len(strides) == 5:
            strides = [strides[1], strides[2], strides[3]]
        if len(dilation) == 5:
            dilation = [dilation[1], dilation[2], dilation[3]]
    else:
        if len(ksize) == 5:
            ksize = [ksize[2], ksize[3], ksize[4]]
        if len(strides) == 5:
            strides = [strides[2], strides[3], strides[4]]
        if len(dilation) == 5:
            dilation = [dilation[2], dilation[3], dilation[4]]

    out_d, out_h, out_w = 0, 0, 0
    d_flag = True
    h_flag = True
    w_flag = True
    if padding_mode == "CALCULATED":
        out_d = get_out_shape(
            inputx.shape[2],
            pads[0],
            pads[1],
            ksize[0],
            dilation[0],
            strides[0],
            ceil_mode,
        )
        out_h = get_out_shape(
            inputx.shape[3],
            pads[2],
            pads[3],
            ksize[1],
            dilation[1],
            strides[1],
            ceil_mode,
        )
        out_w = get_out_shape(
            inputx.shape[4],
            pads[4],
            pads[5],
            ksize[2],
            dilation[2],
            strides[2],
            ceil_mode,
        )
    elif padding_mode == "VALID":
        out_d = get_out_shape(
            inputx.shape[2], 0, 0, ksize[0], dilation[0], strides[0], False
        )
        out_h = get_out_shape(
            inputx.shape[3], 0, 0, ksize[1], dilation[1], strides[1], False
        )
        out_w = get_out_shape(
            inputx.shape[4], 0, 0, ksize[2], dilation[2], strides[2], False
        )
        pads[0], pads[2], pads[4] = 0, 0, 0
    else:
        out_d = (inputx.shape[2] + strides[0] - 1) // strides[0]
        out_h = (inputx.shape[3] + strides[1] - 1) // strides[1]
        out_w = (inputx.shape[4] + strides[2] - 1) // strides[2]
        d_flag, pads[0], pads[1] = get_pad_for_same(
            out_d, strides[0], ksize[0], dilation[0], inputx.shape[2]
        )
        h_flag, pads[2], pads[3] = get_pad_for_same(
            out_h, strides[1], ksize[1], dilation[1], inputx.shape[3]
        )
        w_flag, pads[4], pads[5] = get_pad_for_same(
            out_w, strides[2], ksize[2], dilation[2], inputx.shape[4]
        )

    in_d = (out_d - 1) * strides[0] + (dilation[0] * (ksize[0] - 1) + 1)
    in_h = (out_h - 1) * strides[1] + (dilation[1] * (ksize[1] - 1) + 1)
    in_w = (out_w - 1) * strides[2] + (dilation[2] * (ksize[2] - 1) + 1)
    pads[1] = max(in_d - inputx.shape[2] - pads[0], 0)
    pads[3] = max(in_h - inputx.shape[3] - pads[2], 0)
    pads[5] = max(in_w - inputx.shape[4] - pads[4], 0)

    grad_input = inputx
    if d_flag & h_flag & w_flag:
        grad_input = max_pool3d_grad_GPU(
            grad_input, grad, grad_out, ksize, strides, padding_mode, pads, data_format
        )
    else:
        grad_input = max_pool_3d_grad(inputx, grad_out, ksize, strides, pads)
        if data_format == "NDHWC":
            grad_input = np.transpose(grad_input, (0, 2, 3, 4, 1))
        grad_input = grad_input.astype(input_dtype, copy=False)

    return grad_input
