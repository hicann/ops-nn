# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import socket
import pickle
import struct
import numpy as np
import torch


def get_out_shape(in_width, pad_left, pad_right, kw, dilation, stride, ceil_mode=False):
    out_width = 0
    if ceil_mode:
        out_width = (
            in_width + pad_left + pad_right - (dilation * (kw - 1) + 1) + stride - 1
        ) // stride + 1
        if (out_width - 1) * stride >= in_width + pad_left:
            out_width = out_width - 1
    else:
        out_width = (
            in_width + pad_left + pad_right - (dilation * (kw - 1) + 1)
        ) // stride + 1
    return out_width


def get_pad_for_same(out_width, stride, kw, dilation, in_width):
    pad_need = max((out_width - 1) * stride + ((kw - 1) * dilation + 1) - in_width, 0)
    pad_left = pad_need // 2
    pad_right = pad_need - pad_left
    return pad_left, pad_right


def handle_computation(data):
    inputx = data["input_orig_x"]
    grad_out = data["input_grads"]
    ksize = data["attr_ksize"]
    strides = data["attr_strides"]
    padding_mode = data["attr_padding"]
    pads = data["attr_pads"]
    dilation = data["attr_dilation"]
    ceil_mode = data["attr_ceil_mode"]
    data_format = data["attr_data_format"]

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
        pads[0], _ = get_pad_for_same(
            out_d, strides[0], ksize[0], dilation[0], inputx.shape[2]
        )
        pads[2], _ = get_pad_for_same(
            out_h, strides[1], ksize[1], dilation[1], inputx.shape[3]
        )
        pads[4], _ = get_pad_for_same(
            out_w, strides[2], ksize[2], dilation[2], inputx.shape[4]
        )

    in_d = (out_d - 1) * strides[0] + (dilation[0] * (ksize[0] - 1) + 1)
    in_h = (out_h - 1) * strides[1] + (dilation[1] * (ksize[1] - 1) + 1)
    in_w = (out_w - 1) * strides[2] + (dilation[2] * (ksize[2] - 1) + 1)
    pads[1] = max(in_d - inputx.shape[2] - pads[0], 0)
    pads[3] = max(in_h - inputx.shape[3] - pads[2], 0)
    pads[5] = max(in_w - inputx.shape[4] - pads[4], 0)

    device = torch.device("cuda:0")
    x_tensor = torch.tensor(inputx, device=device, requires_grad=True)

    out = torch.nn.functional.max_pool3d(
        x_tensor,
        ksize,
        stride=strides,
        padding=[pads[0], pads[2], pads[4]],
        dilation=dilation,
        ceil_mode=False,
        return_indices=False,
    )
    g = torch.tensor(grad_out, device=device)
    out.backward(g)

    grad_input = x_tensor.grad.detach().cpu().numpy()

    if data_format == "NDHWC":
        grad_input = np.transpose(grad_input, (0, 2, 3, 4, 1))
    grad_input = grad_input.astype(input_dtype, copy=False)

    return grad_input


def recv_all(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def send_msg(sock, msg):
    msg = pickle.dumps(msg)
    msg = struct.pack(">I", len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock):
    raw_msglen = recv_all(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    return pickle.loads(recv_all(sock, msglen))


def start_server(host="0.0.0.0", port=10253):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        while True:
            conn, addr = s.accept()
            conn.settimeout(300)
            try:
                request = recv_msg(conn)
                if request is None:
                    continue
                result = handle_computation(request)
                send_msg(conn, result)
            except Exception as e:
                send_msg(conn, {"error": str(e)})


if __name__ == "__main__":
    start_server(port=10253)
