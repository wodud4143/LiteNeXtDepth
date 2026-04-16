"""
convert_to_trt.py

Convert LiteNeXtDepth checkpoint weights into a TensorRT FP16 engine.

Pipeline:
    PyTorch (.pth)  ->  ONNX  ->  Optimized ONNX  ->  TensorRT engine

Example:
    python convert_to_trt.py \
        --weights ./tmp/litenext_v1/models/weights_99 \
        --output  ./trt_engine \
        --model-name litenext_v1 \
        --height 192 --width 640
"""

from __future__ import annotations

import argparse
import os
import os.path as osp

import onnx
import onnxoptimizer
import tensorrt as trt
import torch
import torch.nn as nn
from onnx import numpy_helper

import networks


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
class FullDepthModel(nn.Module):
    """Encoder + decoder fused into a single module exposing one output tensor.

    Returning only the full-resolution disparity keeps the ONNX graph clean and
    avoids exporting auxiliary multi-scale outputs that are unused at inference.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        outputs = self.decoder(feats)
        return outputs[("disp", 0)]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def load_models(weights_dir: str, height: int, width: int):
    """Instantiate the network and load encoder/decoder checkpoints."""
    enc_path = osp.join(weights_dir, "encoder.pth")
    dec_path = osp.join(weights_dir, "depth.pth")

    enc_state = torch.load(enc_path, map_location=DEVICE)
    dec_state = torch.load(dec_path, map_location=DEVICE)

    encoder = networks.LiteNeXtDepth(
        drop_path_rate=0.2, height=height, width=width
    )
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0, 1, 2])

    # Filter out keys with mismatched shapes (e.g. saved height/width metadata)
    enc_ref = encoder.state_dict()
    dec_ref = decoder.state_dict()
    encoder.load_state_dict({
        k: v for k, v in enc_state.items()
        if k in enc_ref and enc_ref[k].shape == v.shape
    })
    decoder.load_state_dict({
        k: v for k, v in dec_state.items()
        if k in dec_ref and dec_ref[k].shape == v.shape
    })

    return encoder.to(DEVICE).eval(), decoder.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# ONNX export & optimization
# ---------------------------------------------------------------------------
def export_onnx(model: nn.Module, out_path: str, height: int, width: int) -> None:
    """Export the wrapped model to ONNX with a fixed input shape."""
    dummy = torch.randn(1, 3, height, width, device=DEVICE)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["disparity"],
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes=None,
    )


def optimize_onnx(src_path: str, dst_path: str) -> None:
    """Apply graph-level fusions and cast INT64 initializers to INT32.

    The INT64 -> INT32 cast is important for Jetson, where TensorRT prefers
    32-bit indices and INT64 weights can trigger fallback layers.
    """
    model = onnx.load(src_path)

    passes = [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_transpose",
        "fuse_consecutive_transposes",
        "fuse_bn_into_conv",
        "fuse_pad_into_conv",
        "fuse_add_bias_into_conv",
    ]
    model = onnxoptimizer.optimize(model, passes)

    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.INT64:
            arr32 = numpy_helper.to_array(init).astype("int32")
            init.CopyFrom(numpy_helper.from_array(arr32, init.name))

    onnx.save(model, dst_path)


# ---------------------------------------------------------------------------
# TensorRT engine build
# ---------------------------------------------------------------------------
def build_engine(onnx_path: str, workspace_gb: int = 1) -> bytes:
    """Build a serialized FP16 TensorRT engine from an optimized ONNX file."""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
    )
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build returned None")
    return bytes(serialized)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert LiteNeXtDepth weights to a TensorRT FP16 engine."
    )
    p.add_argument("--weights", required=True,
                   help="Folder containing encoder.pth and depth.pth")
    p.add_argument("--output", default="./trt_engine",
                   help="Output directory for ONNX and engine files")
    p.add_argument("--model-name", default="litenext",
                   help="Base filename for the generated artifacts")
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--workspace-gb", type=int, default=1,
                   help="TensorRT workspace memory pool size in GB")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    onnx_raw = osp.join(args.output, f"{args.model_name}.onnx")
    onnx_opt = osp.join(args.output, f"{args.model_name}_opt.onnx")
    engine   = osp.join(args.output, f"{args.model_name}_fp16.engine")

    print(f"[1/4] Loading weights from {args.weights}")
    encoder, decoder = load_models(args.weights, args.height, args.width)
    full_model = FullDepthModel(encoder, decoder).to(DEVICE).eval()

    print(f"[2/4] Exporting ONNX        -> {onnx_raw}")
    export_onnx(full_model, onnx_raw, args.height, args.width)

    print(f"[3/4] Optimizing ONNX graph -> {onnx_opt}")
    optimize_onnx(onnx_raw, onnx_opt)

    print(f"[4/4] Building TensorRT engine (FP16) -> {engine}")
    engine_bytes = build_engine(onnx_opt, workspace_gb=args.workspace_gb)
    with open(engine, "wb") as f:
        f.write(engine_bytes)

    print(f"\nDone. Engine saved to: {engine}")


if __name__ == "__main__":
    main()