"""Export a trained CaboNet to ONNX for browser inference (onnxruntime-web).

CaboNet doesn't have a single forward() - it has context()/forward_global()/
forward_position() so training can call only the head(s) it needs. For
export, simplicity wins over efficiency: wrap it in a module that always
computes every head's output in one pass (the network is tiny, this costs
nothing) and returns them all, in a fixed order. The web app runs one
inference per decision point and picks out the slice it needs, masking
invalid actions client-side exactly like agent.py's NetPolicy does.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from cabo_rl.features import MAX_HAND, NUM_VALUES, feature_dim
from cabo_rl.net import CaboNet, GLOBAL_HEAD_SIZES, POSITION_HEADS

# Fixed output order - the web side must match this exactly.
OUTPUT_NAMES: list[str] = list(GLOBAL_HEAD_SIZES) + list(POSITION_HEADS)


class ExportWrapper(nn.Module):
    def __init__(self, net: CaboNet):
        super().__init__()
        self.net = net

    def forward(self, flat: torch.Tensor, own_values: torch.Tensor, opp_values: torch.Tensor):
        ctx = self.net.context(flat)
        outs = [self.net.forward_global(ctx, name) for name in GLOBAL_HEAD_SIZES]
        for name, side in POSITION_HEADS.items():
            block = own_values if side == "own" else opp_values
            outs.append(self.net.forward_position(ctx, name, block))
        return tuple(outs)


def export(checkpoint_path: str, out_path: str) -> None:
    net = CaboNet()
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    net.eval()
    wrapper = ExportWrapper(net)
    wrapper.eval()

    example_flat = torch.zeros(1, feature_dim())
    example_own = torch.zeros(1, MAX_HAND, NUM_VALUES)
    example_opp = torch.zeros(1, MAX_HAND, NUM_VALUES)

    torch.onnx.export(
        wrapper,
        (example_flat, example_own, example_opp),
        out_path,
        input_names=["flat", "own_values", "opp_values"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={name: {0: "batch"} for name in ["flat", "own_values", "opp_values", *OUTPUT_NAMES]},
        opset_version=17,
        # The model is ~180KB total - not worth splitting weights into a
        # separate .data file (the default). One self-contained .onnx file
        # is simpler to serve/cache correctly in the browser.
        external_data=False,
    )
    print(f"exported to {out_path}")
    print(f"output order: {OUTPUT_NAMES}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cabo_net_real.pt")
    parser.add_argument("--out", type=str, default="checkpoints/cabo_net.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.out)
