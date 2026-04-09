#!/usr/bin/env python3
"""Test if iOS17 target changes ANE support for basic ops."""
import torch, numpy as np, coremltools as ct
import os, sys, shutil

sys.path.insert(0, "conversion")
from ane_ops import MODEL_DTYPE
from models.gemma4 import Gemma4Model
from models.gemma4_stateless_chunks import StatelessChunk1

OUT = "conversion/output/gemma4-ios17-test"
print("Loading HF model...")
hf_path = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/4742fe843cc01b9aed62122f6e0ddd13ea48b3d3")
model = Gemma4Model.from_pretrained(hf_path)

config = model.config
ctx = config.context_length
hs = config.hidden_size
pld = config.hidden_size_per_layer_input
nl = config.num_hidden_layers
max_hd = config.global_head_dim
td = nl * pld

c1 = StatelessChunk1(model)
c1.eval()

dids = torch.zeros(1, 1, dtype=torch.int32)
dpos = torch.zeros(1, dtype=torch.int32)
dmask = torch.zeros(1, 1, 1, ctx, dtype=MODEL_DTYPE)
dumask = torch.zeros(1, 1, ctx, 1, dtype=MODEL_DTYPE); dumask[0, 0, 0, 0] = 1.0
dplc = torch.zeros(1, 1, td, dtype=MODEL_DTYPE)
dimg = torch.zeros(1, 1, hs, dtype=MODEL_DTYPE)
dK1 = torch.zeros(8, 1, ctx, max_hd, dtype=MODEL_DTYPE)
dV1 = torch.zeros(8, 1, ctx, max_hd, dtype=MODEL_DTYPE)

print("Tracing...")
with torch.no_grad():
    traced = torch.jit.trace(c1, (dids, dpos, dmask, dumask, dplc, dimg, dK1, dV1))

for target_name, target in [("iOS17", ct.target.iOS17), ("iOS18", ct.target.iOS18)]:
    print(f"\n=== Converting with {target_name} ===")
    try:
        mlm = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
                ct.TensorType(name="position_ids", shape=(1,), dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=(1, 1, 1, ctx), dtype=np.float16),
                ct.TensorType(name="update_mask", shape=(1, 1, ctx, 1), dtype=np.float16),
                ct.TensorType(name="per_layer_combined", shape=(1, 1, td), dtype=np.float16),
                ct.TensorType(name="image_embedding", shape=(1, 1, hs), dtype=np.float16),
                ct.TensorType(name="K_in", shape=(8, 1, ctx, max_hd), dtype=np.float16),
                ct.TensorType(name="V_in", shape=(8, 1, ctx, max_hd), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="hidden_states_out", dtype=np.float16),
                ct.TensorType(name="K_out", dtype=np.float16),
                ct.TensorType(name="V_out", dtype=np.float16),
            ],
            minimum_deployment_target=target,
            compute_units=ct.ComputeUnit.ALL,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
        )
        os.makedirs(OUT, exist_ok=True)
        pkg = f"{OUT}/chunk1_{target_name}.mlpackage"
        if os.path.exists(pkg): shutil.rmtree(pkg)
        mlm.save(pkg)
        mlc = f"{OUT}/chunk1_{target_name}.mlmodelc"
        if os.path.exists(mlc): shutil.rmtree(mlc)
        compiled = mlm.get_compiled_model_path()
        shutil.copytree(compiled, mlc)
        print(f"Saved {mlc}")
    except Exception as e:
        print(f"FAILED: {e}")
