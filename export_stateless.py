#!/usr/bin/env python3
"""Export stateless chunks with pre-computed RoPE inputs (no gather/position_ids)."""
import torch, numpy as np, coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
import os, sys, shutil, json

sys.path.insert(0, "conversion")
from ane_ops import MODEL_DTYPE
from models.gemma4 import Gemma4Model
from models.gemma4_stateless_chunks import (
    StatelessChunk1, StatelessChunk2, StatelessChunk3, StatelessChunk4
)

OUT = "conversion/output/gemma4-stateless"
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

# Save RoPE tables as .npy for Swift side
os.makedirs(OUT, exist_ok=True)
np.save(f"{OUT}/cos_sliding.npy", model.cos_sliding.numpy().astype(np.float16))
np.save(f"{OUT}/sin_sliding.npy", model.sin_sliding.numpy().astype(np.float16))
np.save(f"{OUT}/cos_full.npy", model.cos_full.numpy().astype(np.float16))
np.save(f"{OUT}/sin_full.npy", model.sin_full.numpy().astype(np.float16))
print(f"Saved RoPE tables to {OUT}/")

def export_chunk(wrapper, name, inputs_spec, outputs_spec, trace_args):
    wrapper.eval()
    print(f"\n[{name}] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, trace_args)
    print(f"[{name}] Converting...")
    mlm = ct.convert(
        traced, inputs=inputs_spec, outputs=outputs_spec,
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"[{name}] Palettizing INT4...")
    mlm = palettize_weights(mlm, OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=4, mode="kmeans",
                                          granularity="per_grouped_channel", group_size=32)
    ))
    pkg = f"{OUT}/{name}.mlpackage"
    if os.path.exists(pkg): shutil.rmtree(pkg)
    mlm.save(pkg)
    mlmodelc = f"{OUT}/{name}.mlmodelc"
    if os.path.exists(mlmodelc): shutil.rmtree(mlmodelc)
    compiled = mlm.get_compiled_model_path()
    shutil.copytree(compiled, mlmodelc)
    return os.path.getsize(f"{pkg}/Data/com.apple.CoreML/weights/weight.bin")

# Common dummies for RoPE
dcs = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)  # sliding (256 dim)
dss = torch.zeros(1, 1, 1, 256, dtype=MODEL_DTYPE)
dcf = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)  # full (512 dim)
dsf = torch.zeros(1, 1, 1, 512, dtype=MODEL_DTYPE)
dmask = torch.zeros(1, 1, 1, ctx, dtype=MODEL_DTYPE)
dumask = torch.zeros(1, 1, ctx, 1, dtype=MODEL_DTYPE); dumask[0, 0, 0, 0] = 1.0
dplc = torch.zeros(1, 1, td, dtype=MODEL_DTYPE)

rope_inputs = [
    ct.TensorType(name="cos_s", shape=(1, 1, 1, 256), dtype=np.float16),
    ct.TensorType(name="sin_s", shape=(1, 1, 1, 256), dtype=np.float16),
    ct.TensorType(name="cos_f", shape=(1, 1, 1, 512), dtype=np.float16),
    ct.TensorType(name="sin_f", shape=(1, 1, 1, 512), dtype=np.float16),
]

# Chunk 1
c1 = StatelessChunk1(model)
dhs1 = torch.zeros(1, 1, hs, dtype=MODEL_DTYPE)
dK1 = torch.zeros(8, 1, ctx, max_hd, dtype=MODEL_DTYPE)
dV1 = torch.zeros(8, 1, ctx, max_hd, dtype=MODEL_DTYPE)
s1 = export_chunk(
    c1, "chunk1",
    inputs_spec=[
        ct.TensorType(name="hidden_states", shape=(1, 1, hs), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, ctx), dtype=np.float16),
        ct.TensorType(name="update_mask", shape=(1, 1, ctx, 1), dtype=np.float16),
        ct.TensorType(name="per_layer_combined", shape=(1, 1, td), dtype=np.float16),
    ] + rope_inputs + [
        ct.TensorType(name="K_in", shape=(8, 1, ctx, max_hd), dtype=np.float16),
        ct.TensorType(name="V_in", shape=(8, 1, ctx, max_hd), dtype=np.float16),
    ],
    outputs_spec=[
        ct.TensorType(name="hidden_states_out", dtype=np.float16),
        ct.TensorType(name="K_out", dtype=np.float16),
        ct.TensorType(name="V_out", dtype=np.float16),
    ],
    trace_args=(dhs1, dmask, dumask, dplc, dcs, dss, dcf, dsf, dK1, dV1),
)

# Chunk 2
c2 = StatelessChunk2(model)
dhs = torch.zeros(1, 1, hs, dtype=MODEL_DTYPE)
dK2 = torch.zeros(7, 1, ctx, max_hd, dtype=MODEL_DTYPE)
dV2 = torch.zeros(7, 1, ctx, max_hd, dtype=MODEL_DTYPE)
s2 = export_chunk(
    c2, "chunk2",
    inputs_spec=[
        ct.TensorType(name="hidden_states", shape=(1, 1, hs), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, ctx), dtype=np.float16),
        ct.TensorType(name="update_mask", shape=(1, 1, ctx, 1), dtype=np.float16),
        ct.TensorType(name="per_layer_combined", shape=(1, 1, td), dtype=np.float16),
    ] + rope_inputs + [
        ct.TensorType(name="K_in", shape=(7, 1, ctx, max_hd), dtype=np.float16),
        ct.TensorType(name="V_in", shape=(7, 1, ctx, max_hd), dtype=np.float16),
    ],
    outputs_spec=[
        ct.TensorType(name="hidden_states_out", dtype=np.float16),
        ct.TensorType(name="K_out", dtype=np.float16),
        ct.TensorType(name="V_out", dtype=np.float16),
        ct.TensorType(name="kv13_k", dtype=np.float16),
        ct.TensorType(name="kv13_v", dtype=np.float16),
        ct.TensorType(name="kv14_k", dtype=np.float16),
        ct.TensorType(name="kv14_v", dtype=np.float16),
    ],
    trace_args=(dhs, dmask, dumask, dplc, dcs, dss, dcf, dsf, dK2, dV2),
)

# Chunk 3
c3 = StatelessChunk3(model)
dkv13k = torch.zeros(1, 1, ctx, 256, dtype=MODEL_DTYPE)
dkv13v = torch.zeros(1, 1, ctx, 256, dtype=MODEL_DTYPE)
dkv14k = torch.zeros(1, 1, ctx, 512, dtype=MODEL_DTYPE)
dkv14v = torch.zeros(1, 1, ctx, 512, dtype=MODEL_DTYPE)
s3 = export_chunk(
    c3, "chunk3",
    inputs_spec=[
        ct.TensorType(name="hidden_states", shape=(1, 1, hs), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, ctx), dtype=np.float16),
        ct.TensorType(name="update_mask", shape=(1, 1, ctx, 1), dtype=np.float16),
        ct.TensorType(name="per_layer_combined", shape=(1, 1, td), dtype=np.float16),
    ] + rope_inputs + [
        ct.TensorType(name="kv13_k", shape=(1, 1, ctx, 256), dtype=np.float16),
        ct.TensorType(name="kv13_v", shape=(1, 1, ctx, 256), dtype=np.float16),
        ct.TensorType(name="kv14_k", shape=(1, 1, ctx, 512), dtype=np.float16),
        ct.TensorType(name="kv14_v", shape=(1, 1, ctx, 512), dtype=np.float16),
    ],
    outputs_spec=[ct.TensorType(name="hidden_states_out", dtype=np.float16)],
    trace_args=(dhs, dmask, dumask, dplc, dcs, dss, dcf, dsf, dkv13k, dkv13v, dkv14k, dkv14v),
)

# Chunk 4
c4 = StatelessChunk4(model)
s4 = export_chunk(
    c4, "chunk4",
    inputs_spec=[
        ct.TensorType(name="hidden_states", shape=(1, 1, hs), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, ctx), dtype=np.float16),
        ct.TensorType(name="update_mask", shape=(1, 1, ctx, 1), dtype=np.float16),
        ct.TensorType(name="per_layer_combined", shape=(1, 1, td), dtype=np.float16),
    ] + rope_inputs + [
        ct.TensorType(name="kv13_k", shape=(1, 1, ctx, 256), dtype=np.float16),
        ct.TensorType(name="kv13_v", shape=(1, 1, ctx, 256), dtype=np.float16),
        ct.TensorType(name="kv14_k", shape=(1, 1, ctx, 512), dtype=np.float16),
        ct.TensorType(name="kv14_v", shape=(1, 1, ctx, 512), dtype=np.float16),
    ],
    outputs_spec=[
        ct.TensorType(name="token_id", dtype=np.int32),
        ct.TensorType(name="token_logit", dtype=np.float16),
    ],
    trace_args=(dhs, dmask, dumask, dplc, dcs, dss, dcf, dsf, dkv13k, dkv13v, dkv14k, dkv14v),
)

json.dump({
    "model_name": "gemma4-e2b-stateless",
    "architecture": "gemma4",
    "hidden_size": hs, "num_hidden_layers": nl,
    "context_length": ctx, "vocab_size": config.vocab_size,
    "bos_token_id": 2, "eos_token_id": 1,
    "per_layer_dim": pld, "max_head_dim": max_hd,
    "embed_scale": float(hs ** 0.5),
    "per_layer_model_projection_scale": float(model.per_layer_model_projection_scale),
    "per_layer_input_scale": float(model.per_layer_input_scale),
    "per_layer_embed_scale": float(model.per_layer_embed_scale),
    "external_embeddings": True,
    "has_multimodal": True,
    "stateless": True,
    "num_chunks": 4,
    "precomputed_rope": True,
    "chunk_layer_ranges": [[0, 8], [8, 15], [15, 25], [25, 35]],
}, open(f"{OUT}/model_config.json", "w"), indent=2)

print(f"\nTotal: {(s1+s2+s3+s4)/1e9:.2f} GB")
