#!/usr/bin/env python3
"""Test with CPU+NE (no GPU) to force ANE usage."""
import coremltools as ct, numpy as np, time, json
from transformers import AutoTokenizer

DIR = "conversion/output/gemma4-stateless"
EMB_DIR = "conversion/output/gemma4-mobile"

with open(f"{DIR}/model_config.json") as f:
    config = json.load(f)
hs, ctx = config["hidden_size"], config["context_length"]
nl, pld = config["num_hidden_layers"], config["per_layer_dim"]
max_hd = config["max_head_dim"]
vs = config["vocab_size"]
es, pps, pis, pes = config["embed_scale"], config["per_layer_model_projection_scale"], config["per_layer_input_scale"], config["per_layer_embed_scale"]
td = nl * pld

tokenizer = AutoTokenizer.from_pretrained(f"{EMB_DIR}/hf_model")
et_d = np.fromfile(f"{EMB_DIR}/embed_tokens_q8.bin", dtype=np.int8).reshape(vs, hs)
et_s = np.fromfile(f"{EMB_DIR}/embed_tokens_scales.bin", dtype=np.float16).reshape(vs)
epl_d = np.fromfile(f"{EMB_DIR}/embed_tokens_per_layer_q8.bin", dtype=np.int8).reshape(vs, td)
epl_s = np.fromfile(f"{EMB_DIR}/embed_tokens_per_layer_scales.bin", dtype=np.float16).reshape(vs)
proj_w = np.fromfile(f"{EMB_DIR}/per_layer_projection.bin", dtype=np.float16).reshape(td, hs)
norm_w = np.fromfile(f"{EMB_DIR}/per_layer_norm_weight.bin", dtype=np.float32)

# Load RoPE tables from the stateless export
cos_s_full = np.load(f"{DIR}/cos_sliding.npy").astype(np.float16)
sin_s_full = np.load(f"{DIR}/sin_sliding.npy").astype(np.float16)
cos_f_full = np.load(f"{DIR}/cos_full.npy").astype(np.float16)
sin_f_full = np.load(f"{DIR}/sin_full.npy").astype(np.float16)

def rms(x, w, eps=1e-6):
    return (x / np.sqrt(np.mean(x.astype(np.float64)**2) + eps) * w).astype(np.float32)
def plc(tid, emb):
    raw = epl_d[tid].astype(np.float32) * (float(epl_s[tid]) / 127.0 * pes)
    proj = (proj_w.astype(np.float32) @ emb.flatten().astype(np.float32)) * pps
    for li in range(nl):
        s = li*pld; proj[s:s+pld] = rms(proj[s:s+pld], norm_w)
    return ((proj + raw) * pis).astype(np.float16).reshape(1,1,td)
def emb(tid):
    return (et_d[tid].astype(np.float32) * (float(et_s[tid])/127.0 * es)).astype(np.float16).reshape(1,1,hs)

for cu_name, cu in [("ALL", ct.ComputeUnit.ALL), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
    print(f"\n=== {cu_name} ===")
    t0 = time.time()
    c1 = ct.models.MLModel(f"{DIR}/chunk1.mlpackage", compute_units=cu)
    c2 = ct.models.MLModel(f"{DIR}/chunk2.mlpackage", compute_units=cu)
    c3 = ct.models.MLModel(f"{DIR}/chunk3.mlpackage", compute_units=cu)
    c4 = ct.models.MLModel(f"{DIR}/chunk4.mlpackage", compute_units=cu)
    print(f"Loaded in {time.time()-t0:.1f}s")

    K1 = np.zeros((8, 1, ctx, max_hd), dtype=np.float16)
    V1 = np.zeros((8, 1, ctx, max_hd), dtype=np.float16)
    K2 = np.zeros((7, 1, ctx, max_hd), dtype=np.float16)
    V2 = np.zeros((7, 1, ctx, max_hd), dtype=np.float16)

    def step(tid, pos):
        global K1, V1, K2, V2
        mask = np.full((1,1,1,ctx), -65504.0, dtype=np.float16); mask[0,0,0,:pos+1] = 0.0
        um = np.zeros((1,1,ctx,1), dtype=np.float16); um[0,0,pos,0] = 1.0
        cs = cos_s_full[pos:pos+1].reshape(1,1,1,-1)
        ss = sin_s_full[pos:pos+1].reshape(1,1,1,-1)
        cf = cos_f_full[pos:pos+1].reshape(1,1,1,-1)
        sf = sin_f_full[pos:pos+1].reshape(1,1,1,-1)
        e = emb(tid)
        p = plc(tid, e)

        o1 = c1.predict({
            'hidden_states': e, 'causal_mask': mask, 'update_mask': um,
            'per_layer_combined': p,
            'cos_s': cs, 'sin_s': ss, 'cos_f': cf, 'sin_f': sf,
            'K_in': K1, 'V_in': V1,
        })
        K1 = o1['K_out']; V1 = o1['V_out']

        o2 = c2.predict({
            'hidden_states': o1['hidden_states_out'],
            'causal_mask': mask, 'update_mask': um,
            'per_layer_combined': p,
            'cos_s': cs, 'sin_s': ss, 'cos_f': cf, 'sin_f': sf,
            'K_in': K2, 'V_in': V2,
        })
        K2 = o2['K_out']; V2 = o2['V_out']

        o3 = c3.predict({
            'hidden_states': o2['hidden_states_out'],
            'causal_mask': mask, 'update_mask': um,
            'per_layer_combined': p,
            'cos_s': cs, 'sin_s': ss, 'cos_f': cf, 'sin_f': sf,
            'kv13_k': o2['kv13_k'], 'kv13_v': o2['kv13_v'],
            'kv14_k': o2['kv14_k'], 'kv14_v': o2['kv14_v'],
        })

        o4 = c4.predict({
            'hidden_states': o3['hidden_states_out'],
            'causal_mask': mask, 'update_mask': um,
            'per_layer_combined': p,
            'cos_s': cs, 'sin_s': ss, 'cos_f': cf, 'sin_f': sf,
            'kv13_k': o2['kv13_k'], 'kv13_v': o2['kv13_v'],
            'kv14_k': o2['kv14_k'], 'kv14_v': o2['kv14_v'],
        })
        return int(o4['token_id'][0])

    prompt = "<bos><|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n"
    tids = tokenizer.encode(prompt, add_special_tokens=False)

    nid = 0
    for i, tid in enumerate(tids):
        nid = step(tid, i)

    gen = []; pos = len(tids); t0 = time.time()
    for _ in range(16):
        if nid in {1, 106}: break
        gen.append(tokenizer.decode([nid]))
        nid = step(nid, pos); pos += 1
    elapsed = time.time() - t0
    print(f"Output: {''.join(gen)}")
    print(f"{len(gen)} tokens, {elapsed:.2f}s, {len(gen)/elapsed:.1f} tok/s")
