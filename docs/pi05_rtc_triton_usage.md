# PI0.5 RTC Triton Usage

## Scope

This document explains how to use the PI0.5 RTC Triton inference path in FluxVLA:

- training-time RTC configuration
- inference-time prefix RTC configuration
- which runner classes actually inject `prev_actions` / `prefix_len`
- how to run the local verification scripts

This is about the inference path implemented by `PI05FlowMatchingRTCInference`.
Triton is the backend; the RTC semantics come from the runner + model prefix lock.

## 1. What is actually wired

The PI0.5 RTC Triton model lives in:

```text
fluxvla/models/vlas/pi05_flowmatching_inference_rtc.py
```

Main entry point:

```python
PI05FlowMatchingRTCInference
```

What it does:

- builds a unified Triton + CUDA Graph inference pipeline
- accepts `prev_actions` and `prefix_len`
- supports only `rtc_config["method"] == "prefix"`
- locks the prefix region during denoising with `_prepare_prefill()` and `_apply_prefill_mask()`

## 2. Configuration split

There are two different config blocks:

### Training side

Use `rtc_training_config` under `model`:

```python
model = dict(
    type='PI05FlowMatching',
    rtc_training_config=dict(
        enabled=True,
        max_delay=7,
        distribution='exponential',
        temperature=1.0,
    ),
)
```

### Inference side

Use `inference_model` for the model class used at evaluation / deployment time, and `inference` for the runner:

```python
inference_model = model.copy()
```

For Triton RTC inference, make the inference model explicit:

```python
inference_model = model.copy()
inference_model.update(
    dict(
        type='PI05FlowMatchingRTCInference',
        num_views=2,
        triton_max_prompt_len=48,
        num_steps=10,
    ))
```

## 3. Inference runner requirement

The model alone is not enough.

For prefix RTC to work, the runner must inject:

- `prev_actions`
- `prefix_len`
- `rtc_config`

In the current codebase, that injection logic exists in:

```text
AlohaRTCInferenceRunner
```

The runner computes the remaining prefix from the previous chunk and current time offset:

```python
offset = (ctx.inference_start - prev.action_timestamp) / self.dt
remaining = resample_remaining(prev.raw_actions[0], offset)[None]
```

Then it passes `prev_actions` and `prefix_len` into `model.predict_action(...)`.

### Important caveat for UR3

The current `URInferenceRunner` path shown in this repo is plain inference and does **not** inject RTC prefix state in the same way.

So for UR3, you have two options:

- use a runner that already implements the RTC prefix injection path
- or extend `URInferenceRunner` with the same `prev_actions` / `prefix_len` logic

If the runner does not pass those fields, `PI05FlowMatchingRTCInference` will run as plain inference.

## 4. Recommended runtime config

### Aloha RTC example

```python
inference = dict(
    type='AlohaRTCInferenceRunner',
    async_execution=True,
    execute_horizon=10,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=5,
    ),
    seed=7,
)
```

### PI0.5 model example

```python
inference_model = dict(
    type='PI05FlowMatchingRTCInference',
    num_views=2,
    triton_max_prompt_len=48,
    num_steps=10,
    pretrained_name_or_path='./checkpoints/pi05_base/model.safetensors',
    rtc_training_config=dict(
        enabled=True,
        max_delay=7,
        distribution='exponential',
        temperature=1.0,
    ),
    ...
)
```

## 5. Key knobs

### `prefix_len`

- number of already-executed action steps to lock as prefix
- must satisfy `0 <= prefix_len <= action_chunk`
- too small: weak continuity
- too large: less freedom for the new chunk

### `async_execution`

- if `True`, execution and next inference overlap in time
- this is the main reason RTC prefix locking matters
- the runner uses time offset to recover the remaining prefix from the previous chunk

### `execute_horizon`

- how many steps are actually sent to the robot in async execution
- may be smaller than `action_chunk`
- used together with async replay / overlap scheduling

### `triton_max_prompt_len`

- maximum prompt length reserved by the Triton inference buffers
- must be at least as large as the prompt length you expect at runtime
- if your prompts are longer, increase this value before deployment

### `num_steps`

- number of denoising steps used by the Triton inference graph
- default is `10`
- must match the weight precomputation performed in `prepare_triton_inference()`

## 6. What happens at runtime

The runtime flow is:

```text
observe -> runner computes remaining prefix -> model locks prefix -> denoise -> output action chunk
```

More concretely:

1. The previous action chunk is stored in the runner context.
2. When the next inference starts, the runner computes the elapsed time.
3. The runner resamples the remaining part of the previous chunk.
4. `prefix_len` steps of that remaining trajectory are passed into the model.
5. `PI05FlowMatchingRTCInference` writes them into the prefill buffers.
6. Every denoising step re-applies the prefix mask.
7. The returned chunk keeps the prefix region fixed and only completes the suffix.

## 7. Local validation scripts

The repo provides two useful scripts:

### RTC correctness check

```bash
python scripts/verify_pi05_rtc_local.py \
    --config configs/pi05/pi05_paligemma_ur3_rtc_full_finetune.py \
    --device cuda:0 \
    --prefix-len 5
```

What it checks:

- builds the inference model
- runs plain prediction once
- runs RTC prefix prediction twice
- verifies the prefix region is preserved

### Triton vs non-Triton comparison

```bash
python scripts/compare_pi05_rtc_vs_triton.py \
    --config configs/pi05/pi05_paligemma_ur3_rtc_full_finetune.py \
    --device cuda:0 \
    --prefix-len 5
```

What it checks:

- compares a normal PI0.5 inference path against the Triton RTC path
- reports prefix / suffix difference statistics
- is the easiest way to confirm Triton inference matches the reference path

## 8. Minimal example for configs

A compact version that shows the important pieces only:

```python
model = dict(
    type='PI05FlowMatching',
    rtc_training_config=dict(
        enabled=True,
        max_delay=7,
        distribution='exponential',
        temperature=1.0,
    ),
    ...
)

inference_model = model.copy()
inference_model.update(dict(
    type='PI05FlowMatchingRTCInference',
    num_views=2,
    triton_max_prompt_len=48,
    num_steps=10,
))

inference = dict(
    type='AlohaRTCInferenceRunner',
    async_execution=True,
    execute_horizon=10,
    rtc_config=dict(
        enabled=True,
        method='prefix',
        prefix_len=5,
    ),
)
```

## 9. Common mistakes

- Using `rtc_training_config` and expecting inference-time prefix RTC automatically. It only affects training.
- Setting `rtc_config` on a runner that never injects `prev_actions` / `prefix_len`.
- Leaving `inference_model.type` as `PI05FlowMatchingInference` when you actually want the Triton RTC class.
- Choosing `prefix_len` larger than the action chunk length.
- Setting `triton_max_prompt_len` smaller than the real prompt length.

## 10. Memory note: A100 vs RTX 4090

Measured on LimVLA-3 with:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/compare_pi05_rtc_vs_triton.py \
    --config configs/pi05/pi05_paligemma_ur3_rtc_triton_compare.py \
    --device cuda:0 \
    --dtype bf16 \
    --prompt-len 32 \
    --prefix-len 5 \
    --state-dim 32
```

A100 80GB memory sampling result:

```text
baseline_mib = 15215
peak_mib     = 53646
delta_mib    = 38431
```

Interpretation:

- The test peaked at about `53.6GB` total used memory on GPU0.
- The incremental memory over the pre-existing baseline was about `38.4GB`.
- This is larger than the available memory of a 24GB RTX 4090, so OOM on 4090 is expected for this current implementation.

### Single Triton memory profile after CPU prepare optimization

Use the single-model profile script when you want deployment-like memory numbers without constructing the normal baseline model:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/profile_pi05_rtc_triton_memory.py \
    --config configs/pi05/pi05_paligemma_ur3_rtc_triton_compare.py \
    --device cuda:0 \
    --dtype bf16 \
    --prompt-len 32 \
    --prefix-len 5 \
    --state-dim 32
```

Measured on LimVLA-3 A100 after moving Triton weight preparation to CPU and replacing `ConditionGemmaInferenceModel.prepare_triton()` list+`torch.stack()` with preallocated per-layer copies:

```text
baseline_before_mib          = 15215
single_triton_peak_mib       = 22000
single_triton_delta_mib      = 6785
first_predict_ms             = 14669.0  # includes CPU prepare + H2D weight copy + CUDA Graph recording
replay_predict_ms            = 37.8
```

This is the more relevant number for 24GB cards than the compare script. The compare script intentionally builds both normal and Triton models and therefore has a much higher peak.

Why checkpoint size is misleading:

- A checkpoint around 1GB is just serialized parameter storage.
- Runtime memory includes the loaded PyTorch module weights, Triton-reformatted weights, temporary fp32 conversion tensors, transposed contiguous copies, `torch.stack()` copies, static inference buffers, attention/logits buffers, and CUDA Graph capture memory.
- During `prepare_triton()`, weights such as `ffn_down_w` can briefly exist in multiple forms at once: original module weight, fp32 temp, transposed contiguous bf16 tensor, list entries, and final stacked tensor.

Why A100 works but 4090 OOMs:

- A100 has 80GB memory, so a ~54GB peak fits.
- RTX 4090 has 24GB memory, so the current Triton preparation path can exceed memory before graph replay starts.

Main optimization direction:

- Replace list + `torch.stack()` weight preparation with preallocated destination tensors and per-layer `copy_()`.
- Avoid unnecessary `.float()` temporaries where bf16 is sufficient.
- Release or CPU-offload the original eager model weights after Triton weights are prepared if the eager path is not needed.
- Reduce static dimensions when possible, especially `triton_max_prompt_len`, `num_views`, `chunk_size`, and `num_steps`.

## 11. Summary

The shortest correct mental model is:

```text
training-time RTC:
  use rtc_training_config under model

inference-time RTC:
  use PI05FlowMatchingRTCInference + an RTC-capable runner

prefix RTC runtime:
  runner computes remaining actions from the previous chunk
  model locks the prefix region during denoising
```
