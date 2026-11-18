# Quantization Guide

Model quantization is a technique that reduces model size and computational overhead by lowering the numerical precision of weights and activations, thereby saving memory and improving inference speed.

`vLLM Ascend` supports multiple quantization methods. This guide provides instructions for using different quantization tools and running quantized models on vLLM Ascend.

> **Note**
>
> You can choose to convert the model yourself or use the quantized model we uploaded.
> See <https://www.modelscope.cn/models/vllm-ascend/Kimi-K2-Instruct-W8A8>.
> Before you quantize a model, ensure sufficient RAM is available.

## Quantization Tools

vLLM Ascend supports models quantized by two main tools: `ModelSlim` and `LLM-Compressor`.

### 1. ModelSlim (Recommended)

[ModelSlim](https://gitcode.com/Ascend/msmodelslim/blob/master/README.md) is an Ascend-friendly compression tool focused on acceleration, using compression techniques, and built for Ascend hardware. It includes a series of inference optimization technologies such as quantization and compression, aiming to accelerate large language dense models, MoE models, multimodal understanding models, multimodal generation models, etc.

#### Installation

To use ModelSlim for model quantization, install it from its [Git repository](https://gitcode.com/Ascend/msmodelslim):

```bash
# Install 26.0.0 version, this is currently the latest stable branch
git clone https://gitcode.com/Ascend/msmodelslim.git -b 26.0.0

cd msmodelslim

bash install.sh
```

#### Model Quantization

The following example shows how to generate W8A8 quantized weights for the [Qwen3-MoE model](https://gitcode.com/Ascend/msmodelslim/blob/master/example/Qwen3-MOE/README.md).

**Quantization Script:**

```bash
cd example/Qwen3-MOE

# Support multi-card quantization
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False

# Set model and save paths
export MODEL_PATH="/path/to/your/model"
export SAVE_PATH="/path/to/your/quantized_model"

# Run quantization script
python3 quant_qwen_moe_w8a8.py --model_path $MODEL_PATH \
--save_path $SAVE_PATH \
--anti_dataset ../common/qwen3-moe_anti_prompt_50.json \
--calib_dataset ../common/qwen3-moe_calib_prompt_50.json \
--trust_remote_code True
```

After quantization completes, the output directory will contain the quantized model files.

For more examples, refer to the [official examples](https://gitcode.com/Ascend/msmodelslim/tree/master/example).

### 2. LLM-Compressor

[LLM-Compressor](https://github.com/vllm-project/llm-compressor) is a unified compressed model library for faster vLLM inference.

#### Installation

```bash
pip install llmcompressor
```

#### Model Quantization

`LLM-Compressor` provides various quantization scheme examples.

##### Dense Quantization

An example to generate W8A8 dynamic quantized weights for dense model:

```bash
# Navigate to LLM-Compressor examples directory
cd examples/quantization/llm-compressor

# Run quantization script
python3 w8a8_int8_dynamic.py
```

##### MoE Quantization

An example to generate W8A8 dynamic quantized weights for MoE model:

```bash
# Navigate to LLM-Compressor examples directory
cd examples/quantization/llm-compressor

# Run quantization script
python3 w8a8_int8_dynamic_moe.py
```

For more content, refer to the [official examples](https://github.com/vllm-project/llm-compressor/tree/main/examples).

The quantization types currently supported by LLM-Compressor can be viewed in the `vllm_ascend/quantization/compressed_tensors_config.py` file.

### 3. Native FP8 checkpoints (no quantization step needed)

Many models are now published directly in FP8, for example [Qwen/Qwen3.8-27B-FP8](https://modelscope.cn/models/Qwen/Qwen3.8-27B-FP8) and [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash). Their `config.json` carries `"quant_method": "fp8"` together with a `weight_block_size`, meaning every quantized weight is stored as `float8_e4m3fn` plus one `float32` scale per weight block. vLLM Ascend detects and serves these checkpoints as published, so no offline re-quantization is required:

```bash
vllm serve /path/to/Qwen3.8-27B-FP8 --trust-remote-code
vllm serve /path/to/GLM-5.3-Flash --tensor-parallel-size 8 --trust-remote-code
```

How the weights are executed depends on the hardware:

- **950PR&950DT Products**: the block scales are re-grouped into MXFP8 at load time, so weights stay at one byte per element and the native FP8 matmul is used.
- **Other Ascend generations**: the block scales are resolved into the model dtype at load time and served by the BF16 matmul. Numerically equivalent to the checkpoint, but plan for roughly twice the weight memory.

Layers the checkpoint left unquantized, such as `visual.merger.*` on multimodal models, are listed in `ignored_layers` or `modules_to_not_convert` and are served unquantized.

Only the block-wise flavour is supported. A native FP8 checkpoint without `weight_block_size` (per-tensor or per-channel scales) has no Ascend execution path yet; re-quantize it with ModelSlim or LLM-Compressor instead.

## Running Quantized Models

Once you have a quantized model which is generated by **ModelSlim**, you can use vLLM Ascend for inference by specifying the `--quantization ascend` parameter to enable quantization features, while for models quantized by **LLM-Compressor**, it is not necessary to add this parameter.

### Offline Inference

```python
import torch

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
# Set sampling parameters
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(
    model="/path/to/your/quantized_model",
    max_model_len=4096,
    trust_remote_code=True,
    # Set appropriate TP and DP values
    tensor_parallel_size=2,
    data_parallel_size=1,
    # Set an unused port
    port=8000,
    # Set serving model name
    served_model_name="quantized_model",
    # Specify `quantization="ascend"` to enable quantization for models quantized by ModelSlim
    quantization="ascend",
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

### Online Inference

```bash
# Corresponding to offline inference
python -m vllm.entrypoints.api_server \
    --model /path/to/your/quantized_model \
    --max-model-len 4096 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --data-parallel-size 1 \
    --served-model-name quantized_model \
    --trust-remote-code 
```

### Experimental Q/K channel smoothing for FP8 SWA

The BF16-query, dynamic per-token FP8 KV path can load fixed post-RoPE
channel coefficients through `additional_config.c8_qk_smoothing_path`.
It computes `Q_attention = BF16(FP32(Q) * a)` and
`K_cache = BF16(FP32(K) * reciprocal(a))` before K quantization. V is
unchanged, and learned-key sink correction consumes the original Q.
Checkpoint weights do not need to be re-quantized.

Generate coefficients from independent calibration and holdout activations:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python tools/calibrate_c8_qk_smoothing.py \
    /path/to/captures /path/to/coefficients --device npu:0
```

Each capture is a CPU-tensor `.pt` file containing `layer_name`, `tp_rank`,
`tp_size`, `sample_id`, `split` (`calibration` or `validation`), `phase`
(`prefill` or `decode`), BF16 `query[Tq,Nq,D]`, `key[Tkv,Nkv,D]`,
`value[Tkv,Nkv,D]`, int64 `query_positions[Tq]` and `key_positions[Tkv]`,
`sliding_window`, optional `softmax_scale`, and optional `sink_key[Nkv,D]`.
Q/K must already have their model normalization and RoPE applied. Capture
every visible KV position; selected query rows may be a contiguous suffix.
Use different requests for calibration and validation, independent of the
accuracy evaluation set.

The tool selects candidates using calibration attention error, then uses
the holdout data only as a non-regression gate. It writes
`qk_smoothing.json` and `calibration_report.json`. Its explicit attention
reference approximates the native FIA effective BF16 scale convention;
native FIA and model accuracy still require separate validation.

Add the generated path to the existing per-token configuration:

```json
{
  "c8_k_quant_mode": "per_token",
  "c8_v_quant_mode": "per_token",
  "c8_qk_smoothing_path": "/path/to/coefficients/qk_smoothing.json"
}
```

This requires `--dtype bfloat16 --kv-cache-dtype mxfp8`. The JSON uses
`format_version: 1`, the configured `tensor_parallel_size`, and a `ranks`
mapping containing every TP rank. Each rank maps exact attention layer
prefixes to positive finite `[local KV heads, head size]` coefficients.
Unlisted layers retain the original execution path. Listed layers must use
sliding-window FIA with independent KV caches; shared-KV aliases are not
supported. Coefficients are loaded once before graph capture. Restart the
service and clear existing KV/prefix caches when coefficients change.

FP8 accuracy improvement is not guaranteed. Identity coefficients are kept
when candidates fail calibration or holdout checks. The Q/K transforms
also add runtime work; measure inference latency before enabling them.

### Experimental block/channel FP8 K for SWA

Set the token group size to enable dynamic K scales per group, KV head,
and channel on sliding-window layers:

```json
{
  "c8_k_quant_mode": "per_token",
  "c8_v_quant_mode": "per_token",
  "c8_swa_k_quant_group_size": 32
}
```

Use `--dtype bfloat16 --kv-cache-dtype mxfp8 --block-size 512`
with `--no-async-scheduling --disable-hybrid-kv-cache-manager`. The group
size can be 32, 64, or 128; its default is 0 (disabled). V is dynamically
quantized per token and KV head. Full-attention layers retain the existing
per-token FP8 FIA path. Checkpoint weights do not change.

K/V payloads are FP8 E4M3 with FP32 scales. FIA does not natively accept
the block-dependent K channel scales, so a Triton kernel gathers the
visible window and dequantizes it to BF16 before BF16 FIA computes
attention. The original Q supplies both attention and learned-key sink
correction. This is not native FIA FP8 dequantization or FP8 attention
arithmetic, and it adds runtime work and BF16 workspace.

Each physical page also reserves two groups of original BF16 K for
unfinished-group updates and speculative replacement. K is requantized
from those originals, avoiding accumulated error from rescaling old FP8
values. With one KV head and D=128, a 512-token page uses 157696 bytes
at group size 32, versus 135168 bytes in the per-token path. This overhead
is included in the KV memory budget; actual token capacity also depends
on hybrid page padding and temporary workspace. Outside-window pages are
still retained with the required hybrid-manager setting.

This path requires synchronous scheduling because FIA takes host sequence
lengths. It supports speculative lookahead shorter than the group size,
not arbitrary rewinds. Shared-KV aliases, PD transfer, PCP/DCP, and combining
this option with `c8_qk_smoothing_path` are unsupported. Restart the service
and clear existing KV/prefix caches when changing the cache format.

The quantization axes are an experimental accuracy candidate. Local
attention error, model accuracy, and latency must be measured separately;
improvement over per-token FP8 is not guaranteed.

## References

- [ModelSlim GitCode](https://gitcode.com/Ascend/msmodelslim)
- [LLM-Compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [vLLM Quantization Guide](https://docs.vllm.ai/en/latest/features/quantization/)
