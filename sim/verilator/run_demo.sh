#!/bin/bash
# =============================================================================
# run_demo.sh - End-to-end GPT-2 inference demo
# Usage: ./run_demo.sh [--prompt "Hello"] [--max-tokens 10] [--temperature 0.8] [--seed 42] [--kv-cache] [--skip-python]
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPU_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_DIR="$NPU_DIR/python"
BUILD_DIR="$SCRIPT_DIR/build"
DEMO_OUTDIR="${DEMO_OUTDIR:-$BUILD_DIR/demo_data}"

PROMPT="Hello"
MAX_TOKENS=10
SKIP_PYTHON=0
TEMPERATURE=0.0
SEED=42
KV_CACHE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompt) PROMPT="$2"; shift 2;;
        --max-tokens) MAX_TOKENS="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --seed) SEED="$2"; shift 2;;
        --kv-cache) KV_CACHE=1; shift;;
        --skip-python) SKIP_PYTHON=1; shift;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

echo "============================================"
echo "  GPT-2 NPU Inference Demo"
echo "  prompt: '$PROMPT'"
echo "  max_tokens: $MAX_TOKENS"
echo "  temperature: $TEMPERATURE  seed: $SEED"
echo "  kv_cache: $KV_CACHE"
echo "============================================"

# Step 1: Python - export weights, quantize, generate golden
if [ "$SKIP_PYTHON" -eq 0 ]; then
    echo ""
    echo "--- Step 1: Python weight export + quantize + golden ---"
    mkdir -p "$DEMO_OUTDIR"

    # Install requirements if needed
    if ! python3 -c "import transformers" 2>/dev/null; then
        echo "Installing Python requirements..."
        pip3 install -r "$PYTHON_DIR/requirements.txt"
    fi

    # Export FP32 weights from HuggingFace GPT-2
    # Always re-export: dimensions may have changed
    echo "Exporting GPT-2 weights..."
    DEMO_OUTDIR="$DEMO_OUTDIR" python3 "$PYTHON_DIR/tools/export_gpt2_weights.py"

    # Quantize to INT8 and pack weights.bin (always re-quantize from fp32)
    echo "Quantizing and packing weights..."
    DEMO_OUTDIR="$DEMO_OUTDIR" python3 "$PYTHON_DIR/tools/quantize_pack.py"

    # Run golden inference (always re-run for current prompt)
    echo "Running Python golden inference..."
    DEMO_OUTDIR="$DEMO_OUTDIR" python3 "$PYTHON_DIR/golden/gpt2_infer_golden.py" \
        --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" --seed "$SEED" --outdir "$DEMO_OUTDIR"
else
    echo "Skipping Python steps (--skip-python)"
fi

# Step 2: Build C++ demo
echo ""
echo "--- Step 2: Build demo_infer ---"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. 2>&1 | tail -3
cmake --build . --target demo_infer -j$(nproc) 2>&1 | tail -5
echo "Build complete."

# Step 3: Run NPU demo
echo ""
echo "--- Step 3: Run NPU inference ---"
KV_FLAG=""
if [ "$KV_CACHE" -eq 1 ]; then
    KV_FLAG="--kv-cache"
fi
./demo_infer --datadir "$DEMO_OUTDIR" --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" --seed "$SEED" $KV_FLAG

# Step 4: Decode NPU tokens back to text
echo ""
echo "--- Step 4: Decode NPU output ---"
python3 -c "
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Reproduce GPT-2 bytes_to_unicode
bs = (list(range(ord('!'), ord('~')+1))
      + list(range(0xa1, 0xac+1))
      + list(range(0xae, 0xff+1)))
cs = list(bs)
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
byte_encoder = dict(zip(bs, [chr(c) for c in cs]))
vocab = tokenizer.get_vocab()
tid_to_byte = {}
for bval, uchar in byte_encoder.items():
    tid = vocab.get(uchar)
    if tid is not None and tid < 256:
        tid_to_byte[tid] = bval

# Read prompt
with open('$DEMO_OUTDIR/prompt_tokens.txt') as f:
    prompt_toks = list(map(int, f.read().split()))

# Read NPU generated tokens
with open('$DEMO_OUTDIR/npu_tokens.txt') as f:
    npu_toks = [int(line.strip()) for line in f if line.strip()]

all_toks = prompt_toks + npu_toks
all_bytes = bytes([tid_to_byte.get(t, ord('?')) for t in all_toks])
text = all_bytes.decode('utf-8', errors='replace')
print(f'NPU generated text: \"{text}\"')
print(f'  prompt tokens:    {prompt_toks}')
print(f'  generated tokens: {npu_toks}')
"

echo ""
echo "Demo complete. Output in $DEMO_OUTDIR/"
