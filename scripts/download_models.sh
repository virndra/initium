#!/usr/bin/env bash
# Download TinyStories-260K (llama2.c .bin) + tokenizer, and optional TinyLlama GGUF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"
cd "$MODELS"

echo "==> TinyStories-260K weights (karpathy tinyllamas)"
if [[ ! -f stories260K.bin ]]; then
  curl -L -o stories260K.bin \
    "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin"
else
  echo "    stories260K.bin already present"
fi

echo "==> TinyStories-260K tokenizer"
if [[ ! -f tok512.bin ]]; then
  curl -L -o tok512.bin \
    "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin"
else
  echo "    tok512.bin already present"
fi

# Also alias common name
if [[ ! -f tokenizer.bin ]]; then
  cp -f tok512.bin tokenizer.bin
fi

echo "==> Optional: TinyLlama-1.1B chat Q4_0 GGUF (commented — large download)"
# curl -L -o tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
#   "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

echo "Done. Models in $MODELS"
ls -lh "$MODELS"
