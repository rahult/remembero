# benchmarks/runpod/Dockerfile — build from the repository root:
#   docker build --platform linux/amd64 -f benchmarks/runpod/Dockerfile -t <dockerhub-user>/rembero-reader-train:v1 .
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
ENV HF_HOME=/runpod-volume/hf TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN pip install --no-cache-dir "transformers>=5.0,<6" "trl>=0.24" "peft>=0.17" "datasets>=3.0" "accelerate>=1.0" \
    "liger-kernel>=0.8.2" sentencepiece protobuf runpod
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential && rm -rf /var/lib/apt/lists/* \
 && git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
 && cmake --build /opt/llama.cpp/build --target llama-quantize -j 8 \
 && pip install --no-cache-dir gguf
# requirements-convert_hf_to_gguf.txt is deliberately not installed: it pins transformers==4.57.6,
# which cannot read the tokenizer config transformers 5 writes (the bug READER-STRUCTURE.md hit on
# the Mac), and torch==2.11.0 from the CPU wheel index, which would replace this image's CUDA torch
# and leave the trainer on the CPU. Everything convert_hf_to_gguf.py imports — torch, numpy,
# transformers 5, sentencepiece, protobuf — is already here, and the script puts the clone's own
# gguf-py on sys.path ahead of the installed gguf.
WORKDIR /app
COPY benchmarks/train /app/benchmarks/train
COPY benchmarks/runpod/handler.py /app/benchmarks/runpod/handler.py
RUN touch /app/benchmarks/__init__.py /app/benchmarks/runpod/__init__.py
ENV PYTHONPATH=/app
CMD ["python", "-u", "/app/benchmarks/runpod/handler.py"]
