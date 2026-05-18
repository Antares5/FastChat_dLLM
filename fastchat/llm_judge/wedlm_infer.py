# Copyright 2025 Tencent wechat. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
example.py - Simple Demo for WeDLMLLM Inference
"""

import argparse
import time
from transformers import AutoTokenizer
from wedlm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="WeDLMLLM Inference Demo")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--wedlm_entropy_threshold", type=float, default=0.6)
    parser.add_argument("--wedlm_pos_penalty_factor", type=float, default=0.02)
    parser.add_argument("--wedlm_window_size", type=int, default=16, help="WeDLM decoding window size")
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    # Default math prompt
    PROMPT = "A store sells apples for $2 each and oranges for $3 each. Tom bought 5 apples and 4 oranges. How much did he spend in total? Please solve this step by step."

    # Load tokenizer and prepare stop tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stop_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    for token in ["<|im_end|>", "<|endoftext|>"]:
        if token in tokenizer.get_vocab():
            tid = tokenizer.convert_tokens_to_ids(token)
            if tid not in stop_token_ids:
                stop_token_ids.append(tid)

    # Format prompt with chat template
    messages = [{"role": "user", "content": PROMPT}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Initialize LLM (wedlm_selection_mode removed, only entropy_parallel is supported)
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        wedlm_window_size=args.wedlm_window_size,
    )

    # Sampling parameters with WeDLM settings
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_token_ids,
        wedlm_entropy_threshold=args.wedlm_entropy_threshold,
        wedlm_pos_penalty_factor=args.wedlm_pos_penalty_factor,
    )

    # Warmup run
    sampling_params_warmup = SamplingParams(
        temperature=0.0,
        max_tokens=1000,
        stop_token_ids=stop_token_ids,
    )
    outputs = llm.generate(["generate random 1000 tokens"], sampling_params_warmup)

    # Record generation start time
    start_time = time.perf_counter()
    
    outputs = llm.generate([prompt_text], sampling_params)
    
    # Record generation end time
    end_time = time.perf_counter()
    
    # Calculate statistics
    elapsed_time = end_time - start_time
    generated_text = outputs[0]["text"]
    generated_tokens = tokenizer.encode(generated_text, add_special_tokens=False)
    num_tokens = len(generated_tokens)
    tokens_per_second = num_tokens / elapsed_time if elapsed_time > 0 else 0

    # Print result
    print("\n" + "=" * 50)
    print("Prompt:", PROMPT)
    print("=" * 50)
    print("Response:", generated_text)
    print("=" * 50)
    print(f"Generation Statistics:")
    print(f"  - Generated tokens: {num_tokens}")
    print(f"  - Time elapsed: {elapsed_time:.2f} seconds")
    print(f"  - Generation speed: {tokens_per_second:.2f} tokens/s")
    print("=" * 50)


if __name__ == "__main__":
    main()