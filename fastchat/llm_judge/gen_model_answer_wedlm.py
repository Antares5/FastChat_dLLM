"""Generate answers with WeDLM inference engine.

Usage:
python3 gen_model_answer_wedlm.py --model-path /path/to/model --model-id my-model
"""
import argparse
import json
import os
import random
import time

import shortuuid
from tqdm import tqdm
from transformers import AutoTokenizer

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model.model_adapter import get_conversation_template
from wedlm import LLM, SamplingParams


def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    wedlm_entropy_threshold,
    wedlm_pos_penalty_factor,
    wedlm_window_size,
    max_model_len,
    gpu_memory_utilization,
    conv_template,
    revision,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers_wedlm
        ).remote
    else:
        get_answers_func = get_model_answers_wedlm

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path=model_path,
                model_id=model_id,
                questions=questions[i : i + chunk_size],
                answer_file=answer_file,
                max_new_token=max_new_token,
                num_choices=num_choices,
                num_gpus_per_model=num_gpus_per_model,
                max_gpu_memory=max_gpu_memory,
                wedlm_entropy_threshold=wedlm_entropy_threshold,
                wedlm_pos_penalty_factor=wedlm_pos_penalty_factor,
                wedlm_window_size=wedlm_window_size,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                conv_template=conv_template,
                revision=revision,
            )
        )

    if use_ray:
        ray.get(ans_handles)


def get_model_answers_wedlm(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    wedlm_entropy_threshold,
    wedlm_pos_penalty_factor,
    wedlm_window_size,
    max_model_len,
    gpu_memory_utilization,
    conv_template,
    revision,
):
    """Generate model answers using the WeDLM inference engine."""
    # Load tokenizer for stop tokens and fallback chat template
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=True,
    )

    # Build stop_token_ids from tokenizer
    stop_token_ids = []
    if tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    for token in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
        if token in tokenizer.get_vocab():
            tid = tokenizer.convert_tokens_to_ids(token)
            if tid not in stop_token_ids:
                stop_token_ids.append(tid)

    # Initialize WeDLM LLM engine
    llm_kwargs = dict(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        wedlm_window_size=wedlm_window_size,
    )
    # Multi-GPU: pass tensor_parallel_size if > 1
    if num_gpus_per_model > 1:
        llm_kwargs["tensor_parallel_size"] = num_gpus_per_model
    llm = LLM(**llm_kwargs)

    # Warmup run (recommended for WeDLM)
    warmup_params = SamplingParams(
        temperature=0.0,
        max_tokens=1000,
        stop_token_ids=stop_token_ids,
    )
    try:
        llm.generate(["WeDLM warmup: generate random tokens for engine initialization."], warmup_params)
    except Exception:
        pass  # warmup failure is non-fatal

    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            # Build a fresh conversation template for each choice.
            # Note: get_conversation_template is @cache'd, so we must .copy()
            # to avoid mutating the cached template across choices.
            if conv_template is not None:
                conv = conv_template.copy()
                use_chat_template = False
            else:
                try:
                    conv = get_conversation_template(model_id).copy()
                    use_chat_template = False
                except Exception:
                    # Last resort: use tokenizer.apply_chat_template
                    use_chat_template = True
                    conv = None  # will build prompt via tokenizer

            turns = []
            # For tokenizer-based chat template fallback, accumulate messages
            chat_messages = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]

                if use_chat_template:
                    # Build prompt via HuggingFace tokenizer chat template
                    chat_messages.append({"role": "user", "content": qs})
                    prompt = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()

                if temperature < 1e-4:
                    sampling_params = SamplingParams(
                        temperature=0.0,
                        max_tokens=max_new_token,
                        stop_token_ids=stop_token_ids,
                        wedlm_entropy_threshold=wedlm_entropy_threshold,
                        wedlm_pos_penalty_factor=wedlm_pos_penalty_factor,
                    )
                else:
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max_new_token,
                        stop_token_ids=stop_token_ids,
                        wedlm_entropy_threshold=wedlm_entropy_threshold,
                        wedlm_pos_penalty_factor=wedlm_pos_penalty_factor,
                    )

                # some models may error out when generating long outputs
                try:
                    outputs = llm.generate([prompt], sampling_params)
                    output = outputs[0]["text"]

                    # Apply stop_str truncation (same as original, only when conv is available)
                    if not use_chat_template:
                        if conv.stop_str and isinstance(conv.stop_str, list):
                            stop_str_indices = sorted(
                                [
                                    output.find(stop_str)
                                    for stop_str in conv.stop_str
                                    if output.find(stop_str) > 0
                                ]
                            )
                            if len(stop_str_indices) > 0:
                                output = output[: stop_str_indices[0]]
                        elif conv.stop_str and output.find(conv.stop_str) > 0:
                            output = output[: output.find(conv.stop_str)]

                    # Clean up special tokens
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if not use_chat_template and conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except Exception as e:
                    print("ERROR question ID: ", question["question_id"])
                    print("ERROR details: ", str(e))
                    output = "ERROR"

                if use_chat_template:
                    chat_messages.append({"role": "assistant", "content": output})
                else:
                    conv.update_last_message(output)
                turns.append(output)

            choices.append({"index": i, "turns": turns})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model answers using WeDLM inference engine."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    # WeDLM-specific arguments
    parser.add_argument(
        "--wedlm-entropy-threshold",
        type=float,
        default=0.6,
        help="WeDLM entropy threshold for early exiting (default: 0.6).",
    )
    parser.add_argument(
        "--wedlm-pos-penalty-factor",
        type=float,
        default=0.02,
        help="WeDLM positional penalty factor (default: 0.02).",
    )
    parser.add_argument(
        "--wedlm-window-size",
        type=int,
        default=16,
        help="WeDLM decoding window size (default: 16).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096).",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization ratio for WeDLM engine (default: 0.8).",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        default=None,
        help="Override the conversation template name (e.g., 'llama-3', 'chatml'). "
             "If not set, uses the default template for the model_id.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load for the tokenizer.",
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    # Resolve conversation template
    conv_template = None
    if args.conv_template:
        from fastchat.conversation import get_conv_template
        conv_template = get_conv_template(args.conv_template)
        print(f"Using conversation template: {args.conv_template}")
    else:
        try:
            conv_template = get_conversation_template(args.model_id)
            print(f"Using auto-detected conversation template for: {args.model_id}")
        except Exception:
            # Fallback to tokenizer's chat_template or a simple default
            from fastchat.conversation import get_conv_template
            try:
                conv_template = get_conv_template("one_shot")
                print("Warning: Could not auto-detect conversation template. "
                      "Falling back to 'one_shot' template. "
                      "Use --conv-template to specify one explicitly.")
            except Exception:
                print("Warning: No conversation template available. "
                      "Will try to use tokenizer.apply_chat_template as last resort.")
                conv_template = None

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        wedlm_entropy_threshold=args.wedlm_entropy_threshold,
        wedlm_pos_penalty_factor=args.wedlm_pos_penalty_factor,
        wedlm_window_size=args.wedlm_window_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        conv_template=conv_template,
        revision=args.revision,
    )

    reorg_answer_file(answer_file)
