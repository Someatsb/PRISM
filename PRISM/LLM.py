import os
import re
import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from utils.utils import get_clustered_neighbors


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_llm(model_id=DEFAULT_MODEL_ID):
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        gpu_memory_utilization=0.6,
        max_model_len=8192,
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    return llm, tokenizer


def get_sampling_params(args):
    return SamplingParams(
        temperature=getattr(args, "temperature", 0),
        top_p=getattr(args, "top_p", 0.7),
        top_k=getattr(args, "top_k", 50),
        repetition_penalty=getattr(args, "repetition_penalty", 1.0),
        max_tokens=getattr(args, "max_tokens", 1024),
    )


def Generate_feature_by_LLM(node_indices, data, graph_nx, args, cur_round, llm=None, tokenizer=None):
    if llm is None or tokenizer is None:
        model_id = getattr(args, "llm_model_id", DEFAULT_MODEL_ID)
        llm, tokenizer = load_llm(model_id)

    sampling_params = get_sampling_params(args)

    prompts = []

    for idx in tqdm(node_indices, desc="Build LLM prompts"):
        node_id = idx.item() if isinstance(idx, torch.Tensor) else int(idx)

        node_prompts = generate_prompts_for_node(
            node_id=node_id,
            data=data,
            graph_nx=graph_nx,
            args=args,
            tokenizer=tokenizer,
        )

        prompts.extend(node_prompts)

    if len(prompts) == 0:
        return []

    prompt_texts = [item["prompt"] for item in prompts]
    outputs = llm.generate(prompt_texts, sampling_params)

    return parse_llm_outputs(
        prompts=prompts,
        outputs=outputs,
        label_names=data.label_names,
    )


def generate_prompts_for_node(node_id, data, graph_nx, args, tokenizer):
    cluster_data = get_clustered_neighbors(
        target_idx=node_id,
        x=data.x,
        graph_nx=graph_nx,
        args=args,
    )

    target_text = get_node_text(data, node_id)
    prompts = []

    for key, neighbor_ids in cluster_data.items():
        use_neighbor = int(key != -1)
        neighbor_text = ""

        if use_neighbor:
            neighbor_text = format_neighbor_texts(data, neighbor_ids)

        prompt = build_llama_prompt(
            target_text=target_text,
            neighbor_text=neighbor_text,
            category_names=data.label_names,
            tokenizer=tokenizer,
        )

        prompts.append({
            "node_id": node_id,
            "use_neighbor": use_neighbor,
            "neighbor_ids": neighbor_ids if use_neighbor else [],
            "prompt": prompt,
        })

    return prompts


def get_node_text(data, node_id):
    if hasattr(data, "raw_texts"):
        return data.raw_texts[node_id]

    if hasattr(data, "raw_text"):
        return data.raw_text[node_id]

    raise AttributeError("data must have raw_texts or raw_text")


def format_neighbor_texts(data, neighbor_ids):
    return "\n".join(
        f"Neighbor text {i + 1}: {get_node_text(data, nid)}"
        for i, nid in enumerate(neighbor_ids)
    )


def build_llama_prompt(target_text, neighbor_text, category_names, tokenizer):
    sorted_categories = sorted(list(category_names))
    label_text = ", ".join(sorted_categories)

    system_prompt = "You are an expert AI researcher classifying academic papers."

    user_prompt = f"""
Please read the following node abstract carefully.

[TARGET NODE TEXT]
{target_text}
""".strip()

    if neighbor_text:
        user_prompt += f"""

[NEIGHBOR NODE TEXT]
{neighbor_text}
""".rstrip()

    user_prompt += f"""

Based on the text above, classify the target node into the correct category from the list below:
{label_text}

[GUIDELINES]
1. START with 'Category: <Selected_Category>'.
2. Provide a single 'Confidence' score between 0.0 and 1.0.
3. Under 'Probability:', output a dictionary mapping category names to probabilities.
4. The probabilities must sum to 1.0.
5. Assign non-zero probabilities to at least 3 different categories.
6. Provide a brief explanation.

Output Format:
Category: <Selected_Category>
Confidence: <0.0 to 1.0>
Probability: {{'class1': prob1, 'class2': prob2, ...}}
Explanation: <Reasoning>
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_llm_outputs(prompts, outputs, label_names):
    parsed_outputs = []

    for prompt_info, output in zip(prompts, outputs):
        probability, confidence, explanation, category = parse_single_output(
            output=output,
            label_names=label_names,
        )

        parsed_outputs.append({
            "node_id": prompt_info["node_id"],
            "use_neighbor": prompt_info["use_neighbor"],
            "neighbor_ids": prompt_info.get("neighbor_ids", []),
            "category": category,
            "probability": probability,
            "confidence": confidence,
            "explanation": explanation,
        })

    return parsed_outputs


def parse_single_output(output, label_names):
    full_text = output.outputs[0].text

    category = parse_category(full_text, label_names)
    confidence = parse_confidence(full_text)
    probability = parse_probability(full_text, label_names)
    explanation = parse_explanation(full_text)

    if category == "Unknown" and sum(probability) > 0:
        category = label_names[int(np.argmax(probability))]

    return probability, confidence, explanation, category


def parse_category(text, label_names):
    matches = list(re.finditer(r"(?i)Category:\s*(.*)", text))

    if not matches:
        return "Unknown"

    category_text = matches[-1].group(1).split("\n")[0].strip()
    norm_category = normalize_label(category_text)

    label_map = {
        normalize_label(label): label
        for label in label_names
    }

    return label_map.get(norm_category, "Unknown")


def parse_confidence(text):
    match = re.search(r"(?i)Confidence:\s*([0-9.]+)", text)

    if not match:
        return 0.0

    try:
        confidence = float(match.group(1))
        return max(0.0, min(confidence, 1.0))
    except ValueError:
        return 0.0


def parse_probability(text, label_names):
    prob_section_match = re.search(
        r"(?i)Probability:\s*(.*?)(?=(?i)Explanation:|$)",
        text,
        re.DOTALL,
    )

    prob_text = prob_section_match.group(1) if prob_section_match else ""

    label_map = {
        normalize_label(label): label
        for label in label_names
    }

    items = re.findall(
        r"[\"']?([a-zA-Z0-9_.+\- /]+)[\"']?\s*:\s*([0-9]*\.?[0-9]+)",
        prob_text,
    )

    prob_dict = {}

    for key, value in items:
        norm_key = normalize_label(key)

        if norm_key not in label_map:
            continue

        try:
            prob_dict[label_map[norm_key]] = float(value)
        except ValueError:
            continue

    prob_array = np.array(
        [prob_dict.get(label, 0.0) for label in label_names],
        dtype=np.float64,
    )

    if prob_array.sum() > 0:
        prob_array = prob_array / prob_array.sum()

    return prob_array.tolist()


def parse_explanation(text):
    match = re.search(r"(?i)Explanation:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def normalize_label(label):
    return re.sub(r"[^a-z0-9]", "", str(label).lower())