import warnings
import numpy as np
import heapq

warnings.filterwarnings("ignore")

import torch
import argparse
import json
import os
import time
import re
import sys

from tqdm import tqdm
from streaming_llm.utils import load, download_url, load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):

    if model.device.type == 'cpu':
        model = model.float()

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )

        now = len(generated_text) - 1
        if now > pos:
            print(" ".join(generated_text[pos:now]), end=" ", flush=True)
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    print(" ".join(generated_text[pos:]), flush=True)
    return past_key_values

# Reconcatenate the evicted data
def reintegrate_evicted_data(past_key_values, evicted_data, start_idx):
    reintegrated_kv = []
    #print("evicted_data: ", evicted_data)

    for layer_idx in range(len(past_key_values)):
        # Extracting the keys and values for both original and evicted data
        original_keys, original_values = past_key_values[layer_idx]
        evicted_keys, evicted_values = evicted_data[layer_idx]

        # Reintegrating the keys and values
        reintegrated_keys = torch.cat([original_keys[:, :, :start_idx], evicted_keys, original_keys[:, :, start_idx:]], dim=-2)
        reintegrated_values = torch.cat([original_values[:, :, :start_idx], evicted_values, original_values[:, :, start_idx:]], dim=-2)

        reintegrated_kv.append((reintegrated_keys, reintegrated_values))

    return reintegrated_kv

# Cosine Similarity
# def calculate_and_retrieve_top_slices(current_kv_sets, evicted_data_sets, top_k):
#     top_slices = []

#     # Iterate over each set in evicted_data_sets
#     for evicted_kv_pair in evicted_data_sets:
#         # Priority queue to store top similarities (min heap, hence using negative similarity)
#         top_similarities = []

#         # Compare every token in evicted_kv_pair with every token in each set in current_kv_sets
#         for token_pos in range(evicted_kv_pair[0].size(2)):
#             evicted_k = evicted_kv_pair[0][:, :, token_pos, :].unsqueeze(2)
#             evicted_v = evicted_kv_pair[1][:, :, token_pos, :].unsqueeze(2)

#             for current_kv_pair in current_kv_sets:
#                 for curr_pos in range(current_kv_pair[0].size(2)):
#                     curr_k = current_kv_pair[0][:, :, curr_pos, :].unsqueeze(2)
#                     curr_v = current_kv_pair[1][:, :, curr_pos, :].unsqueeze(2)

#                     k_similarity = torch.cosine_similarity(curr_k, evicted_k, dim=-1).mean()
#                     v_similarity = torch.cosine_similarity(curr_v, evicted_v, dim=-1).mean()
#                     avg_similarity = (k_similarity + v_similarity) / 2

#                     # Use negative similarity because heapq is a min heap
#                     if len(top_similarities) < top_k:
#                         heapq.heappush(top_similarities, (-avg_similarity, token_pos))
#                     else:
#                         heapq.heappushpop(top_similarities, (-avg_similarity, token_pos))

#         # Retrieve the top k slices based on the recorded token positions
#         top_positions = [pos for _, pos in sorted(top_similarities, reverse=True)]
#         top_kv_slices = [(evicted_kv_pair[0][:, :, top_positions, :], evicted_kv_pair[1][:, :, top_positions, :])]
#         top_slices.extend(top_kv_slices)

#     return top_slices

def calculate_and_retrieve_top_slices_cosine_similarity(current_kv_sets, evicted_data_sets, top_k):
    top_slices = []

    # Aggregate current_kv_sets
    aggregated_current_k = torch.mean(torch.cat([kv[0] for kv in current_kv_sets], dim=2), dim=2)
    aggregated_current_v = torch.mean(torch.cat([kv[1] for kv in current_kv_sets], dim=2), dim=2)

    for evicted_kv_pair in evicted_data_sets:
        top_similarities = []

        for token_pos in range(evicted_kv_pair[0].size(2)):
            evicted_k = evicted_kv_pair[0][:, :, token_pos, :].squeeze(2)
            evicted_v = evicted_kv_pair[1][:, :, token_pos, :].squeeze(2)

            # Calculate dot product for k and v, and take the average
            k_similarity = torch.cosine_similarity(aggregated_current_k, evicted_k, dim=-1)
            v_similarity = torch.cosine_similarity(aggregated_current_v, evicted_v, dim=-1)
            avg_similarity = (k_similarity + v_similarity) / 2

            # Convert avg_dot_product to a single scalar value
            avg_similarity_score = avg_similarity.mean().item()

            if len(top_similarities) < top_k:
                heapq.heappush(top_similarities, (-avg_similarity_score, token_pos))
            else:
                heapq.heappushpop(top_similarities, (-avg_similarity_score, token_pos))


        top_positions = [pos for _, pos in sorted(top_similarities, reverse=True)]
        top_kv_slices = [(evicted_kv_pair[0][:, :, top_positions, :], evicted_kv_pair[1][:, :, top_positions, :])]
        top_slices.extend(top_kv_slices)

    return top_slices

# def calculate_and_retrieve_top_slices_dot_product(current_kv_sets, evicted_data_sets, top_k):
#     top_slices = []

    # Aggregate current_kv_sets
    # aggregated_current_k = torch.mean(torch.cat([kv[0] for kv in current_kv_sets], dim=2), dim=2)
    # aggregated_current_v = torch.mean(torch.cat([kv[1] for kv in current_kv_sets], dim=2), dim=2)

    # for evicted_kv_pair in evicted_data_sets:
    #     top_similarities = []

    #     for token_pos in range(evicted_kv_pair[0].size(2)):
    #         evicted_k = evicted_kv_pair[0][:, :, token_pos, :].squeeze(2)
    #         evicted_v = evicted_kv_pair[1][:, :, token_pos, :].squeeze(2)

    #         # Calculate dot product for k and v, and take the average
    #         k_dot_product = torch.sum(aggregated_current_k * evicted_k, dim=-1)
    #         v_dot_product = torch.sum(aggregated_current_v * evicted_v, dim=-1)
    #         avg_dot_product = (k_dot_product + v_dot_product) / 2

    #         # Convert avg_dot_product to a single scalar value
    #         avg_similarity_score = avg_dot_product.mean().item()

    #         if len(top_similarities) < top_k:
    #             heapq.heappush(top_similarities, (-avg_similarity_score, token_pos))
    #         else:
    #             heapq.heappushpop(top_similarities, (-avg_similarity_score, token_pos))


    #     top_positions = [pos for _, pos in sorted(top_similarities, reverse=True)]
    #     top_kv_slices = [(evicted_kv_pair[0][:, :, top_positions, :], evicted_kv_pair[1][:, :, top_positions, :])]
    #     top_slices.extend(top_kv_slices)

    # return top_slices

def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    for idx, prompt in enumerate(prompts):
        prompt = "USER: " + prompt + "\n\nASSISTANT: "
        print("\n" + prompt, end="")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

        # read in evicted key_values from local file
        evicted_file_path = "data/evicted_data.pt"
        try:
            evicted_data = torch.load(evicted_file_path)
        except FileNotFoundError:
            evicted_data = []

        if past_key_values:
            if evicted_data != []:
                # Assuming you have past_key_values and evicted_data defined
                top_kv_sets = calculate_and_retrieve_top_slices_cosine_similarity(past_key_values, evicted_data, 200)

                # insert my evicted_data into correct part of the code
                past_key_values = reintegrate_evicted_data(past_key_values, top_kv_sets, 4)
                    
        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )


# split into many questions version
# @torch.no_grad()
# def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=100):
#     past_key_values = None
#     for idx, prompt in enumerate(prompts):
#         if prompt[0:15] == "I will tell you":
#             prompt = "USER: " + prompt 
#         # if prompt[0:21] == "Can you please repeat":
#         #     prompt = prompt + "\n\nASSISTANT: "
#         if prompt[0:4] == "Fill":
#             prompt = prompt + "\n\nASSISTANT: "

#         print("\n" + prompt, end="")
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#         # print(tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False))
#         input_ids = input_ids.to(model.device)
#         seq_len = input_ids.shape[1]
#         if kv_cache is not None:
#             space_needed = seq_len + max_gen_len
#             past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
#         if prompt[0:4]=="Fill":       
#             past_key_values = greedy_generate(
#                     model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
#                 )

# success. all in one question version
# @torch.no_grad()
# def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
#     past_key_values = None
#     for idx, prompt in enumerate(prompts):
#         prompt = "USER: " + prompt + "\n\nASSISTANT: "
#         print("\n" + prompt, end="")
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#         input_ids = input_ids.to(model.device)
#         seq_len = input_ids.shape[1]
#         if kv_cache is not None:
#             space_needed = seq_len + max_gen_len
#             past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

#         past_key_values = greedy_generate(
#             model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
#         )

# unique stories in the original format
# @torch.no_grad()
# def streaming_inference(model, tokenizer, story_sets, kv_cache=None, max_gen_len=500):
#     past_key_values = None

#     for idx, prompt in enumerate(story_sets):
#         # Check if the turn is a story or a question
#         if "repeat the story" in prompt:
#                 # This is the follow-up question
#             formatted_prompt = f"USER: {prompt}\n\nASSISTANT: "
#             generate_response = True
#         else:
#             # This is a story line
#             formatted_prompt = f"{prompt}"
#             generate_response = False

#         print("\n" + formatted_prompt, end="")
#         input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
#         input_ids = input_ids.to(model.device)
#         seq_len = input_ids.shape[1]

#         if kv_cache is not None:
#             space_needed = seq_len + max_gen_len
#             past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

#         # Generate response only for questions
#         if generate_response:
#             past_key_values = greedy_generate(
#                 model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
#             )

# unique stories 
# @torch.no_grad()
# def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
#     past_key_values = None
#     for idx, prompt in enumerate(prompts):
#         # Extract the stories and the follow-up question
#         if "Tell me more about" in prompt:
#             formatted_prompt = f"USER: {prompt}\n\nASSISTANT: "
#         else:
#             formatted_prompt = f"Stories: {prompt}"  # Non-question fact lines

#         print("\n" + formatted_prompt, end="")
#         input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
#         input_ids = input_ids.to(model.device)
#         seq_len = input_ids.shape[1]

#         if kv_cache is not None:
#             space_needed = seq_len + max_gen_len
#             past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

#         # Generate response only for questions
#         if "Tell me more about" in prompt:
#             past_key_values = greedy_generate(
#                 model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
#             )

# for asking facts 
# @torch.no_grad()
# def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
#     past_key_values = None
#     for idx, prompt in enumerate(prompts):
#         # Check if the prompt is a question
#         if "In which line can we learn about" in prompt:
#             formatted_prompt = f"USER: {prompt}\n\nASSISTANT: "
#         else:
#             formatted_prompt = f"FACT: {prompt}"  # Non-question fact lines

#         print("\n" + formatted_prompt, end="")
#         input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids
#         input_ids = input_ids.to(model.device)
#         seq_len = input_ids.shape[1]

#         if kv_cache is not None:
#             space_needed = seq_len + max_gen_len
#             past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)

#         # Generate response only for questions
#         if "In which line can we learn about" in prompt:
#             past_key_values = greedy_generate(
#                 model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
#             )

## for integer questions
#@torch.no_grad()
# def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
#     past_key_values = None
#     for idx, prompt in enumerate(prompts):
#         if prompt[0:6] == "Line 0":
#             prompt = "USER: " + prompt 
#         elif prompt[0] == "W":
#             prompt = prompt + "\n\nASSISTANT: "
#         # else just print prompt
            
#         print("\n" + prompt, end="")
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#         # print(tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False))
#         input_ids = input_ids.to(model.device)
#         seq_len = input_ids.shape[1]
#         if kv_cache is not None:
#             space_needed = seq_len + max_gen_len
#             past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
#         if prompt[0]=="W":       
#             past_key_values = greedy_generate(
#                     model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
#                 )



def main(args):
    model_name_or_path = args.model_name_or_path
    model, tokenizer = load(model_name_or_path)
    #test_filepath = os.path.join(args.data_root, "questions_joined.jsonl")
    test_filepath = os.path.join(args.data_root, "A_test_story.jsonl")
    print(f"Loading data from {test_filepath} ...")

    # if not os.path.exists(test_filepath):
    #     download_url(
    #         "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
    #         args.data_root,
    #     )
    #     os.rename(os.path.join(args.data_root, "question.jsonl"), test_filepath)

    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    if args.enable_streaming:
        kv_cache = enable_streaming_llm(
            model, start_size=args.start_size, recent_size=args.recent_size
        )
    else:
        kv_cache = None

    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="lmsys/vicuna-13b-v1.3"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--enable_streaming", action="store_true")
    parser.add_argument("--start_size", type=int, default=4)
    parser.add_argument("--recent_size", type=int, default=2000)
    args = parser.parse_args()

    main(args)
