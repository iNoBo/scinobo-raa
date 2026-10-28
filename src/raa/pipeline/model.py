import os
import torch
import importlib
from functools import lru_cache
from peft import PeftModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

BASE_PATH = importlib.resources.files(__package__.split(".")[0])

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", cache_dir=os.path.join(BASE_PATH, "models_cache"), torch_dtype="auto", device_map="auto")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", cache_dir=os.path.join(BASE_PATH, "models_cache"))

# Lora model
if torch.cuda.is_available():
    model = PeftModel.from_pretrained(model, os.path.join(BASE_PATH, "model_checkpoints/raa_lora_flan_t5"), device_map={'': 0})
else:
    model = PeftModel.from_pretrained(model, os.path.join(BASE_PATH, "model_checkpoints/raa_lora_flan_t5"), device_map={'': 'cpu'})

print(f"Running merge_and_unload")
model = model.merge_and_unload()

print(model.hf_device_map)
def infer_text_web(sentence, max_len, **kwargs):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
    
    if max_len == 0:
        outputs = model.generate(input_ids=inputs, **kwargs)
    else:
        outputs = model.generate(input_ids=inputs, max_new_tokens=max_len, **kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def infer_text(sentence, **kwargs):
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids=inputs, **kwargs)
    if 'return_dict_in_generate' in kwargs:
        return outputs
    return [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]


def tokenize_text(sentence):
    return tokenizer.encode(sentence, add_special_tokens=False)


def compute_relative_probs(choices, predictions):
    abs_probs = {}
    for choice in choices:
        choice_score = 0.0
        tokenized_choice = [tok[1:] if tok.startswith('▁') else tok for tok in tokenizer._tokenize(choice)]
        for tok_i, tok in enumerate(tokenized_choice):
            if tok_i == len(predictions):
                break
            if tok in predictions[tok_i]:
                choice_score += predictions[tok_i][tok]
        abs_probs[choice] = choice_score
    
    Z = sum(abs_probs.values())
    if Z < 0.6:
        print(f"{1-Z} of unaccounted probability in classify")
        print(str(predictions))
        print(str(abs_probs))

    rel_probs = (
        {choice: prob / Z for (choice, prob) in abs_probs.items()}
        if Z != 0.0
        else abs_probs
    )
    return rel_probs


def classify_text(prompt, choices, top_k, **kwargs):
    if 'output_scores' not in kwargs:
        kwargs['output_scores'] = True
    if 'return_dict_in_generate' not in kwargs:
        kwargs['return_dict_in_generate'] = True
    
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(input_ids=inputs, **kwargs)

    return scores_to_relative_probs([step_scores[0] for step_scores in outputs.scores], choices, top_k)


@lru_cache(maxsize=None)
def decode_token(token_id):
    return tokenizer.decode(token_id)


def scores_to_relative_probs(step_scores, choices, top_k):
    # One topk per generation step, moved to Python in bulk, with cached token decoding: the per-token
    # decode/.item() calls were the main CPU cost of classify_text
    predictions = []
    for scores in step_scores:
        values, indices = scores.topk(top_k)
        predictions.append([[decode_token(i), p] for i, p in zip(indices.tolist(), torch.exp(values).tolist())])
    # Solve the bug where some indices have the same token value (e.g. Yes has 2 indices)
    predictions_dict = []
    for prediction in predictions:
        prediction_dict = dict()
        for token, prob in prediction:
            if token in prediction_dict:
                prediction_dict[token] += prob
            else:
                prediction_dict[token] = prob
        predictions_dict.append(prediction_dict)

    rel_probs = compute_relative_probs(choices, predictions_dict)

    return rel_probs


def length_sorted_batches(prompts, batch_size):
    # Indices of the prompts in batches of similar length, to keep the padding small
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def infer_text_batch(prompts, batch_size=16, **kwargs):
    # infer_text(prompt, **kwargs)[0] for every prompt, with batch_size prompts per generate call
    results = [None] * len(prompts)
    for idx in length_sorted_batches(prompts, batch_size):
        inputs = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(**inputs, **kwargs)
        for i, text in zip(idx, tokenizer.batch_decode(outputs, skip_special_tokens=True)):
            results[i] = text
    return results


def classify_text_batch(prompts, choices, top_k, batch_size=16, **kwargs):
    # classify_text(prompt, choices, top_k, **kwargs) for every prompt, with batch_size prompts per generate call
    if 'output_scores' not in kwargs:
        kwargs['output_scores'] = True
    if 'return_dict_in_generate' not in kwargs:
        kwargs['return_dict_in_generate'] = True

    results = [None] * len(prompts)
    for idx in length_sorted_batches(prompts, batch_size):
        inputs = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(**inputs, **kwargs)
        generated = outputs.sequences[:, 1:]  # without the decoder start token
        for row, i in enumerate(idx):
            # Keep the steps this prompt generates on its own: up to and including its first EOS
            eos = (generated[row] == tokenizer.eos_token_id).nonzero()
            n_steps = eos[0].item() + 1 if len(eos) else len(outputs.scores)
            results[i] = scores_to_relative_probs([outputs.scores[s][row] for s in range(n_steps)], choices, top_k)
    return results


def multilabel_classify_text(prompt, choices, top_k, **kwargs):
    # Get the probabilities for each choice using a Yes/No classifier from "classify_text" function
    choice_probs = dict()
    for choice in choices:
        modded_prompt = f"{prompt}\nAnswer: {choice}\nQuestion: Is the answer given above correct, Yes or No?"
        classify_res = classify_text(modded_prompt, ['Yes', 'No'], top_k, **kwargs)
        choice_probs[choice] = classify_res['Yes']
    return choice_probs


if __name__ == '__main__':
    import timeit

    def time_infer_text():
        infer_text('"### Snippet:\nThe training of our deep learning model was performed using the PyTorch framework (version 1.9.0), with the CIFAR-10 dataset. The framework can be downloaded from https://pytorch.org.\n\n### Question:\nList all the artifacts in the above snippet.\n\n### Answer:\n"', max_new_tokens=5)

    k = 20
    print(f"Average time taken for {k} runs of infer_text():", timeit.timeit(time_infer_text, number=k) / k)
