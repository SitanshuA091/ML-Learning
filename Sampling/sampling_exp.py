import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sampling import sample_token


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto"
).to(device)

model.eval()

messages = [
    {
        "role": "user",
        "content": "Explain why the sky appears blue in simple terms."
        # change content to a suitable prompt
    }
]

# Qwen is instruction tuned, so use its chat template.
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(device)

input_ids = inputs["input_ids"]


# autoregressive generation loop

max_new_tokens = 100

for _ in range(max_new_tokens):

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids
        )

    # Shape: [batch_size, sequence_length, vocabulary_size]
    logits = outputs.logits

    next_token_logits = logits[0, -1, :]

    # sampling implementation
    next_token_id = sample_token(
        next_token_logits,
        temperature=0.8,
        top_p=0.9
    )

    next_token = torch.tensor(
        [[next_token_id]],
        device=device
    )

    # Append sampled token to the sequence.
    input_ids = torch.cat(
        [input_ids, next_token],
        dim=-1
    )

    # Stop if the model generates EOS.
    if next_token_id == tokenizer.eos_token_id:
        break


prompt_length = inputs["input_ids"].shape[1]

generated_ids = input_ids[0, prompt_length:]

response = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)

print(response)