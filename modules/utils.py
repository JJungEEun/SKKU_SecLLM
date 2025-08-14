import torch
import re
from tqdm import tqdm

def extract_cpp_code(text: str) -> str:
    """
    입력 문자열에서 C++/C 코드 블록만 추출하는 함수.
    - Markdown 코드 블록 형식(````cpp ... `````)을 찾아 반환
    """
    m = re.findall(r"```(?:cpp|c\+\+)\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[0].strip()
    m = re.findall(r"```c\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m[0].strip()
    return text.strip()

def apply_template(tokenizer, prompt_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )

@torch.inference_mode()
def generate_one(model, tokenizer, prompt_text: str, max_input_len=1536, max_new_tokens=512) -> str:
    """
    입력 받은 프롬프트에 대해 코드 생성하는 함수
    """
    templated = apply_template(tokenizer, prompt_text)

    inputs = tokenizer(
        templated,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_input_len,
    )

    for k in inputs:
        inputs[k] = inputs[k].to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,

        do_sample=False,
        num_beams=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,

        use_cache=True,
        return_dict_in_generate=True,
    )

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out.sequences[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)