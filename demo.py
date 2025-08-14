import os
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from modules.utils import extract_cpp_code, apply_template, generate_one


# Basic environment settings
REPO_ID   = "ChaeSJ/llama-3.1-8b-finetuned"

## Local input/output path settings
SUBFOLDER = "llama_3_1_8b_finetuned"    
CACHE_DIR = "cache"   
TMP_DIR   = "tmp" 

input_path       = "dataset/selected_prompts_cpp.json"
json_output_path = "output/llama_3_1_8b_demo_generated.json"
cpp_output_dir   = "output/extracted_cpp"


## Set environment variables for Transformers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)
os.environ["TMPDIR"] = TMP_DIR

# Load tokenizer & model
print(">> Loading tokenizer from Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(
    REPO_ID,
    subfolder=SUBFOLDER,
    cache_dir=CACHE_DIR,
    use_fast=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    print(">> Added pad_token as eos_token")

print(">> Loading model from Hugging Face (full finetuned weights)...")
model = AutoModelForCausalLM.from_pretrained(
    REPO_ID,
    subfolder=SUBFOLDER,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16,  
    device_map="auto",           
    low_cpu_mem_usage=True,
)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

model.eval()


# Input prompts
print(">> Loading prompts...")
with open(input_path, "r", encoding="utf-8") as f:
    prompts_data = json.load(f)


# Prepare to generate code and save as JSON
print(">> Generating code...")
os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
os.makedirs(cpp_output_dir, exist_ok=True)


# Generate code
results = []

for idx, item in enumerate(tqdm(prompts_data, desc="Generating")):
    prompt_text = item.get("nl_prompt", "")
    prompt_id   = str(item.get("Prompt ID", "")).strip() or None

    generated_text = generate_one(model, tokenizer, prompt_text, max_input_len=1536, max_new_tokens=512)

    results.append({
        "Prompt ID": prompt_id if prompt_id is not None else f"{idx+1}",
        "nl_prompt": prompt_text,
        "generated_code": generated_text
    })

## Save JSON
print(f">> Saving generated results to {json_output_path}")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

## Extract C/C++ code and save to file
print(f">> Extracting cpp code to directory: {cpp_output_dir}")
saved = 0
for i, item in enumerate(results):
    raw_code = item.get("generated_code", "")
    code = extract_cpp_code(raw_code)
    input_code = code.replace("    ", "\t")+'\n'

    pid = item.get("Prompt ID")
    if pid and pid != "unknown":
        safe_pid = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", str(pid))[:64]
        filename = f"Llama_3.1_8b_demo_{safe_pid}.cpp"
    else:
        filename = f"Llama_3.1_8b_demo_{i+1:04d}.cpp"

    filepath = os.path.join(cpp_output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as w:
        w.write(code)
    saved += 1

print(f">> Extraction complete. {saved} files saved to {cpp_output_dir}")


# Vulnerability detection model
detector_tokenizer = AutoTokenizer.from_pretrained("eunJ/codebert_vulnerabilty_detector")
detector_model = AutoModelForSequenceClassification.from_pretrained("eunJ/codebert_vulnerabilty_detector")

from vulnerability_detector import print_result, detect_code

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5 

prob, is_vul = detect_code(detector_model, detector_tokenizer, input_code, DEVICE, THRESHOLD)
print_result("Sample", code, prob, is_vul, THRESHOLD)