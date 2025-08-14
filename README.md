# SKKU_SecLLM

This project uses a fine-tuned **LLaMA 3.1 8B** model to generate C/C++ code from natural language prompts and detect potential vulnerabilities in the generated code using a CodeBERT-based vulnerability detection model.

---

## 📂 Project Structure
```
.
├── dataset/
│   ├── selected_prompts_cpp.json   # Prompts for code generation
│   └── test.csv                    # Sample CSV for vulnerability detection
├── modules/
│   └── utils.py                    # Helper functions (code extraction, etc.)
├── extracted_cpp/                  # Generated C++ files (output)
├── demo.py                         # Code generation + extraction script
├── generate.py                     # Additional generation logic
├── vulnerability_detector.py       # Vulnerability detection logic
└── README.md                       # This file
```

---

## ⚙️ Installation
```bash
# 1. Clone this repository
git clone https://github.com/your-username/SKKU_SecLLM.git
cd SKKU_SecLLM

# 2. Create a Python environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 📜 Usage

### 1. Generate C++ Code from Prompts (Code Generation)
```bash
python generate.py
```
- Loads prompts from `dataset/selected_prompts_cpp.json`
- Generates C/C++ code using the fine-tuned LLaMA 3.1 8B model
- Saves results in:
  - `llama_3_1_8b_demo_generated.json` (full model outputs)
  - `extracted_cpp/` (only `.cpp` files extracted from the output)

---

### 2. Run Vulnerability Detection on CSV Samples (Detection)
```bash
python vulnerability_detector.py
```
- Reads `test.csv` from the `dataset/` folder
- Detects vulnerabilities for rows labeled as:
  - `generated_LLM = 'gpt4o'`
  - `generated_LLM = 'skku'`
- Prints probability and classification results in the console

---

### 3. Run Full Demo (Generation + Detection)
```bash
python demo.py
```
- Generates C/C++ code from prompts
- Extracts and saves `.cpp` files
- Runs vulnerability detection on generated code
- Prints results for each prompt including vulnerability probability and classification
---
