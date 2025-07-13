import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set model ID
# comment out the model you want to use
# model_id = "gpt2" # for testing purposes only
# model_id = "deepseek-ai/deepseek-coder-1.3b"
# model_id = "deepseek-ai/deepseek-coder-1.3b-base"
model_id = "deepseek-ai/deepseek-coder-1.3b-instruct"


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_code(prompt):
    if not prompt.strip():
        return "⚠ Please enter a valid prompt."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip the prompt if it appears at the start
    if output_text.startswith(prompt):
        output_text = output_text[len(prompt):].lstrip()

    return output_text

demo = gr.Interface(
    fn=generate_code,
    inputs=gr.Textbox(lines=5, label="Enter Prompt"),
    outputs=gr.Textbox(label="Generated Output"),
    title="Code Generator using DeepSeek"
)

demo.launch()