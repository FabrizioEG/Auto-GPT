from autogpt import main

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Definir la prompt para la generación de texto
prompt = "Háblame sobre la historia de la pizza."

# Generar el texto utilizando la prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
generated_text = model.generate(
    input_ids=input_ids,
    max_length=500,
    temperature=0.7,
    top_p=0.92,
    repetition_penalty=1.2,
    do_sample=True,
    num_return_sequences=1,
)

# Decodificar el texto generado y mostrarlo
decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
print(decoded_text)
