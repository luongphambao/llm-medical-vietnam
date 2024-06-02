#ollama serve
import ollama

model_id ="ontocord/vistral"
prompt = open("prompt.txt").read()
print(prompt)
respone = ollama.generate(model=model_id, prompt=prompt)
print(prompt.respone)
