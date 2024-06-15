#ollama serve
import ollama

model_id ="ontocord/vistral"
for i in range(200):
    prompt = open("prompt.txt").read()
    print(prompt)
    respone = ollama.generate(model=model_id, prompt=prompt)
    print(respone)
