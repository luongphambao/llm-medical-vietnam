# coding: utf8
import torch, accelerate,transformers, einops
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_path = "PhoGPT-4B-Chat-v0.1"
#model_path = "vinai/PhoGPT-4B-Chat"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
config.init_device = "cuda"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, use_auth_token=None,config=config)
# If your GPU does not support bfloat16:
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"
def model_generator(model, instruction):
  input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})
  input_ids = tokenizer(input_prompt, return_tensors="pt")

  outputs = model.generate(
  inputs=input_ids["input_ids"].to("cuda"),
  # attention_mask=input_ids["attention_mask"].to("cuda"),
  do_sample=True,
  temperature=1.0,
  top_k=50,
  top_p=0.9,
  max_new_tokens=1024,
  eos_token_id=tokenizer.eos_token_id,
  pad_token_id=tokenizer.pad_token_id,
  # return_dict_in_generate=True, output_scores=True
  # output_hidden_states = True,
  )

  response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
  response = response.split("### Trả lời:")[1]
  return response
     

# Some instruction examples
# instruction = "Viết bài văn nghị luận xã hội về {topic}"
# instruction = "Viết bản mô tả công việc cho vị trí {job_title}"
# instruction = "Sửa lỗi chính tả:\n{sentence_or_paragraph}"
# instruction = "Dựa vào văn bản sau đây:\n{text}\nHãy trả lời câu hỏi: {question}"
# instruction = "Tóm tắt văn bản:\n{text}"

#instruction = "Viết bài văn nghị luận xã hội về an toàn giao thông"
# instruction = "Sửa lỗi chính tả:\nTriệt phá băng nhóm kướp ô tô, sử dụng \"vũ khí nóng\""

instruction = "Hãy đưa ra 10 ý tưởng về marketing bán hàng và các bước để thực hiện những ý tưởng đó"
     

# model_path = "PhoGPT-4B-Chat-v0.1"
#instruction = "em tên là gì? em nhà ở đâu thế? sở thích của em là gì?"
print(model_generator(model, instruction))