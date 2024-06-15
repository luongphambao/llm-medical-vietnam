from langchain.vectorstores import FAISS
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import torch, accelerate,transformers, einops
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer,pipeline,BitsAndBytesConfig, TextStreamer

def load_phogpt():
    model_path = "../PhoGPT-4B-Chat-v01"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.init_device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, use_auth_token=None,config=config)
    # If your GPU does not support bfloat16:
    # model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer
def load_vinallama():
    model_path = "vilm/vinallama-7b-chat"
    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # Seting config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.init_device = "cuda"
    config.temperature = 0.1
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,quantization_config=bnb_config,
        config=config,
        trust_remote_code=True
    )
    model.eval()
    return model, tokenizer
def load_vistral():
    model_path = "Viet-Mistral/Vistral-7B-Chat"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.init_device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, use_auth_token=None,config=config)
    # If your GPU does not support bfloat16:
    # model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer
def model_generator(model, question, context):
    prompt_template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết.
                    {context}
                    Câu hỏi: {question}
                    Trả lời:"""
    input_prompt = prompt_template.format_map({"context": context, "question": question})
    input_ids = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs=input_ids["input_ids"].to("cuda"),
        # attention_mask=input_ids["attention_mask"].to("cuda"),
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.9,
        max_new_tokens=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # return_dict_in_generate=True, output_scores=True
        # output_hidden_states = True,
        )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(response)
    #response = response.split("### Trả lời:")[1]
    return response
device = 'cpu'
embed_model_id = 'BAAI/bge-m3'
cache_dir = "cache/"
embed_model = SentenceTransformerEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'batch_size': 16, "normalize_embeddings": True, "device": device },
    cache_folder=cache_dir
)
DB_SAVE_NAME = f"endoc_{embed_model_id.split('/')[-1].replace('.','-')}"

loaded_db = FAISS.load_local(
    DB_SAVE_NAME,
    embeddings=embed_model
)
# level3_1,Hương đang mang thai và lo lắng mình có thể gặp phải rau tiền đạo. Hương có thể kiểm tra phát hiện bệnh này từ tuần thứ mấy của thai kỳ?,A. Tuần 10,B.Tuần 20,C. Tuần 30,D. Tuần 40,,
question = """Hương đang mang thai tuần thứ 5 và lo lắng mình có thể gặp phải rau tiền đạo. Hương có thể kiểm tra phát hiện bệnh này sau bao nhiêu tuần nữa?
A. 5 tuần
B. 15 tuần
C. 25 tuần
D. 35 tuần
"""

prompt_template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết.
{context}
Câu hỏi: {question}
Trả lời:"""
# query = """
# Đâu là triệu chứng của bệnh van tim?
# A. Khó thở
# B. Tăng cân nhanh chóng
# C. Vàng da
# D. Rụng tóc
# """

result = loaded_db.similarity_search(query=question, k=1)
print(result)
context = result[0].page_content
#system_prompt = """"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
input_prompt = prompt_template.format_map({"context": context, "question": question})
print(input_prompt)
with open("prompt.txt", "w") as f:
    f.write(input_prompt  )

import os

#load_dotenv()
with open("context.txt", "w") as f:
    f.write(context)
# model_phogpt, tokenizer = load_phogpt()
# result = model_generator(model_phogpt, question, context)
model_vinallama, tokenizer = load_vinallama()
result = model_generator(model_vinallama, question, context)
print(result)
# openai_api_key=os.getenv('OPENAI_API_KEY', 'sk-proj-aQI7CfOG3adLxGpE4zfKT3BlbkFJh3fnuL9nEPhU9HH3T2tj')
# llm = ChatOpenAI(temperature=0,openai_api_key=openai_api_key)
# result=llm.predict(text=PROMPT.format_prompt(
#     context=context,
#     question=question
# ).text)

print(result)