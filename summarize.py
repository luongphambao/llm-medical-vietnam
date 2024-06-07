from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(api_key=openai_api_key)
def build_final_context(chunks):
    context = ""
    for index, chunk in enumerate(chunks):
        context +=  f"Context {index + 1}: " + chunk + "\n"
    return context
def split_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Remove leading/trailing whitespace and split the content into sentences
    sentences = content.strip().split('.')

    # Remove empty sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Calculate the number of sentences per chunk
    total_sentences = len(sentences)
    chunk_size = total_sentences // 10

    chunks = []
    start = 0
    for i in range(9):
        end = start + chunk_size
        chunk = sentences[start:end]
        chunks.append('. '.join(chunk) + '.')
        start = end

    # Add the remaining sentences to the last chunk
    last_chunk = sentences[start:]
    chunks.append('. '.join(last_chunk) + '.')

    return chunks
class Prompts:

    @staticmethod
    def summarize_prompt(context):
        return f"""
        Sau đây là một nội dung được trích xuất của một văn bản, nhiệm vụ của bạn là hãy tóm tắt nó
        ---CONTEXT---
        {context}
        ---END CONTEXT---
        Hãy đưa ra tóm tắt của văn bản trên. Tôi chỉ cần phần tóm tắt, và không cần thêm bất kì thứ gì khác.
        Tóm Tắt:
        """

    @staticmethod
    def final_ans_prompt(context):
        return f"""
        Dưới đây là các đoạn tóm tắt của từng đoạn nhỏ của một văn bản lớn.
        ---CONTEXT---
        {context}
        ---END CONTEXT---
        Dựa trên các đoạn trên, hãy đưa ra tóm tắt tổng của tất cả các đoạn trên. Tôi chỉ cần tóm tắt tổng, không cần thêm bất kì thứ gì khác.
        Tóm tắt tổng:
        """
    

def summarize_chunk(chunk, results_queue, llm):
    message = [{"role": "user", "content": Prompts.summarize_prompt(chunk)}]
    response = llm.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=message,
        temperature=0.1,
        max_tokens=4096
    )
    summarized_chunk = response.choices[0].message.content
    results_queue.put(summarized_chunk) 
if __name__ == "__main__":
    data_path = "corpus/benh-cum"
    chunks = split_file(data_path)
    summarize_chunks = []
    question = """
    Hãy tóm tắt văn bản: {context}
    """

    summarize_prompt = """
    Sau đây là một nội dung được trích xuất của một văn bản, nhiệm vụ của bạn là hãy tóm tắt nó
    ---CONTEXT---
    {context}
    ---END CONTEXT---
    Hãy đưa ra tóm tắt của văn bản trên. Tôi chỉ cần phần tóm tắt, và không cần thêm bất kì thứ gì khác.
    Tóm Tắt:
    """

    final_ans_prompt = """
    Dưới đây là các đoạn tóm tắt của từng đoạn nhỏ của một văn bản lớn.
    ---CONTEXT---
    {context}
    ---END CONTEXT---
    Dựa trên các đoạn trên, hãy đưa ra tóm tắt tổng của tất cả các đoạn trên. Tôi chỉ cần tóm tắt tổng, không cần thêm bất kì thứ gì khác.
    Tóm tắt tổng:
    """
    for chunk in chunks:
        message = [{"role": "user", "content": summarize_prompt.format(context=chunk)}]
        response = llm.chat.completions.create(model='gpt-3.5-turbo',
                messages= message,
                temperature=0.1,
                max_tokens=4096)
        summarize_chunks.append(response.choices[0].message.content)

    context = build_final_context(summarize_chunks)
    message = [{"role": "user", "content": final_ans_prompt.format(context=context)}]

    final_response = llm.chat.completions.create(model='gpt-3.5-turbo',
        messages= message,
        temperature=0.1,
        max_tokens=4096)
    print(final_response)
    with open("benh-cum-summary.txt", "w") as file:
        file.write(final_response.choices[0].message.content)