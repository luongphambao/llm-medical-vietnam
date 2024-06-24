import json

DEFAULT_PROMPT = """<|im_start|>system\nYou are a helpful assistant in Medical Field. You were developed with the purpose of understanding the Vietnamese language to provide detailed, polite, and helpful answers to questions from humans in Vietnamese.<|im_end|>
<|im_start|>user\n{user_message}<|im_end|>
<|im_start|>assistant\n"""
DEFAULT_PROMPT_BLOSSOM = """<s>system\n [INST] You are a helpful assistant in Medical Field with the purpose of understanding the Vietnamese language to provide detailed, polite, and helpful answers to questions from humans in Vietnamese. [/INST]</s>
<s>user\n|Human|{user_message}</s>
<s>assistant\n"""
USER_MESSAGE = """Please answer the following multiple choice questions, each question has 2 to 6 options, of which at least one is correct:

Question:|Human|{question}

Multi answer choices:\n|Bot|{answer_choices}

For each question with multi choices, you need to return a json answer, where the key of the json is the ith choice and the value is is 0 if the ith choice in the question is false and vice versa.
If option is nan then the value is always 0. Please output your answer in JSON format:
{{  "A": "", 
    "B": "", 
    "C": "", 
    "D": "", 
    "E": "", 
    "F": "", }} 
"""

USER_MESSAGE_WITH_CONTEXT = """Please answer the following multiple choice questions based on the given context, each question has 2 to 6 options, of which at least one is correct:

Context:\n{context}

Question: {question}

Multi answer choices:\n{answer_choices}

For each question with multi choices, you need to return a json answer, where the value is is 0 if the key choice in the question is false and vice versa. If option is nan then the value is always 0.
Please output your answer in JSON format,:
{{  "A": "", 
    "B": "", 
    "C": "", 
    "D": "", 
    "E": "", 
    "F": "", }}
"""

USER_MESSAGE_WITH_CONTEXT_VER_2 = """根据越南语中有关与医疗问题相关的疾病的背景，通过选择相关字母来确定合适的答案。每个问题都会向您提供 2 到 6 个选项，标记为 A、B、C、D、E、F。请忽略问题中的任何非医疗信息。

# References：\n{context}

# User requirement： {question}

# Options：\n{answer_choices}

从给定的上下文中提取答案，即使问题包含误导性或无关的细节。请注意，有些问题可能有多个准确答案，而其他问题可能只有一个。如果该标签上的答案无效，则它不是正确答案. 为了清楚起见，仅输出选择的标签，您的答案可以采用以下格式：
例子：
如果 A 是正确答案，则必须回答：A
同样，如果 B 和 C 都是正确答案，它们必须回答：B、C
如果 A、B、C、D、E 都是正确答案，则必须回答：A、B、C、D、E
...

不要用我给你的例子作为结果，你必须自己回答问题。如果无法选择答案，请选择最接近 {question} 的答案"""

USER_MESSAGE_WITH_CONTEXT_QA = """Dựa trên nền tảng của vấn đề y tế liên quan đến bệnh bằng tiếng Việt, Hãy trả lời câu hỏi. Vui lòng bỏ qua bất kỳ thông tin không liên quan tới y tế có trong câu hỏi.

# References：\n{context}

# User requirement： {question}
Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp, ngay cả khi câu hỏi chứa các chi tiết gây hiểu lầm hoặc không liên quan,câu trả lời ngắn gọn xúc tích tập trung vào vấn đề của câu hỏi. 
"""

USER_MESSAGE_WITH_CONTEXT_VER_3 = """Dựa trên nền tảng của vấn đề y tế liên quan đến bệnh bằng tiếng Việt, Hãy trả lời câu hỏi bắt đầu bằng các nhãn liên quan. Mỗi câu hỏi sẽ cung cấp cho bạn từ 2 đến 6 lựa chọn, có nhãn A, B, C, D, E, F. Vui lòng bỏ qua bất kỳ thông tin không liên quan tới y tế có trong câu hỏi.

# References：\n{context}

# User requirement： {question}

# Options：\n{answer_choices}

Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp, ngay cả khi câu hỏi chứa các chi tiết gây hiểu lầm hoặc không liên quan. 
Lưu ý rằng một số câu hỏi có thể có nhiều hơn một câu trả lời chính xác, trong khi những câu hỏi khác có thể chỉ có một. Nếu câu trả lời tại nhãn đó không có giá trị thì đó câu trả lời đó không phải đáp án đúng
Để rõ ràng, chỉ các nhãn đã chọn mới được xuất ra và câu trả lời của bạn có thể ở dạng sau:
ví dụ：
Nếu A là câu trả lời đúng, bạn phải trả lời: A
Tương tự nếu B và C đều là đáp án đúng thì phải trả lời: B, C
Nếu A, B, C, D, E đều là đáp án đúng thì phải trả lời: A, B, C, D, E
...

Đừng sử dụng những ví dụ mà tôi đã đưa ra cho bạn như đáp án, bạn phải tự trả lời câu hỏi. Nếu không thể chọn đáp án nào, hãy chọn một đáp án mà gần với {question} nhất"""

DEFAULT_PROMPT_MISTRAL = """<s>[INST] system\nYou are a helpful assistant in Medical Field. You were developed by VillaLabs with the purpose of understanding the Vietnamese language to provide detailed, polite, and helpful answers to questions from humans in Vietnamese.<|im_end|>
<|im_start|>user\n{user_message}<|im_end|>
<|im_start|>assistant\n [/INST]"""

USER_MESSAGE_MISTRAL = """<s>[INST] Please answer the following multiple choice questions, each question has 2 to 6 options, of which at least one is correct:

Question: {question} [/INST]

Multi answer choices:\n{answer_choices} </s>[INST]

For each question with multi choices, you need to return a json answer, where the key of the json is the ith choice and the value is is 0 if the ith choice in the question is false and vice versa.
If option is nan then the value is always 0. Please output your answer in JSON format:
{{  "A": "", 
    "B": "", 
    "C": "", 
    "D": "", 
    "E": "", 
    "F": "", }} [/INST]
"""
USER_MESSAGE_MISTRAL_WITH_CONTEXT = """<s>[INST] Please answer the following multiple choice questions, each question has 2 to 6 options, of which at least one is correct:

Question: {question} [/INST]

Multi answer choices:\n{answer_choices} </s>[INST]

For each question with multi choices, you need to return a json answer, where the key of the json is the ith choice and the value is is 0 if the ith choice in the question is false and vice versa.
If option is nan then the value is always 0. For clarity, only the selected labels will be output, and your answer may be in the following form:
example:
If A is the correct answer, you must answer: A
Similarly, if B and C are both correct answers, they must answer: B, C
If A, B, C, D, E are all correct answers, they must answer: A, B, C, D, E [/INST]
"""

DEFAULT_PROMPT_VN = """<|im_start|>system\n Bạn là một trợ lý AI hữu ích của Bệnh viên đa khoa Tâm Anh chuyên môn trong lĩnh vực y học. Bạn được phát triển với kiến thức y tế từ bệnh viên đa khoa Tâm Anh nhằm mục đích hiểu rõ ngôn ngữ tiếng Việt nhằm cũng cấp một cách chính xác, rõ ràng những thắc mắc từ người bệnh. <|im_end|>
<|im_start|>user\n{user_message}<|im_end|>
<|im_start|>assistant\n"""

# USER_MESSAGE = """Hãy trả lời câu hỏi có nhiều đáp án dưới đây, mỗi câu hỏi có từ 2 tới 6 đáp án có thể chọn. Hãy đưa ra ít nhất một phương án trả lời đúng:

# Câu hỏi: {question}

# Những kết quả trả về:\n{answer_choices}

# Đối với mỗi câu hỏi có nhiều đáp án, bạn cần trả về cho tôi một danh sách câu trả lời định dạng JSON, mà giá trị key chính là các câu trả lời có thể xảy ra và giá trị value sẽ nhận giá trị 0 đối với câu trả lời sai và nhận giá trị 1 nếu câu trả lời đó đúng.
# Hãy đưa ra kết quả của bạn theo định dạng JSON ,:
# {{  "A": "", 
#     "B": "", 
#     "C": "", 
#     "D": "", 
#     "E": "", 
#     "F": "", }}
# """

# USER_MESSAGE_WITH_CONTEXT = """ Hãy trả lời câu hỏi có nhiều đáp án dưới đây dựa trên ngữ cảnh cho trước, mỗi câu trả lời có từ 2 tới 6 lựa chọn, và có ít nhất một lựa chọn đúng đối với mỗi câu hỏi:

# Ngữ cảnh:\n{context}

# Câu hỏi: {question}

# Những kết quả trả về:\n{answer_choices}

# Đối với mỗi câu hỏi có nhiều đáp án, bạn cần trả về cho tôi một danh sách câu trả lời định dạng JSON, mà giá trị key chính là các câu trả lời có thể xảy ra và giá trị value sẽ nhận giá trị 0 đối với câu trả lời sai và nhận giá trị 1 nếu câu trả lời đó đúng.
# Hãy đưa ra kết quả của bạn theo định dạng JSON ,:
# {{  "A": "", 
#     "B": "", 
#     "C": "", 
#     "D": "", 
#     "E": "", 
#     "F": "", }}
# """

# USER_MESSAGE_WITH_CONTEXT_VER_2 = """根据越南语中有关与医疗问题相关的疾病的背景，通过选择相关字母来确定合适的答案。每个问题都会向您提供 2 到 6 个选项，标记为 A、B、C、D、E、F。请忽略问题中的任何非医疗信息。

# # References：\n{context}

# # User requirement： {question}

# # Options：\n{answer_choices}

# 从给定的上下文中提取答案，即使问题包含误导性或无关的细节。请注意，有些问题可能有多个准确答案，而其他问题可能只有一个。为了清楚起见，仅输出选择的标签，您的答案可以采用以下格式：
# 例子：
# 如果 A 是正确答案，则必须回答：A
# 如果 A 和 B 都是正确答案，则必须回答：A、B
# 如果 B、D、E 都是正确答案，则必须回答：B、D、E
# 如果 A、B、C、D、E 都是正确答案，则必须回答：A、B、D、E、F
# ...

# 不要使用我给你的这些例子，你必须自己回答这个问题。"""
