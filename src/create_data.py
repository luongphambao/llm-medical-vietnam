import json 

data =json.loads(open("qa_medical_pairs.json").read())

# data_sample =json.load(open("sharegpt_data.json"))
# print(data_sample[0])
json_final = []
for i in data:
    print(i)
    human_question = i["question"]
    gpt_answer = i["answer"]
    dict_conversation = {}
    dict_question={"from":"human","value":human_question}
    dict_answer={"from":"gpt","value":gpt_answer}

    dict_conversation["conversation"] = []
    dict_conversation["conversation"].append(dict_question)
    dict_conversation["conversation"].append(dict_answer)
    #print(dict_conversation)
    json_final.append(dict_conversation)
with open("medical_conversation.json","w") as f:
    json.dump(json_final,f,indent=4)