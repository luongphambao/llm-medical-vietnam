import pandas as pd
import numpy as np

class Medical_Data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.public_test = pd.read_csv(data_path + 'public_test.csv')
        self.private_test = pd.read_csv(data_path + 'private_test.csv')
        self.gt_public_test = pd.read_csv(data_path + 'medical_public_test_label.csv')
        self.gt_private_test = pd.read_csv(data_path + 'medical_private_test_label.csv')
        self.prompt_template = """Sử dụng các trích đoạn sau đây để trả lời câu hỏi trắc nghiệm. Nếu không biết, chọn không biết, nếu có nhiều hơn 1 đáp án đúng thì liệt kê các đáp án đúng.
                    {context}
                    Câu hỏi: {question}
                    Trả lời:"""
    def load_public_data(self):
        df_public = self.public_test
        df_public['answer'] = self.gt_public_test['answer']
        return df_public
    def load_private_data(self):
        df_private = self.private_test
        df_private['answer'] = self.gt_private_test['answer']
        return df_private
    def final_data(self):
        df_public = self.load_public_data()
        df_private = self.load_private_data()
        df = pd.concat([df_public, df_private], axis=0)
        return df
if __name__ == "__main__":
    data_class = Medical_Data('data/')
    public_data = data_class.load_public_data()
    private_data = data_class.load_private_data()
    public_data.to_csv("data/public_test_ans.csv", index=False)
    private_data.to_csv("data/private_test_ans.csv", index=False)
    final_data = data_class.final_data()
    final_data.to_csv("data/final_test_ans.csv", index=False)