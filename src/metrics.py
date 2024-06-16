import os
import numpy as np 
import pandas as pd 

def calc_score(pred,gt):
    """calculate score"""
    score = 0
    #count "1" in gt
    gt =str(gt)
    pred = str(pred)
    if len(gt) > len(pred):
        pred = pred + "0"*(len(gt)-len(pred))
    print("gt:",gt)
    print("pred:",pred)
    num_correct_answer = 0
    for i in range(len(gt)):
        
        if gt[i] == "1":
            num_correct_answer += 1
        if pred[i] == gt[i] and pred[i] == "1":
            score += 1
        else:
            if gt[i]=="0" and pred[i]!="0":
                return 0
    return score/num_correct_answer
            
class Metrics():
    def __init__(self,type="public"):
        self.data_public = pd.read_csv("data/public_test_ans.csv")
        self.data_private = pd.read_csv("data/private_test_ans.csv")
        self.type = type
        if type == "public":
            self.data = self.data_public
        else:
            self.data = self.data_private
    def get_score(self,df_submision):
        df_submision = df_submision.sort_values(by="id")
        df_gt = self.data.sort_values(by="id")
        #print("df_submision:",df_submision)
        #print("df_gt:",df_gt)
        for i in range(len(df_submision)):

            gt = df_gt.iloc[i]["answer"]
            pred = df_submision.iloc[i]["answer"]
            score = calc_score(pred,gt)
            #round to 3 decimal places
            df_submision.loc[i,"score"] = np.round(score,3)
        return df_submision
    def get_accuracy(self,df_submision):
        df_submision = df_submision.sort_values(by="id")
        df_gt = self.data.sort_values(by="id")
        for i in range(len(df_submision)):
            gt = df_gt.iloc[i]["answer"]
            pred = df_submision.iloc[i]["answer"]
            if gt == pred:
                df_submision.loc[i,"accuracy"] = 1
            else:
                df_submision.loc[i,"accuracy"] = 0
        return df_submision
    def get_result(self,df_submision):
        """get score and accuracy"""
        df_submision = self.get_score(df_submision)
        df_submision = self.get_accuracy(df_submision)
        accuracy = df_submision["accuracy"].mean()
        score = df_submision["score"].mean()
        return accuracy,score
if __name__ == "__main__":
    metrics = Metrics(type ="public")
    df_submision = pd.read_csv("submit_vietcuna.csv")
    accuracy,score = metrics.get_result(df_submision)
    print("accuracy:",accuracy)
    print("score:",score)