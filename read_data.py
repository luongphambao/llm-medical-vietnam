import pandas as pd
import numpy as np

testset = 'data/public_test.csv'
df = pd.read_csv(testset)
ans = pd.DataFrame(columns=['id', 'answer'])

newline = '\n'
labels = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F'}
chars = "ABCDEF"

for index, row in df.iterrows():
    if index == 10:
        break
    options = [str(p) for p in row.loc[['option_' + str(i) for i in range(1,7)]] if str(p) != 'nan']
    options = [labels[i] + '. ' + p if not p.startswith(f'{labels[i]}.') else p for i, p in enumerate(options)]
    print(options)
    query = f"""{row.loc['question']}
{newline.join(options)}
    """
    print(query)
    #exit()
# #     print(sample_query)
#     response = qa(query) # only one character
#     binary_mask = ['1' if c == response['result'].strip() else '0' for c in chars[:len(options)]]
#     row_ans = ''.join(binary_mask)

#     ans.loc[index] = [row['id'], row_ans]

# ans.to_csv('submission.csv', index=False)