import pandas as pd


ori_file = '/disk3/yangle/diagnose/code/data/data_value.xlsx'
res_file = '/disk3/yangle/diagnose/code/data/data_change.xlsx'
threshold = [0, 500, 1000, 2600, 3000, 4000, 5300, 8000, 23000, 40000, 5942000]


def obtain_place(data):
    for i in range(1, 11):
        if (data > threshold[i-1]) and (data <= threshold[i]):
            return i * 0.1
    return


df = pd.read_excel(ori_file)
value = df['value'].to_list()

label = list()
for v in value:
    place = obtain_place(v)
    label.append(place)

raw_data = {'value': value, 'label': label}
df_new = pd.DataFrame(raw_data)
df_new.to_excel(res_file)
