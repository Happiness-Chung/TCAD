import pandas as pd

bbox_df = pd.read_csv('C:/Users/hb/Desktop/code/ICASC++/BBox_List_2017.csv')

imgs = bbox_df['Image Index'].values.tolist()

file_name = 'bbox_list.txt'

with open(file_name, 'w+') as file:
    file.write('\n'.join(imgs))

print(imgs)