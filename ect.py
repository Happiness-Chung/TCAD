import pandas as pd

bbox_df = pd.read_csv('C:/Users/hb/Desktop/Data/CheXpert-v1.0-small/labels(former)/val_labels.csv')

pathes = bbox_df['Path'].values.tolist()

folder = 'CheXpert-v1.0-small'

for i in range(len(pathes)):
    
    row = bbox_df.iloc[i]

    path = row['Path']

    items = path.split('/')
    
    new_path = folder + '/train/' + items[2] + '/' + items[3] + '/' + items[4]

    bbox_df.at[i,'Path'] = new_path

print(bbox_df['Path'].values.tolist())    

bbox_df.to_csv('valid_as_train.csv', index=False)