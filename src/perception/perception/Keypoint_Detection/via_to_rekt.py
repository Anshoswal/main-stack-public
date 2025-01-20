# MAKE SURE PHYSICAL POINTS ARE TAKEN TOP TO BOTTOM, LEFT TO RIGHT ON CONE
import pandas as pd

df = pd.read_csv('owo_csv.csv')
df = df[['filename', 'region_id', 'region_shape_attributes']]

# res = pd.DataFrame([], columns=['', '', 'top', 'mid_L_top', 'mid_R_top', 'mid_L_bot', 'mid_R_bot', 'bot_L', 'bot_R'])
res = pd.read_excel('output_test.xlsx') #output path

for file in df['filename'].unique():
    df1 = df[df['filename'] == file].drop('filename', axis=1)
    df1 = df1.sort_values(by=['region_id']).reset_index()
    xys = [file, 'N/A']

    for pt in range(7):
        coords = df1['region_shape_attributes'][pt]
        coords = coords.split(':')
        x = coords[2].split(',')[0]
        y = coords[3].split('}')[0]
        xys.append(f'[{x},{y}]')

    res.loc[len(res.index)] = xys

print(df)
print(res)

oppath = 'output_test'
res.to_excel(f'{oppath}.xlsx', index = False)
