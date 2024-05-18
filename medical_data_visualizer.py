import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')


# 2
# create an boolean overweight column with BMI (weight(kg)/height(m)^2) > 25
df['overweight'] = (df['weight'] / (df['height']/100)**2) > 25
# convert boolean overweight column to int
df['overweight'] = df['overweight'].astype(int)

# 3
# replace cholesterol values (1 to 0, 2 and 3 to 1)
df['cholesterol'] = df['cholesterol'].replace({1: 0, 2: 1, 3: 1})
# replace glucose values (1 to 0, 2 and 3 to 1)
df['gluc'] = df['gluc'].replace({1: 0, 2: 1, 3: 1})

# 4

def draw_cat_plot():
    # 5
    # melt/unpivot the dataframe into df_cat
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # 6
    # group and show counts of unique values of df_cat
    df_cat = df_cat.groupby(['cardio'], as_index=False).value_counts()
    # sort df_cat by variables
    df_cat = df_cat.sort_values(by=['variable'])
    df_cat = df_cat.rename(columns={"count": "total"})

    # 7
    # plot a bar chart
    sns.catplot(
        data=df_cat, x='variable', y="total", col="cardio", kind="bar",hue="value")


    # 8
    fig = sns.catplot(
        data=df_cat, x='variable', y="total", col="cardio", kind="bar",hue="value").fig

    # 9
    fig.savefig('catplot.png')
    return fig
draw_cat_plot()
# 10
def draw_heat_map():
    # 11
    # create a mask to filter less than 2.5th percentile and greater than 97.5th percentile in height and weight
    heat_mask = (df['ap_lo'] <= df['ap_hi']) & ((df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(1-0.025))) & ((df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(1-0.025)))
    # filter out dataframe
    df_heat = df[heat_mask]

    # 12
    corr = df_heat.corr()
    
    # 13
    # mask the top half of the correlation matrix
    mask = np.triu(np.ones_like(corr,dtype=bool))
    # 14
    fig, ax = plt.subplots(figsize=(10,8))

    # 15
    sns.heatmap(corr,mask=mask, annot=True, square=True, robust=True, fmt='.1f')

    # 16
    fig.savefig('heatmap.png')
    return fig
draw_heat_map()