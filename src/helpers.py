import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score

def prepare_df(df):
    #the following column is empty
    df.drop('business_neighborhoods', axis=1, inplace=True)
    #going to get only restaurants
    df = df[df['business_categories'].str.contains('Restaurant') == True]
    #grabbing the six empty reviews entries 
    no_reviews = df[pd.isna(df['text'])==True].index
    #dropping empty reviews and resetting index
    df.drop(labels=no_reviews, axis=0, inplace=True)
    df = df.reset_index().drop('index', axis=1)

    
    # now going to group 1-5 ratings into 3 categories: 
    # 1, 2, 3 (Bad, Neutral, and Good, respectively)

    df['stars'] = df['stars'].apply(lambda x: 1 if x <= 2 else x)
    df['stars'] = df['stars'].apply(lambda x: 2 if x == 3 else x)
    df['stars'] = df['stars'].apply(lambda x: 3 if x >=4 else x)

    return df

def calc_scores(y_true, y_predict):
    precision = precision_score(y_true, y_predict,average='micro')
    accuracy = accuracy_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict, average='micro')
    print(f'Precision : {precision} \n'  
          f'Accuracy : {accuracy} \n'
         f'Recall: {recall}')
    return [precision, accuracy, recall]

def check_class_balance(df, column, labels):
    total = len(df)
    classes = {'total':total}
    for i, label in enumerate(labels):
        classes[label] = len(df[df[column] == label])
    return classes



def downsample(df:pd.DataFrame, label_col_name:str) -> pd.DataFrame:
    # find the number of observations in the smallest group
    nmin = df[label_col_name].value_counts().min()
    return (df
            # split the dataframe per group
            .groupby(label_col_name)
            # sample nmin observations from each group
            .apply(lambda x: x.sample(nmin))
            # recombine the dataframes
            .reset_index(drop=True)
           )

def hard(soft_hat):
    hard_hat = []
    for row in soft_hat:
        cls = int(np.where(row == np.max(row))[0]) + 1
        hard_hat.append(cls)
    hard_hat = np.array(hard_hat)
    return hard_hat
