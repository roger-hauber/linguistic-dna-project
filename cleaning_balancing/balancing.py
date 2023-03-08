'''
To do:
- include train/test split and infer maximum n_female and n_male from df_train
- accents and age categories could be made into parameters as well
'''

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# importing raw data
def load_data(path):
    df = pd.read_csv(f"{path}validated.tsv", sep="\t", header=0)
    print('Data loaded')
    return df


# creating copy of dataframe and customizing it
def delete_rows(df, min_num_words = 9):
    '''
    Deleting samples with missings in accent, age, gender,
    Deleting samples with accent other than our 5 target accents,
    Deleting samples with age equals teens or more than fourty,
    Deleting samples with less words than min_num_words (default = 9)
    '''
    # Dropping rows with missing values
    df = df.dropna(subset=['accent', 'age', 'gender'])

    # Choose accents
    df = df[(df['accent'] == 'australia') | (df['accent'] == 'canada') | (df['accent'] == 'indian')
            | (df['accent'] == 'england') | (df['accent'] == 'us')]

    # Choose age
    df = df[(df['age'] == 'twenties') | (df['age'] == 'thirties') | (df['age'] == 'fourties')]

    # Count number of words per sample and drop rows with less number of words than min_num_words
    df["num_words"] = df["sentence"].str.split().apply(len)
    print('Number of words counted')
    df = df[df["num_words"] >= min_num_words]

    print('Rows deleted')
    return df


# Creating a dataframe grouped by client_id to get number of samples per person
def group_per_person(df):
    '''
    Use df with deleted rows;
    Group data by person, include num_samples = count of samples per person, age,
    gender, num_words = total number of words per person
    '''
    df_grouped_person = df.groupby(by=["client_id", "accent"]).agg({"path": "count", "age": "min", "gender": "min", "num_words": "sum"}).reset_index()
    df_grouped_person.rename(columns={"path": "num_samples"}, inplace=True)

    return df_grouped_person


# Creating a dataframe additionally grouped by accent to get an overview of number of persons and samples per accent
def group_per_accent(df):
    '''
    Use df grouped by person;
    Group data additionally by accent, include num_persons= count of persos per accents,
    num_samples = total of samlpes per accent, num_words = minimum number of words in a sample,
    '''
    df_grouped_accent = df.groupby(by="accent").agg({"client_id": "count", "num_samples": "sum", "num_words": "min", "age": "min", "gender": "min"})
    df_grouped_accent.rename(columns={"client_id": "num_persons"}, inplace=True)

    return df_grouped_accent


# Creating an overview of female / male samples for accents
def show_demographs_per_accent(df):
    '''
    Use df with deleted rows;
    Shows overview
    '''
    df = group_per_person(df)

    df = df.groupby(by=["client_id", "accent"]).agg({"path": "count", "age": "min", "gender": "min", "num_words": "sum"}).reset_index()
    df.rename(columns={"path": "num_samples"}, inplace=True)
    df_demographs_per_accent = df.groupby(by=["accent", "gender"]).agg({"client_id": "count", "num_samples": "sum", "num_words": "min"})

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_demographs_per_accent)


# get a list with client_ids to then create a balanced data set with
def get_target_ids(df, n_female, n_male):
    '''
    Use df with deleted rows;
    Creates table with n = 67 of both female and male samples per accent,
    returns a list of client_ids of these samples
    '''
    accents = ['australia', 'england', 'indian', 'canada', 'us']
    gender = ['female', 'male']
    n = {'female': n_female, 'male': n_male}
    i = 0

    df_grouped = group_per_person(df)

    for a in accents:
        for g in gender:
            temp = df_grouped[(df_grouped['accent'] == a) & (df_grouped['gender'] == g)].sample(n[g])
            if i == 0:
                data = temp
            else:
                data = pd.concat([data, temp], axis=0)
            i += 1

    target_ids = data['client_id']

    print('Target ids created')
    return target_ids


# create a balanced data set according to the list with client_ids
def get_balanced_data(df, target_ids):
    '''
    Use df with deleted rows;
    Creates balanced table from target_ids,
    returns dataframe
    '''
    i = 0

    for id in target_ids:
        # extracting samples
        count = df[df['client_id']== id].count()['client_id'] # counting number of samples per client_id
        if count < 5:
            temp = df[df['client_id'] == id].sample(n=count) # taking these samples if less than 5
        else:
            temp = df[df['client_id'] == id].sample(n=5) # taking a 5 random samples if more than 5 samples per client_id

        # appending dataframe
        if i == 0:
            final_data = temp
        else:
            final_data = pd.concat([final_data, temp], axis=0)
        i += 1

    print('Balanced data set created')
    return df


# saving the balanced data set locally to raw data
def save_balanced_set(df, path):
    df.to_csv(f'{path}balanced.csv', index=False)


def create_balanced_set(path, n_male = 67, n_female = 67, test_size = 0.3, save_test = False, save_balanced = False):
     # load raw dataset
    df = load_data(path)

    # delete rows with missings, non-used accents, word count smaller 9, non-used age groups
    df = delete_rows(df, min_num_words = 9)

    #  # split into train and test set, save test set if save_test=True (not the default)
    # df_train, df_test = train_test_split(df, test_size)
    # print('data splitted into train and test set')
    # if save_test:
    #     df_test.to_csv(f'{path}test_set.csv', index=False)

     # get client_ids that will be included in balanced data set
    target_ids = get_target_ids(df, n_male, n_female)

    # create balanced data set, save if save_balanced=True (not the default)
    df_train_balanced = get_balanced_data(df, target_ids)
    if save_balanced:
        save_balanced_set(df, path)

    return df_train_balanced

create_balanced_set(path = '../raw_data/')
