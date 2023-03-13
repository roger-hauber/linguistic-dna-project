import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def load_data(path: str, file: str, sep: str =",") -> pd.DataFrame:
    '''
    Loads data set into a dataframe
    '''
    df = pd.read_csv(f"{path}{file}", sep=sep, header=0)
    print('Data loaded')
    return df


def remove_missings(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Deletes samples with missings in accent, age, and/or gender
    '''
    df = df.dropna(subset=['accent', 'age', 'gender'])
    return df


def select_by_accents(df: pd.DataFrame, target_accents: list) -> pd.DataFrame:
    '''
    Only includes samples with accents from target accents list
    '''
    df = df[df.accent.isin(target_accents)]
    return df


def select_by_age_groups(df: pd.DataFrame, target_ages: list) -> pd.DataFrame:
    '''
    Only includes samples from age groups from target ages list
    '''
    df = df[df.age.isin(target_ages)]
    return df


def select_by_num_words(df: pd.DataFrame, min_num_words: int) -> pd.DataFrame:
    '''
    Counts number of words per sample and drops rows with less number of words than min_num_words
    '''
    df["num_words"] = df["sentence"].str.split().apply(len)
    df = df[df["num_words"] >= min_num_words]
    return df


def clear_dataframe(df: pd.DataFrame,
                    target_accents: list = ['us', 'canada', 'australia', 'england', 'indian'],
                    target_ages: list = ['twenties', 'thirties', 'fourties'],
                    min_num_words: int = 9) -> pd.DataFrame:
    '''
    Combines functions above: removes missings,
    selects samples by accents, by age groups, by number of words
    '''
    df = remove_missings(df)
    df = select_by_accents(df, target_accents)
    df = select_by_age_groups(df, target_ages)
    df_presampled = select_by_num_words(df, min_num_words)

    print('Dataframe ready to be splitted & balanced')
    return df_presampled

def downsampling_per_person(df_presampled: pd.DataFrame, max_num_samples: int = 5) -> pd.DataFrame:
    '''
    including max_num_samples per person
    '''
    df_downsampled = df_presampled.groupby(by=["client_id", "accent"]).head(max_num_samples).reset_index()
    return df_downsampled


def group_per_person(df_presampled: pd.DataFrame) -> pd.DataFrame:
    '''
    Groups data by person,
    includes num_samples = count of samples per person,
    age, gender,
    num_words = count of number of words per person,
    '''
    df_grouped_person = df_presampled.groupby(by=["client_id", "accent"]).agg(
        {"path": "count", "age": "min", "gender": "min", "num_words": "sum"}).reset_index()
    df_grouped_person.rename(columns={"path": "num_samples"}, inplace=True)

    return df_grouped_person


def get_demographs_per_accent(df_presampled: pd.DataFrame) -> pd.DataFrame:
    '''
    Creates and displays a Daraframe with number of samples per gender and accents
    '''
    df_grouped_person = group_per_person(df_presampled)

    df_demographs_per_accent = df_grouped_person.groupby(by=["accent", "gender"]).agg(
        {"client_id": "count", "num_samples": "sum", "num_words": "min"})
    df_demographs_per_accent.rename(columns={"client_id": "num_persons"}, inplace=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_demographs_per_accent)

    return df_demographs_per_accent


def get_min_num_persons(df_presampled: pd.DataFrame) -> int:
    temp = df_presampled[(df_presampled['gender'] == 'male') | (df_presampled['gender'] == 'female')] # exclude gender == 'other'

    df_demographs_per_accent = get_demographs_per_accent(temp)

    min_gender = df_demographs_per_accent[df_demographs_per_accent['num_persons']
                                          == df_demographs_per_accent['num_persons'].
                                          min()].reset_index().iloc[0, 1]
    min_num_persons = df_demographs_per_accent[df_demographs_per_accent['num_persons']
                                               == df_demographs_per_accent['num_persons'].
                                               min()].reset_index().iloc[0, 2]

    print(f'Gender of smallest sample in set: {min_gender}')

    if min_gender == 'female':
        n_female = min_num_persons
        return n_female
    return None


def get_target_ids(df_presampled: pd.DataFrame,
                   n_female: int,
                   n_gender_ratio: int = 1,
                   target_accents: list = ['us', 'canada', 'australia', 'england', 'indian']) -> list:
    '''
    Creates table with n_female and n_male samples per accent,
    returns a list of client_ids of these samples
    '''
    gender = ['female', 'male']
    n_male = n_female * n_gender_ratio
    n = {'female': n_female, 'male': n_male}
    i = 0

    df_grouped = group_per_person(df_presampled)

    for a in target_accents:
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


def get_balanced_data(df_presampled: pd.DataFrame,
                      target_ids: list,
                      max_num_samples: int = 5) -> pd.DataFrame:
    '''
    Creates balanced table from target_ids,
    returns dataframe
    '''
    i = 0

    for id in target_ids:
        # extracting samples
        count = df_presampled[df_presampled['client_id']== id].count()['client_id'] # counting number of samples per client_id
        if count < max_num_samples:
            temp = df_presampled[df_presampled['client_id'] == id].sample(n=count) # taking these samples if less than 5
        else:
            temp = df_presampled[df_presampled['client_id'] == id].sample(n=max_num_samples) # taking a x random samples if more than x samples per client_id

        # appending dataframe
        if i == 0:
            df_balanced = temp
        else:
            df_balanced = pd.concat([df_balanced, temp], axis=0)
        i += 1

    print('Balanced data set created')
    return df_balanced


def save_balanced_set(df_balanced, path):
    '''
    Saves the balanced data set locally to path
    '''
    df_balanced.to_csv(f'{path}balanced.csv', index=False)


def create_balanced_set(path: str,
                        file: str,
                        target_accents: list = ['us', 'canada', 'australia', 'england', 'indian'],
                        target_ages: list = ['twenties', 'thirties', 'fourties'],
                        min_num_words: int = 9,
                        max_num_samples: int = 5,
                        n_female: int = 67,
                        n_gender_ratio: int = 1,
                        test_size: float = 0.3,
                        save_test: bool = False,
                        save_balanced: bool = False):
     # load raw dataset
    df = load_data(path, file)

    # delete rows with missings, non-used accents, word count smaller 9, non-used age groups
    df_sampled = clear_dataframe(df, target_accents, target_ages, min_num_words)

    # split into train and test set, save test set if save_test=True (not the default)
    df_train, df_test = train_test_split(df_sampled, test_size=0.3, random_state=0)
    print('data splitted into train and test set')
    if save_test:
        df_test.to_csv(f'{path}test_set.csv', index=False)

    # get smallest sample of genderxaccent
    #df_demographics_per_accent = get_demographs_per_accent(df_train)
    n_female = get_min_num_persons(df_train)

    # get client_ids that will be included in balanced data set
    target_ids = get_target_ids(df_train, n_female, n_gender_ratio)

    # create balanced data set, save if save_balanced=True (not the default)
    df_train_balanced = get_balanced_data(df_train, target_ids)
    if save_balanced:
        save_balanced_set(df_train_balanced, path)

    return df_train_balanced, df_test

# create_balanced_set(path = '../raw_data/', file= 'validated.tsv')
