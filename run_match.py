#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import process, fuzz

import multiprocessing as mp


# load the data
def load_data():
    ex = pd.read_csv("ex.csv", delimiter=";")
    ex.columns = ["ext_original_name", "ext_nation", "ext_region"]

    inter = pd.read_csv("int.csv", delimiter=";")
    inter.columns = ["customer_id", "int_original_name",
                     "int_nation", "int_region"]

    # remove the company type from the name
    ex['clean_name'] = remove_words(ex['ext_original_name'])

    # inter split the region from name
    int_name_region = inter["int_original_name"].str.split(',', expand=True)
    inter['clean_name'] = remove_words(int_name_region[0])
    # inter["int_sub_region"] = int_name_region[1]
    del int_name_region

    # drop the duplicated rows
    clean_ex_d = ex.drop_duplicates()
    clean_int = inter.drop_duplicates()

    # remove the un-know company
    clean_ex_d = clean_ex_d.loc[clean_ex_d["clean_name"] != "unknown", :]
    clean_int = clean_int.loc[clean_int["clean_name"] != "unknown", :]
    return clean_ex_d, clean_int


def remove_words(original_name_series):
    # lower the word
    original_name_series = original_name_series.str.lower()

    # remove the special characters
    original_name_series = original_name_series.str.replace(
        '[^\w\s]', '', regex=True)

    remove_word = ["llc", "ltd", 'limited', "co", 'corp', "inc", 'bv',
                   'holding', 'holdings', 'plc', 'group', 'bvba', 'sa']
    remove_word = r'\b(?:{})\b'.format('|'.join(remove_word))
    new_name_series = original_name_series.str.replace(remove_word, '',
                                                       regex=True)

    # remove the extra space
    new_name_series = new_name_series.str.replace(r'\s+', ' ', regex=True)
    new_name_series = new_name_series.str.strip()

    return new_name_series


# count the freq words
def get_word_freq(series, top=20):
    return pd.Series(' '.join(series).split()).value_counts()[:top]


def get_best_match(df, scores_):
    if len(df) > 0:
        best_match_name = df[scores_].T.idxmax().str.replace("_score",
                                                             "_match")
        for name in best_match_name.unique():
            df.loc[best_match_name == name, "final_match"] = df[name]
    else:
        df["final_match"] = np.nan
    return df


def get_region_match(df, region_int, match_ration_dic):
    ex_name_length = df["clean_name"].str.len()
    scores_ = []

    for ratio_name, ratio_ in match_ration_dic.items():
        # match with different functions
        region_matched = df["clean_name"].apply(lambda x: process.extractOne(
            x, region_int["clean_name"], scorer=ratio_))

        match_name = f"{ratio_name}_match"
        match_score = f"{ratio_name}_score"
        scores_.append(match_score)
        if len(region_matched) > 0:
            df[[match_name, match_score]] = pd.DataFrame(
                region_matched.tolist(), index=df.index)[[0, 1]]

            # replace the lower score result with nan
            length_different = abs(
                ex_name_length - df[match_name].str.len()) / ex_name_length
            nan_match_indicate = df[match_score] < 89
            nan_match_indicate |= length_different > 0.31

            df.loc[nan_match_indicate, match_name] = np.nan
            df[match_score + "_new"] = df[match_score] * (1 - length_different)
        else:
            df[[match_name, match_score, match_score + "_new"]] = np.nan

    # clean the match results, return the best one
    df = get_best_match(df, scores_)
    return df


def run_match(ex_df, int_df):
    match_ration_dic = {"ratio": fuzz.ratio,
                        "token_sort_ratio": fuzz.token_sort_ratio}

    all_region_result = []
    for region, single_region_ex in tqdm(ex_df.groupby("ext_region")):
        region_int = int_df.loc[int_df["int_region"] == region]

        # do the match
        n_cores = (mp.cpu_count() - 1)
        split_df = np.array_split(single_region_ex, n_cores)
        in_arg = []
        for i in range(len(split_df)):
            if split_df[i].size > 0:
                in_arg.append([split_df[i], region_int, match_ration_dic])
            else:
                in_arg.append([pd.DataFrame(columns=single_region_ex.columns),
                               region_int, match_ration_dic])

        with mp.Pool(processes=n_cores) as multi_process:
            results = multi_process.starmap(get_region_match, in_arg)
        single_region_rs = pd.concat(results)
        assert len(single_region_rs) == len(single_region_ex)

        single_region_rs = single_region_rs.merge(
            region_int,
            left_on=f"final_match",
            right_on="clean_name",
            how="left",
            suffixes=('', f'_matchedby')).drop(
            f'clean_name_matchedby', axis=1)
        all_region_result.append(single_region_rs)

    # concat different region results
    match_df = pd.concat(all_region_result)

    # save the result
    print("save results ...")
    internal_info_col = ["final_match", "customer_id", "int_original_name",
                         "int_nation", "int_region"]
    final_mathch = match_df[
        ex_df.columns.tolist() + internal_info_col].dropna(
        subset=["final_match"], how='all').drop(["clean_name","final_match"],axis=1)

    final_mathch.to_excel("final_mathched.xlsx",index=False) 
    match_df.to_excel("match_result_info.xlsx",index=False) 

    return final_mathch, match_df


if __name__ == "__main__":
    clean_ex_df, clean_int_df = load_data()

    final_mathched, match_details = run_match(clean_ex_df, clean_int_df)
