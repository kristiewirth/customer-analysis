import pandas as pd
import progressbar

# Reading in csv
meta_df = pd.read_csv('meta-data.csv', error_bad_lines=False)

# Cutting rows that are random code
meta_df = meta_df[[type(x) == int or x.isdigit() for x in meta_df['meta_id']]]

bar = progressbar.ProgressBar()
lst = []
for row in bar(meta_df.iterrows()):
    # Putting each row in a dictionary
    d = {}
    column = row[1]['meta_key']
    value = row[1]['meta_value']
    post_id = row[1]['post_id']
    d[column] = value
    d['post_id'] = post_id
    # Appending each row's dictionary to a list
    lst.append(d)


bar = progressbar.ProgressBar()
meta_cleaned_df = pd.DataFrame()
for my_dict in bar(lst):
    # Changing each row's dictionary into a df
    row = pd.DataFrame.from_dict(my_dict, orient='index').transpose()
    # Concatenating all rows into one df
    meta_cleaned_df = pd.concat([meta_cleaned_df, row], axis=0)

# Condensing meta df into one row per post id
meta_cleaned_df = meta_cleaned_df.groupby('post_id').first()

# Exporting to csv
meta_cleaned_df.to_csv('cleaned_meta_df.csv')
