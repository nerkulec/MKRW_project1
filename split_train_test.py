import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description='Dataset split into train/test sets followed by saving into indicated files.')
parser.add_argument(
    '-of', '--original_file', type=str, metavar='', required=True, help='Path to file with movie ratings.')
parser.add_argument(
    '-tr', '--train_file', type=str, metavar='', required=True, help='Training ratings save path')
parser.add_argument(
    '-ts', '--test_file', type=str, metavar='', required=True, help='Testing ratings save path')
# parser.add_argument(
#     '-r', '--split_ratio', type=int, metavar='', help='Train/test split ratio')
args = parser.parse_args()

FILE_PATH = '/Users/JakubMichalowski/Documents/UWR_II/methods_class_dim_red/labs/project_1/ml-latest-small/ratings.csv'
TEST_PATH  = '/Users/JakubMichalowski/Documents/UWR_II/methods_class_dim_red/labs/project_1/ml-latest-small/test_ratings.csv'
TRAIN_PATH = '/Users/JakubMichalowski/Documents/UWR_II/methods_class_dim_red/labs/project_1/ml-latest-small/train_ratings.csv'

def split_train_test(rating_file=FILE_PATH, train_file=TRAIN_PATH, test_file=TEST_PATH, train_size=0.9):
    '''This function aims to split data in a provided file with ratings into two seperate ones s.t. train file contains 90% of all entities/ratings of each user and the rest of user's ratings will be saved in test file. '''
    rating_df = pd.read_csv(filepath_or_buffer=rating_file)
    # print(rating_df.head())
    IDs = rating_df['userId'].unique()
    column_names = rating_df.columns
    train_df = pd.DataFrame(columns = column_names)
    test_df = pd.DataFrame(columns = column_names)

    for id in IDs:
        temp_df = rating_df[rating_df['userId'] == id]
        train_split, test_split = train_test_split(
            temp_df, train_size=train_size, random_state=42, shuffle=True)
        train_df = train_df.append(train_split, ignore_index=True)
        test_df  = test_df.append(test_split, ignore_index=True)
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print("Dataset of users' movie ratings has been split.")


if __name__ == '__main__':
    split_train_test(
        rating_file=args.original_file,
        train_file=args.train_file,
        test_file=args.test_file
    )

