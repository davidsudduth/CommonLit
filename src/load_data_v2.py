import pandas as pd
import numpy as np
from numpy import setdiff1d as diff
import random as random
import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import pickle
import re
import ipdb
import pdb


class LoadData:
    def __init__(self):

        self.test = None



    def load_data(self, filename):
        '''
        '''
        # read from csv
        df = pd.read_csv(filename)
        print('the primary data frame is loaded in\n')
        return df

        # # load small csv for testing
        # n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
        # s = 10000 #desired sample size
        # skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
        # df = pd.read_csv(filename, skiprows=skip)
        # return df


        # example=True
        #
        # if have_df:
        #     # get cached dataframes/ read from pickle
        #     df = pd.read_pickle('cache/df_train.pkl')
        #
        # else:
        #     if proportion < 1.0: #skip rows to reduce df size
        #         # nth row to keep
        #         n = int(1.0 / proportion)
        #         # length of dataset
        #         row_count = sum(1 for row in open('data/churn_train.csv'))
        #         # Row indices to skip
        #         skipped = [x for x in range(1, row_count) if x % n != 0]
        #     else:
        #         skipped = None
        # return df


    def load_data_spark(self):
        '''
        '''
        import pyspark as ps
        import csv


        spark = ps.sql.SparkSession.builder \
                .master("local[4]") \
                .appName("individual") \
                .getOrCreate()

        sc = spark.sparkContext
        sc.setLogLevel('ERROR')

        # filename = "raw_data/2017.08.02cfu_responses.csv"
        # dfs_cfu = spark.read.csv(filename, header=True, mode="DROPMALFORMED")

        filename2 = 'raw_data/2017.10.16text_responses.csv'
        dfs_tdq = spark.read.csv(filename2, header=True, mode="DROPMALFORMED")

        filename3 = 'raw_data/TDQs_2017-11-13_1944.csv'
        dfs_tdq_key = spark.read.csv(filename3, header=True, mode="DROPMALFORMED")

        filename4 = 'raw_data/tdqanswers.csv'
        dfs_tdq_answers = spark.read.csv(filename4, header=True, mode="DROPMALFORMED")


        return dfs_tdq, dfs_tdq_key, dfs_tdq_answers


    def load_sample_data(self, rowlimiter=100000):
        '''
        '''

        # CFU (check for understanding) responses by students
        n = 6259179 #number of rows in the file
        s = rowlimiter #desired sample size
        skip = sorted(random.sample(range(n),n-s))
        df_cfu = pd.read_csv('./raw_data/2017.08.02cfu_responses.csv', skiprows=skip)#, nrows=nrows) #500mb

        # TDQ (text dependent question) responses by students.
        n2 = 15363428 #number of rows in the file
        s2 = rowlimiter #desired sample size
        skip2 = sorted(random.sample(range(n2),n2-s2))
        df_tdq = pd.read_csv('./raw_data/2017.10.16text_responses.csv', skiprows=skip2)#, nrows=nrows) #1.7gb

        # TDQ (text dependent question) answer Key
        df_tdq_key = pd.read_csv('./raw_data/TDQs_2017-11-13_1944.csv')

        # CFU (check for understanding) answer key
        df_cfu_key = pd.read_csv('./raw_data/CFUs_2017-11-13_1942.csv')

        #TDQ (text dependent question) answers
        df_tdq_answers = pd.read_csv('./raw_data/tdqanswers.csv')

        print('the following sample (random columns) dataframes are now loaded:\n df_cfu,\n df_tdq')
        print('the following full dataframes are now loaded:\n df_tdq_key,\n df_cfu_key,\n df_tdq_answers')
        return df_cfu, df_tdq, df_tdq_key, df_cfu_key, df_tdq_answers


    def feature_saver(feat_list):
        '''
        '''
        with open('cache/feats.pkl','wb') as f: pickle.dump(feat_list,f)


    def feature_loader(self, df):
        '''
        '''
        # with open('cache/feats.pkl','rb') as f: feats = pickle.load(f)
        # print('initial df features: {}'.format(feats))
        # return feats
        feats = df.columns.tolist()
        print('df features: {}\n'.format(feats))
        return feats



class Preprocessor:
    def __init__(self):

        self.test = None



    def update_columns(self, df):
        '''
        '''
        columns = df.columns.values
        df.columns = [re.sub(r";:*&^%#@!,", "", columns[x]).lower() for x in range(len(columns))]
        return df


    def make_immediate_drops(self, df):
        '''
        '''
        drops = ['title1_schoolwide']

        # drops = ['unique',
        #          'month',
        #          'title1',
        #          'lea_trim',
        #          'group',
        #          'group_temp',
        #          'title1_info2',
        #          'title1_yes_mean_1',
        #          'title1_schoolwide',
        #          'title1',
        #          'month',
        #          'submit_time_raw',
        #          'filter_$']

        df = df.drop(drops, axis=1)
        print('dropped the following columns: {}'.format(drops))

        # remove lines that have only a " "
        # df = df[(df['filter_$']!=" ")]
        # df = df[(df['month']!=" ")]
        # df = df[(df['grade_id']!=" ")]

        return df


    def datetime_conversion(self, df):
        '''
        '''
        df.submit_time = pd.to_datetime(df.submit_time)
        return df


    def time_btwn_assignments(self, df):
        '''
        '''
        # calculate the time elapsed between last assignment
        temp = df.sort_values(['student_id', 'submit_time'])
        df['delta'] = (temp['submit_time']-temp['submit_time'].shift()).fillna(0)
        df['delta'] = df.delta.dt.days
        df['delta'] = df.delta.clip(lower=0)
        return df


    def apply_length_col(self, df, col):
        '''
        '''
        new_col = "len_" + col
        df[new_col] = df[col].apply(len)
        df.pop(col)
        return df


    def yes_no_col(self, df, col):
        '''
        '''
        df[col] = 1.0 *(df[col]=='t')
        return df


    def calc_class_size(self, df):
        '''
        '''
        # calculate number of students in each class
        class_size = df.groupby('class_roster_id')['student_id'].nunique().reset_index()
        class_size.columns = ['class_roster_id', 'class_size']
        df = df.merge(class_size, on='class_roster_id')
        return df


    def calc_stu_per_teacher(self, df):
        '''
        '''
        # calculate number of students per teacher
        stu_per_teacher = df.groupby('teacher_id')['student_id'].nunique().reset_index()
        stu_per_teacher.columns = ['teacher_id', 'stu_per_teacher']
        df = df.merge(stu_per_teacher, on='teacher_id')
        return df


    def calc_mean_first_scores(self, df):
        '''
        '''
        # calculate ave grade of the first two scores of a student
        first_scores = df.groupby('student_id').apply(lambda x: x.assignment_average[:2].mean()).reset_index()
        first_scores.columns = ['student_id', 'first_scores']
        df = pd.merge(df, first_scores, on='student_id')
        return df


    def calc_productivity(self, df):
        '''
        '''
        temp = df.sort_values(['student_id', 'submit_time'])
        b = temp.groupby('student_id').last()
        a = temp.groupby('student_id').first()
        c = (b['submit_time'] - a['submit_time']).reset_index() # / datetime.timedelta(days=1)).reset_index()

        df.pop('submit_time')
        result = pd.merge(df, c, on="student_id")
        result['sub_delta'] = result['submit_time']

        result['submit_time'] = (result['submit_time'] / np.timedelta64(1, 's')) / 604800
        result = result[result['sign_in_count'] >= 1]
        result['productivity'] = result['compltd_assigmts'] / result['submit_time'] # completed assignments per week
        result['productivity'] = pd.to_numeric(result['productivity'], downcast='float')
        result = result.dropna(how="any")

        return result


    def apply_param_constraints(self, df, class_size=5, min_spt=5, max_spt=115, n_assignments=6):
        '''
        '''
        # print('parameter contraints in load data function\n class_size={}, min_spt={}, max_spt={}, n_assignments={}'.format(class_size, min_spt, max_spt, n_assignments))
        # remove rows where class size is less than specified
        df = df[df['class_size'] >= class_size]
        # remove rows where student per teacher is less than contraint
        df = df[df['stu_per_teacher'] >= min_spt]
        # remove rows where student per teacher is more than contraint
        df = df[df['stu_per_teacher'] <= max_spt]
        # remove rows where students have only completed less than n_assignments to new df
        df = df[df['compltd_assigmts'] >= n_assignments]

        

        print('class size is at least {},\n     \
               students per teacher is at least {},\n     \
               students per teacher is less than {},\n     \
               number of completed assignments is at least {}\n'.format(class_size, min_spt, max_spt, n_assignments))

        return df


    def calculate_response(self, df):
        '''
        '''
        # calculate student improvement by takeing diffence between ave first two assessments and last two
        improvement = df.groupby('student_id').apply(lambda x: x.assignment_average[-2:].mean() - x.assignment_average[:2].mean()).reset_index()
        improvement.columns = ['student_id', 'improvement']
        df = pd.merge(df, improvement, on='student_id')
        df.pop('assignment_average')
        df.pop('student_id')

        choice_features = ['improvement', # response
                           'sign_in_count',
                           'compltd_assigmts',
                        #    'class_roster_id',
                           'class_size',
                           'stu_per_teacher',
                        #    'teacher_id',
                           'assignment_id',
                           'productivity',
                           'delta',
                           'grade_id']

        return df, choice_features


    def calculate_response_finalscores(self, df):
        # set final test scores as response
        final_scores = df.groupby('student_id').apply(lambda x: x.assignment_average[-2:].mean()).reset_index()
        final_scores.columns = ['student_id', 'final_scores']
        df = pd.merge(df, final_scores, on='student_id')
        df.pop('assignment_average')
        df.pop('student_id')

        df = df.dropna(how="any")

        choice_features = ['final_scores', # response
                           'first_scores',
                           'sign_in_count',
                           'compltd_assigmts',
                        #    'class_roster_id',
                           'class_size',
                           'stu_per_teacher',
                        #    'teacher_id',
                           'assignment_id',
                           'delta',
                        #    'grade_id',
                           'productivity'] # add back productivity

        return df, choice_features


    def compute_X_y(self, df):
        '''
        Assign response to "y"

        Input: final dataframe
        Output: X = feature matrix from final dataframe
                y = response, "improvement calculated on test scores"
        '''
        # df.pop('first_scores') # do I need to drop this to prevent data leakage?

        # choose response as either improvement or final_scores...which one is better?
        # y = df.pop('improvement').values
        y = df.pop('final_scores').values

        X = df.values
        return df, X, y



















# buffer
