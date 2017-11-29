import pandas as pd
import numpy as np
from load_data_v2 import LoadData, Preprocessor
# from model_pipeline import ModelMaker

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

import statsmodels.api as sm

from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint as print

import pdb

sns.set(color_codes=True)
sns.set(style="ticks")
sns.set(font_scale=1.5)

class Run:
    def __init__(self):

        self.n = 1


    def get_X_y(self, df, **kwargs):
        '''
        '''
        process = Preprocessor()

        print('current instance has the following parameter contraints:\n'.format(kwargs))
        # apply parameters to the dataframe
        df = process.apply_param_constraints(df, **kwargs)

        # calculate response - CHOOSE "RESPONSE HERE"
        # df_final, choice_features = process.calculate_response(df)
        df_final, choice_features = process.calculate_response_finalscores(df)

        df_final = df_final[choice_features]
        df_final = df_final.dropna(how="any")
        print("final dataframe feature matrix:\n rows={}, features={}".format(df_final.shape[0], df_final.shape[1]))

        # normalize data
        # df_norm = (df_final - df_final.mean()) / (df_final.max() - df_final.min())

        dfa, X, y = process.compute_X_y(df_final)
        return dfa, X, y


    def get_X_y_norm(self, df, **kwargs):
        '''
        '''
        process = Preprocessor()

        print('current instance has the following parameter contraints:\n'.format(kwargs))
        # apply parameters to the dataframe
        df = process.apply_param_constraints(df, **kwargs)

        # calculate response - CHOOSE "RESPONSE HERE"
        # df_final, choice_features = process.calculate_response(df)
        df_final, choice_features = process.calculate_response_finalscores(df)

        df_final = df_final[choice_features]
        df_final = df_final.dropna(how="any")
        print("final dataframe feature matrix:\n rows={}, features={}".format(df_final.shape[0], df_final.shape[1]))

        # normalize data
        df_norm = (df_final - df_final.mean()) / (df_final.max() - df_final.min())
        dfa, X, y = process.compute_X_y(df_norm)
        return dfa, X, y


    def random_forest_feature_importances(self, model, df):
        fi = model.feature_importances_.tolist()
        col = df.columns.tolist()
        feat = []
        for f, i in zip(col, fi):
            feat.append((f, i))
            # print('Importance: {:.2f}\t Feature: {}'.format(i, f))
        return feat


    def adaboost_feature_importances(self, model, df):
        adi = model.feature_importances_.tolist()
        col = df.columns.tolist()
        feat = []
        for f, i in zip(col, adi):
            feat.append((f, i))
            # print('Importance: {:.2f}\t Feature: {}'.format(i, f))
        return feat


    def partial_dependency_plots(self, names, X, y):
        # split 80/20 train-test
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=1)

        print("Training GBRT...")
        clf = GradientBoostingRegressor(n_estimators=100, max_depth=8,
                                        learning_rate=0.1, loss='huber',
                                        random_state=1)
        clf.fit(X_train, y_train)
        print(" done.")

        print('Convenience plot with ``partial_dependence_plots``')
        print('Features by number:')
        print([i for i in enumerate(names)])

        features = [5, 1, 0, 2, 3, (5, 1)]
        fig, axs = plot_partial_dependence(clf, X_train, features,
                                           feature_names=names,
                                           n_jobs=3, grid_resolution=50)
        fig.suptitle('Partial Dependence Plots of Student Improvement\n'
                     'for the CommonLit dataset')
        plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle

        print('Custom 3d plot via ``partial_dependence``')
        fig = plt.figure()

        target_feature = (7, 3)
        pdp, axes = partial_dependence(clf, target_feature,
                                       X=X_train, grid_resolution=50)
        XX, YY = np.meshgrid(axes[0], axes[1])
        Z = pdp[0].reshape(list(map(np.size, axes))).T
        ax = Axes3D(fig)
        surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
                               cmap=plt.cm.BuPu, edgecolor='k')
        ax.set_xlabel(names[target_feature[0]])
        ax.set_ylabel(names[target_feature[1]])
        ax.set_zlabel('Partial dependence')
        #  pretty init view
        ax.view_init(elev=22, azim=122)
        plt.colorbar(surf)
        plt.suptitle('Partial dependence of student final scores on class size\n'
                     'and productivity')
        plt.subplots_adjust(top=0.9)

        # plt.show()


        # filename = './plots/run' + str(self.n)
        # plt.savefig(filename)
        # self.n += 1


    def seaborn_scatter(self, df, **kwargs):
        '''
        '''
        # sns.regplot(x, y, data=None, x_estimator=None, x_bins=None, x_ci='ci',
        #             scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None,
        #             order=1, logistic=False, lowess=False, robust=False, logx=False,
        #             x_partial=None, y_partial=None, truncate=False, dropna=True,
        #             x_jitter=None, y_jitter=None, label=None, color=None,
        #             marker='o', scatter_kws=None, line_kws=None, ax=None)
        process = Preprocessor()

        print('current instance has the following parameter contraints:\n'.format(kwargs))
        # apply parameters to the dataframe
        df = process.apply_param_constraints(df, **kwargs)

        # calculate response - CHOOSE "RESPONSE HERE"
        # df_final, choice_features = process.calculate_response(df3)
        df, choice_features = process.calculate_response_finalscores(df)

        df_final = df[choice_features]
        df_final = df_final.dropna(how="any")
        print("final dataframe feature matrix:\n rows={}, features={}".format(df_final.shape[0], df_final.shape[1]))

        # normalize data
        # df_final = df_final[df_final['sign_in_count'] <= df_final.sign_in_count.std() * 3]
        # df_final = df_final[df_final['class_size'] <= df_final.class_size.std() * 3]
        # df_final = df_final[df_final['productivity'] <= df_final.productivity.std() * 3]
        # df_norm = (df_final - df_final.mean()) / (df_final.max() - df_final.min())
        # dfa, X, y = process.compute_X_y(df_norm)

        # limit to three standad devations
        df_final = df_final[df_final['sign_in_count'] <= df_final.sign_in_count.std() * 3]
        df_final = df_final[df_final['class_size'] <= df_final.class_size.std() * 3]
        df_final = df_final[df_final['productivity'] <= df_final.productivity.std() * 3]
        dfa, X, y = process.compute_X_y(df_final)


        # # for reference => kwargs = {'class_size': 5, 'min_spt': 10, 'max_spt': 200, 'n_assignments': 4}
        # x = dfa.class_size.values
        # title = "Class Size vs Test Score Performance\n (for a class size of at least 5)"
        # x, y = pd.Series(x, name="Class Size"), pd.Series(y, name="Final Scores")
        # return sns.regplot(x=x, y=y, x_bins=25).set_title(title)
        #
        # # for reference => kwargs = {'class_size': 5, 'min_spt': 1, 'max_spt': 2000, 'n_assignments': 10}
        # x = dfa.sign_in_count.values
        # title = "Sign-In Count vs Test Score Performance\n (for those who have completed at least 10 assignments)"
        # x, y = pd.Series(x, name="Sign-In Count"), pd.Series(y, name="Final Scores")
        # return sns.regplot(x=x, y=y, x_bins=25).set_title(title)
        #
        # productivity is the amount of time per completed assignment.
        # optimal productivity parameters for reference => kwargs = {'class_size': 1, 'min_spt': 1, 'max_spt': 2000, 'n_assignments': 30}
        # x = dfa.productivity.values
        # title = "Productivity vs Test Score Performance\n (for those who have completed at least 15 assignments)"
        # x, y = pd.Series(x, name="Productivity"), pd.Series(y, name="Final Scores")
        # return sns.regplot(x=x, y=y, x_bins=100).set_title(title)

        # # for reference => kwargs = {'class_size': 5, 'min_spt': 1, 'max_spt': 2000, 'n_assignments': 10}
        # x = dfa.first_scores.values
        # title = "First Scores vs Test Score Performance\n (for those who have completed at least 10 assignments)"
        # x, y = pd.Series(x, name="First Scores"), pd.Series(y, name="Final Scores")
        # return sns.regplot(x=x, y=y, x_bins=40).set_title(title)

        # # for reference => kwargs = {'class_size': 5, 'min_spt': 5, 'max_spt': 115, 'n_assignments': 4}
        # x = dfa.stu_per_teacher.values
        # title = "Students per Teacher vs Test Score Performance\n (for students per teacher within range of 5 to 115)"
        # x, y = pd.Series(x, name="Student per Teacher"), pd.Series(y, name="Final Scores")
        # return sns.regplot(x=x, y=y, x_bins=40).set_title(title)

        # # for reference => kwargs = {'class_size': 5, 'min_spt': 5, 'max_spt': 115, 'n_assignments': 4}
        # x = dfa.compltd_assigmts.values
        # title = "Completed Assignments vs Test Score Performance\n (for those who have completed at least 4 assignments)"
        # x, y = pd.Series(x, name="Completed Assignments"), pd.Series(y, name="Final Scores")
        # return sns.regplot(x=x, y=y, x_bins=80).set_title(title)

        return dfa

    def plot_hist_dist(self, y1, y2):
        '''
        '''
        plt.figure()
        plt.subplot(2, 2, 2)
        sns.distplot(y1, bins=20)

        plt.subplot(2, 2, 1)
        plt.hist(y2, bins='auto')

        plt.subplot(2, 1, 2)
        sns.distplot(y1, kde=False, fit=stats.beta, bins=25)


        plt.hist(y1, bins=50, alpha=0.5, label='y1 = min class size = 1')
        plt.hist(y2, bins=50, alpha=0.5, label='y2 = min class size = 5')
        plt.legend(loc='upper right')

        # plt.show()


    def joy_plot(self, y):
        sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        # Create the data
        rs = np.random.RandomState(1979)
        x = rs.randn(150)
        g = np.tile(list("ABC"), 50)
        df = pd.DataFrame(dict(x=x, g=g))
        m = df.g.map(ord)
        df["x"] += m

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        g = sns.FacetGrid(df, row="g", hue="g", aspect=15, size=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
        g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
        g.map(plt.axhline, y=0, lw=2, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, "x")

        # Set the subplots to overlap
        g.fig.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play will with overlap
        g.set_titles("")
        g.set(yticks=[])
        g.despine(bottom=True, left=True)


    def partial_loop(self, df, **kwargs):
        '''
        '''
        # df_final, X, y = self.get_X_y(df, **kwargs)  # raw X & y (i.e. not normalized)
        df_final, X, y = self.get_X_y_norm(df, **kwargs)  # normalized!!

        # create partial dependency plots with non normalized features
        print('calculating partial dependency plots with GradientBoostingRegressor')
        names = df_final.columns.tolist()
        self.partial_dependency_plots(names, X, y)
        print('partial dependency plots finished.\n initial features were as follows:\n {}'.format(df_final.columns))
        return df_final, y


    def model_loop(self, df, **kwargs):
        '''
        '''
        df_final, X, y = self.get_X_y_norm(df, **kwargs) # normalized
        # df_final = df_final[df_final['sign_in_count'] <= 30]
        # df_final = df_final[df_final['class_size'] <= 35]


        scaler = StandardScaler()
        X_new = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25)

        # # linear regression regressor
        # lr = LinearRegression(normalize=True)
        lr = ElasticNet(normalize=True)
        lr.fit(X_train, y_train)
        print(lr.score(X_test, y_test)) # R^2...0-1, higher is better
        lr_coefficients = lr.coef_.tolist()
        print([i for i in zip(df_final.columns.tolist(), lr_coefficients)])
        y_pred_lr = lr.predict(X_test)
        print("ElasticNet RMSE = {}".format(mean_squared_error(y_test, y_pred_lr)**.5))
        print('')

        # random forest model
        rf = RandomForestRegressor(n_estimators=100, max_depth=8)
        rf.fit(X_train, y_train)
        print("training r^2={}".format(rf.score(X_train, y_train)))
        print("test r^2={}".format(rf.score(X_test, y_test)))
        y_pred_rf = rf.predict(X_test)
        print("RandomForest RMSE = {}".format(mean_squared_error(y_test, y_pred_rf)**.5))
        rf_important_features = self.random_forest_feature_importances(rf, df_final)
        rfi = np.array(rf_important_features)
        print(rfi[np.argsort(rfi[:, 1])])
        print('')

        # MLP Neural Network
        mlp = MLPRegressor()
        mlp.fit(X_train, y_train)
        print(mlp.score(X_test, y_test))
        y_pred_mlp = mlp.predict(X_test)
        print("MLP RMSE = {}".format(mean_squared_error(y_test, y_pred_mlp)**.5))
        print('')

        # AdaBoostRegressor
        ada = AdaBoostRegressor()
        ada.fit(X_train, y_train)
        print(ada.score(X_test, y_test))
        y_pred_ada = ada.predict(X_test)
        print("ADABOOST RMSE = {}".format(mean_squared_error(y_test, y_pred_ada)**.5))
        ada_important_features = self.adaboost_feature_importances(ada, df_final)
        adi = np.array(ada_important_features)
        print(adi[np.argsort(rfi[:, 1])])

        return df_final


f2 =        ['student_id',
             'assignment_id',
             'status',
             'submit_time',
             'assignment_average',
             'class_roster_id',
             'sign_in_count',
             'grade_id',
             'text_id',
             'level_id',
             'lexile',
             'common_core_category',
             'slug',
             'compltd_assigmts',
             'teacher_id',
             'school_nces',
             'title1_schoolwide',
             'include_cfus']

f =         ['class_roster_id',
             'student_id',
             'assignment_id',
             'unique', # drop
             'teacher_id',
             'LEA_trim', # drop
             'group', # drop
             'group_temp', # drop
             'Title1_info2', # drop
             'Title1_yes_mean_1', # drop
             'school_nces',
             'title1_schoolwide', # drop
             'Title1', # drop
             'SRSA_info2',
             'SRSA_yes_mean_1',
             'status',
             'month', # drop
             'submit_time',
             'submit_time_raw', # drop
             'assignment_average',
             'sign_in_count',
             'text_id',
             'level_id',
             'lexile',
             'common_core_category',
             'slug',
             'student_grade_level',
             'grade_id',
             'grade_r',
             'reading_diff',
             'reading_diff_r',
             'grade_below',
             'grade_at',
             'grade_above',
             'state_lea_id',
             'filter_$', # drop
             'compltd_assigmts']



if __name__ == '__main__':
    # imports commonlit data from filename
    # filename = './data/aron_query 2017 0705 working.csv'
    filename = './raw_data/assignments_all_data.csv'

# LOADDATALOADDATALOADDATALOADDATALOADDATALOADDATALOADDATALOADDATALOADDATALOADDATALOADDATA
    data = LoadData()

    # load primary dataframe
    print('loading raw dataframe from file')
    df = data.load_data(filename=filename)

    # # load spark data frames
    # dfs_tdq, dfs_tdq_key, dfs_tdq_answers = data.load_data_spark()

    # # load large detailed files choose a random sample of size rowlimiter
    # df_cfu, df_tdq, df_tdq_key, df_cfu_key, df_tdq_answers = data.load_sample_data(rowlimiter=5000)

    # initial columns    pdb.set_trace()
    feats = data.feature_loader(df)
    print('finished loading raw dataframe')



# PROCESSPROCESSPROCESSPROCESSPROCESSPROCESSPROCESSPROCESSPROCESSPROCESSPROCESSPROCESSPROCESS
    process = Preprocessor()


    # clean column titles, remove the garbage columns and convert datetime
    print('cleaning columns, dropping columns, converting datetime columns')
    df = process.update_columns(df)
    df = process.make_immediate_drops(df)
    df = process.datetime_conversion(df)
    print('stage 1 finished')
    print('')

    # create features
    print('performing feature engineering')
    df2 = process.time_btwn_assignments(df)
    df2 = process.apply_length_col(df2, 'slug') # apply length to text column
    df2 = process.yes_no_col(df2, 'include_cfus') # convert true false to 1's and 0's
    df2 = process.calc_class_size(df2)
    df2 = process.calc_stu_per_teacher(df2)
    df2 = process.calc_mean_first_scores(df2)
    df2 = process.calc_productivity(df2) # number of assignment per submit time
    print("df with the following initial features: {}".format(df2.columns.tolist()))
    print('stage 2 finished')
    print('')


# MODELMODELMODELMODELMODELMODELMODELMODELMODELMODELMODELMODELMODELMODELMODELMODELMODELMODEL
# EDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZEDAVIZ
    run = Run()

    # # gridsearch attempt
    # kwargs = {'class_size': 5, 'min_spt': 10, 'max_spt': 200, 'n_assignments': 25}
    # param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
    #           "base_estimator__splitter" :   ["best", "random"],
    #           "n_estimators": [1, 2]}
    # df_final, X, y = run.get_X_y_norm(df2, **kwargs) # normalized
    # ada = AdaBoostRegressor()
    # clf = GridSearchCV(ada, param_grid=param_grid)
    # clf.fit(X, y)
    # sorted(clf.cv_results_.keys())


    #1 determine parameters
    print('#1 parameters beginning...')
    kwargs = {'class_size': 5, 'min_spt': 5, 'max_spt': 115, 'n_assignments': 30}
    run = Run()
    # dfone, yone = run.partial_loop(df2, **kwargs)
    dfonenorm = run.model_loop(df2, **kwargs)
    print('#1 parameters END')

    #2 determine parameters
    print('#2 parameters beginning...')
    kwargs = {'class_size': 5, 'min_spt': 5, 'max_spt': 5000, 'n_assignments': 6}
    run2 = Run()
    # dftwo, ytwo = run2.partial_loop(df2, **kwargs)
    dftwonorm = run2.model_loop(df2, **kwargs)
    print('#2 parameters END')
    #
    # #3 determine parameters
    # print('#3 parameters beginning...')
    # kwargs = {'class_size': 10, 'min_spt': 10, 'max_spt': 200, 'n_assignments': 8}
    # run3 = Run()
    # dfthree, ythree = run3.partial_loop(df2, **kwargs)
    # dfthreenorm = run3.model_loop(df2, **kwargs)
    # print('#3 parameters END')
    #
    # #4 determine parameters
    # print('#4 parameters beginning...')
    # kwargs = {'class_size': 15, 'min_spt': 20, 'max_spt': 115, 'n_assignments': 15}
    # run4 = Run()
    # dffour, yfour = run4.partial_loop(df2, **kwargs)
    # dffournorm = run4.model_loop(df2, **kwargs)
    # print('#4 parameters END')
    #
    # #5 determine parameters
    # print('#5 parameters beginning...')
    # kwargs = {'class_size': 5, 'min_spt': 5, 'max_spt': 115, 'n_assignments': 10}
    # run5 = Run()
    # dffive, yfive = run5.partial_loop(df2, **kwargs)
    # dffivenorm = run5.model_loop(df2, **kwargs)
    # print('#5 parameters END')



    kwargs = {'class_size': 1, 'min_spt': 1, 'max_spt': 200000, 'n_assignments': 4}
    # print(run.seaborn_scatter(df2, **kwargs))
    dftest = run.seaborn_scatter(df2, **kwargs)
    # ytwo = yfive
    # print(run.plot_hist_dist(yone, ytwo))

    plt.show()


# count    358255.000000
# mean         60.548342
# std          21.473541
# min           0.000000
# 25%          45.625000
# 50%          62.500000
# 75%          77.500000
# max         125.000000
# Name: first_scores, dtype: float64
# count    358255.000000
# mean         11.199020
# std           6.997614
# min           1.000000
# 25%           6.000000
# 50%          10.000000
# 75%          15.000000
# max          32.000000
# Name: sign_in_count, dtype: float64
# count    358255.000000
# mean          8.609130
# std           6.406728
# min           4.000000
# 25%           5.000000
# 50%           7.000000
# 75%          10.000000
# max          63.000000
# Name: compltd_assigmts, dtype: float64
# count    358255.000000
# mean         23.025926
# std           6.928519
# min           1.000000
# 25%          19.000000
# 50%          23.000000
# 75%          27.000000
# max          51.000000
# Name: class_size, dtype: float64
# count    358255.000000
# mean         92.375009
# std          69.046908
# min           1.000000
# 25%          60.000000
# 50%          83.000000
# 75%         115.000000
# max         700.000000
# Name: stu_per_teacher, dtype: float64
# count    358255.00000
# mean     101224.43303
# std       54325.83849
# min           6.00000
# 25%       58098.00000
# 50%      100318.00000
# 75%      145421.00000
# max      210566.00000
# Name: assignment_id, dtype: float64
# count    358255.000000
# mean          7.719359
# std          14.584000
# min           0.000000
# 25%           0.000000
# 50%           2.000000
# 75%           8.000000
# max         203.000000
# Name: delta, dtype: float64
# count    358255.000000
# mean          4.649099
# std          48.385220
# min           0.133946
# 25%           0.684868
# 50%           1.124688
# 75%           2.001408
# max        1369.444702
# Name: productivity, dtype: float64


# #cache dataframes
# df_train.to_pickle('cache/df_train.pkl')
# df_test.to_pickle('cache/df_test.pkl')





# multi layer perseptron
# interaciton terms
















# buffer
