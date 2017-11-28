# CommonLit - Student Success

![screenshot](https://s3.amazonaws.com/owler-image/logo/commonlit-org_owler_20170203_094053_original.png)

*Mission: to develop informed and engaged citizens by improving students' reading, writing and speaking skills*

## About

CommonLit delivers high-quality, free instructional materials to support literacy development for students in grades 5-12. Our resources are:
  * Flexible;
  * Research-Based;
  * Aligned to the Common Core State Standards;
  * Created by teachers, for teachers.

*"We believe in the transformative power of a great text, and a great question. That’s why we are committed to keeping CommonLit completely free, forever."*


## Predicting Student Success Through Advanced Machine Learning

### Goals & Objectives

* Define **"success"** in the context of CommonLit's mission

  *--How can we predict what improves a students ability to read and write using the CommonLit methodology and resources--*

* **Data cleaning and preprocessing** - Munge the data

* **Feature Engineering** - Search for predictors

* **Build a Model** that is **interpretable** and can effectively predict **student success**

* Apply **machine learning** methodologies and advanced **statistical analytics** to provide insights to the data.  

* Provide **clear recommendations** for **action items** management can implement

* Create **professional data visualizations** for management report

![screenshot](https://d1e2bohyu2u2w9.cloudfront.net/education/sites/default/files/website_review_-_commonlit.png)

  ### Defining the problem:  What is "student success"?

*Success in the context of CommonLit's mission is the improvement of a given students reading and writing ability measured by their assignment score improvement over time.*

  * ##### How do we measure it?
   * Student success is measured by **improvement over time** as calculated by the measured difference in first two assessment scores and the last two assessment scores.

  * ##### What factors influence it?
     * Student improvement in literacy development is a complicated and nuanced problem.  My focus was to find predictors in the data at hand.  At a high level, the analysis showed **class size**, **teachers per student**, **sign-in count**, and **student productivity** were strong influencers.

  * ##### How can we influence it?
     * As a multitude of studies have shown, the success of a student is strongly influenced by many factors outside the classroom.  However, the focus of this study was, controlling for other factors, how can CommonLit use the tools at hand to make the most impact on a students ability to read and write.  


  In addition to performing data munging and preprocessing of the initial dataset, I also engineered a number of features I thought would be relevant to predicting improvement in student assessment scores.  Specifically, I looked number of students per classroom id, the number of students per teacher id, the number of assignments completed by submit time as well as the time between the first assignment and last assignment.  

  After preliminary EDA showed that students per teacher, classroom size, and number of signs and productivity were influential in a students success, I attempted to look at school location and district information.  I wanted to control for poorly performed district/school in my analysis in an attempt to tease out more specific factors that determined improvement.  I also wanted to confirm that school location and school district played an important role in student improvement.


  <center></center>

![screenshot](https://www.ohio.edu/zanesville/images/WORDL2_full.jpg)

  ### Part 1 - Scoping the problem: Data munging and data preprocessing

  I started by performing a column-by-column analysis on the data looking for easy to incorporate features for an MVP.  I wanted to tackle this in steps to ensure I could could get working prototype as quickly as possible.

  * **Initial Features** First, I identified columns with null values and missing or incorrect data.  I then looked at histograms, scatter matrices, and other visualizations to get an idea of the data's distribution and possible relationships between the features.

    Initial Features for MVP:
```
['student_id', 'assignment_id', "
 "'status', 'assignment_average', 'class_roster_id', 'sign_in_count', "
 "'grade_id', 'text_id', 'level_id', 'lexile', 'common_core_category', "
 "'compltd_assigmts', 'teacher_id', 'school_nces', 'include_cfus', 'delta', "
 "'len_slug', 'class_size', 'stu_per_teacher', 'first_scores', 'submit_time', "
 "'sub_delta', 'productivity']
```

    Histogram Distribution of Assessment Scores



  * **Columns requiring minimal preprocessing - categorical one hot encoding, scaling:** These features were tagged to work on next as they required little additional work in order to be incorporated into the models.

      Minimal Preprocessing: **

  * **Columns requiring more involved feature engineering** I wanted to keep these features for further investigation at a later date.  While there may be opportunities to discover additional insights form these features, it would require

      Advanced Features: *?*

  * **Columns requiring extensive further investigation (out of scope)** While there may be opportunities to discover additional insights from these features, it would require more time than was available for this project.

      Out-of-Scope Features: *?*

  * **Drop Columns** Finally, we designated certain columns too sparse, irrelevant for consideration, or having too much missing or corrupt data.  We eliminated these from the analysis.

      Drop Features: *?*

  ### Part 2 - Building the Model

  I built an sklearn Pipeline to facilitate the rapid iteration and testing of various models and their respective parameters. I loaded a normalized and preprocessed DataFrame split into a train-test split to feed into the following models:
    * Linear Regression
    * Random Forest Regressor
    * AdaBoost Regressor
    * sklearn MLP Neural Network


  The performance of each model was assessed using the following metrics:
  * **R^2** - a measure of how close the data are to the fitted regression line.

      *"the coefficient of determination, denoted R2 or r2 and pronounced "R squared", is the proportion of the variation in the dependent variable that is predictable from the independent variable(s)"* [1]

  * **RMSE (Root Mean Squared Error)** - a measure of how good our predictive model is over the actual data.  

      *"...root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample and population values) predicted by a model or an estimator and the values actually observed. The RMSD represents the sample standard deviation of the differences between predicted values and observed values. These individual differences are called residuals when the calculations are performed over the data sample that was used for estimation, and are called prediction errors when computed out-of-sample. The RMSD serves to aggregate the magnitudes of the errors in predictions for various times into a single measure of predictive power. RMSD is a measure of accuracy, to compare forecasting errors of different models for a particular data and not between datasets, as it is scale-dependent"* [2]


[1] Wikipedia contributors. "Coefficient of determination" *Wikipedia, The Free Encyclopedia.* Wikipedia, The Free Encyclopedia, 16 November 2017. Web. 16 November 2017.

[2] Wikipedia contributors. "Root-mean-square deviation." *Wikipedia, The Free Encyclopedia.* Wikipedia, The Free Encyclopedia, 8 November 2017. Web. 8 November 2017.


  ```
  Final Model Features:

  ['first_scores', 'sign_in_count', 'compltd_assigmts', 'class_size',
         'stu_per_teacher', 'teacher_id', 'delta', 'grade_id', 'productivity']

  Results:

  'current instance has the following parameter constraints:\n'

  class size is at least 5,
  students per teacher is at least 5,
  students per teacher is less than 115,
  number of completed assignments is at least 30

  'final dataframe feature matrix:\n rows=5068, features=9'

  linear Regression with ElasticNet Results:
  R^2 = -1.197e-05
  'ElasticNet RMSE = 0.261'


  Random Forest Regressor Results:
  R^2 = 0.999
  'RandomForest RMSE = 0.003'


  AdaBoost Regressor Results:
  0.635
  'ADABOOST RMSE = 0.158'


  MLP Neural Network Results:
  R^2 = 0.5972
  'MLP RMSE = 0.166'

  ```

  We decided on our **“optimal” model** by following the Process Flow below:
* **Preprocessing** - calculate our **"response"** from *assessment scores*
* Create **MVP** with initial features requiring no preprocessing
* Add to model **features** needing only one hot encoding, scaling, other minimal preprocessing
* Work on more involved **feature engineering**
* **Drop** remaining
* Model **performance metrics** selected - **R^2, RMSE**

* TBD - Additional Example Code
   ```
   ```
* Validation and testing methodology - **sklearn model_selection train_test_split**
* **Parameter tuning** involved in generating the model - (in process)




![screenshot](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRg7WCyp2WxM188595NTk4yVSF4D6mqbK5JI-hV0Jn0P006P1MFlg)

  ### Part 4 - Communicating the Results

  In addition to the readme markdown file that communicated the work done and methodolgies implemented, I prepared a brief management report with the capstone findings.

<img src="/home/david/galvanize/2.0_dsi-immersive-boulder-g53ds/09_WEEK_Final and Projects/common_lit/plots/class_size_scatter_cls5_1stu2000_ca4.png" width="500">

<img src="/plots/class_size_scatter_cls5_1stu2000_ca4.png" width="500">

![screenshot](http://ecampusnews.eschoolmedia.com/files/2015/02/Upace.jpg)

  ### Part 5 - Final Thoughts

  #### What did we learn?

MongoDB was used to store the information. After the data is uploaded to the web app, the predictions, time the predictions were made, and the original data is all stored in a MongoDB database in a JSON like format.

  #### What would we have done differently?

The database is continuously updates as the web app runs. On the web app, when the ‘view predictions’ button is clicked, the web app takes the data from the MongoDB database and displays the case id, prediction, and time predicted.


## Recommendations & Next Steps


* While our SVC model performed okay, we would like to improve this
* Additional feature engineering
* Further Model Optimization - parameter tuning
* Optimize Threading
* Dashboard Improvement - Form Request with Time Period, Location and other Factors
* Visualization - D3
* Further steps we might have taken if we were to continue the project
* Additional feature engineering
* Further Model Optimization
* Optimize Threading
* Visualization - D3
