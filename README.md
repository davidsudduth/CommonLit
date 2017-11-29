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
   <!-- * Student success is measured by **improvement over time** as calculated by the measured difference in first two assessment scores and the last two assessment scores. -->
   * Student success is measured by **the average of the final two assessment scores** as the dependent variable that we are trying to predict

  * ##### What factors influence it?
     * Student improvement in literacy development is a complicated and nuanced problem.  My focus was to find predictors in the data at hand.  At a high level, the analysis showed **class size**, **teachers per student**, **sign-in count**, **completed assignments**, and **student productivity** were strong influencers.

  * ##### How can we influence it?
     * As a multitude of studies have shown, the success of a student is strongly influenced by many factors outside the classroom.  However, the focus of this study was, controlling for other factors, how can CommonLit use the tools at hand to make the most impact on a students ability to read and write.  


  In addition to performing **data munging and preprocessing** of the initial dataset, I also engineered a number of features I thought would be relevant to predicting improvement in student assessment scores.  Specifically, I looked number of students per classroom id, the number of students per teacher id, the number of assignments completed by submit time (labeled productivity) as well as the time between the first assignment and last assignment (delta).  

  After preliminary EDA, the initial model runs showed that students per teacher, classroom size, number of sign-ins and productivity were influential in a students success.
  <!-- I attempted to look at school location and district information.  I wanted to control for poorly performed district/school in my analysis in an attempt to tease out more specific factors that determined improvement.  I also wanted to confirm that school location and school district played an important role in student improvement. -->


  <center></center>

![screenshot](https://cdn-images-1.medium.com/max/1200/0*UqNyn9Os0Lgq0aO8.)
![screenshot](https://cdn-images-1.medium.com/max/1200/0*e2jLo5s6cum3gUrI.)

  ### Part 1 - Scoping the problem: Data munging and data preprocessing

  I started by performing a column-by-column analysis on the data looking for easy to incorporate features for an MVP.  I wanted to tackle this in steps to ensure I could could get working prototype as quickly as possible.

  * **Initial Features** First, I identified columns with null values and missing or incorrect data.  I then looked at histograms, scatter matrices, and other visualizations to get an idea of the data's distribution and possible relationships between the features.


  * **Columns requiring minimal preprocessing - categorical one hot encoding, scaling:** Added features that were tagged to work on next as they required little additional work in order to be incorporated into the models.


  * **Columns requiring more involved feature engineering** I wanted to keep these features for further investigation at a later date.  While there may be opportunities to discover additional insights form these features, it would require


  * **Columns requiring extensive further investigation (out of scope)** While there may be opportunities to discover additional insights from additional datasets and the features contained therein, it would require more time than was available for this project.


  * **Drop Columns** Finally, I designated certain columns too sparse, irrelevant for consideration, or having too much missing or corrupt data.  I eliminated these from the analysis.



  ### Part 2 - Building the Model

  I built an model pipeline to facilitate the rapid iteration and testing of various models and their respective parameters. I loaded a normalized and preprocessed DataFrame that was split into train / test subsets to feed into the following models:


  * Linear Regression
  * Random Forest Regressor
  * AdaBoost Regressor
  * sklearn MLP Neural Network


  The performance of each model was assessed using the following metrics:
  * **R^2** - a measure of how close the data are to the fitted

  * **RMSE (Root Mean Squared Error)** - a measure of how good our predictive model is over the actual data.  


  <!-- ```
  Final Model Features:

  ['first_scores',
   'sign_in_count',
   'compltd_assigmts',
   'class_size',
   'stu_per_teacher',
   'teacher_id',
   'delta', # time between assignments for each student
   'grade_id',
   'productivity']

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
  R^2 = 0.597
  'MLP RMSE = 0.166'

  ``` -->

  I decided on an **“optimal” model** by following the Process Flow below:
* **Preprocessing** - calculate our **"response"** from *assessment scores*
* Create **MVP** with initial features requiring no preprocessing
* Add to model **features** needing only one hot encoding, scaling, other minimal preprocessing
* Add more involved **feature engineering**
* **Drop** remaining
* Model **performance metrics** selected - **R^2, RMSE**
* Conduct **Partial Dependency Plots** Looking for relationships
* Validation and testing methodology - **sklearn model_selection train_test_split**
* **Parameter tuning** involved in generating the model


#### Partial Dependency of a Gradient Boosting Regressor
<!-- <center>
<img src="/home/david/galvanize/2.0_dsi-immersive-boulder-g53ds/09_WEEK_Final and Projects/common_lit/plots/partial_dependence_plot.png" width="700">
</center> -->

<img src="/plots/partial_dependence_plot.png" width="700">

Through the noise in the Partial dependency plot above, we can see the completed assignments and class size appear to be influential in the predicting student improvement.  In this case I used a response that was calculated as the difference between the average of the first two scores and an average of the final two scores.


![screenshot](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRg7WCyp2WxM188595NTk4yVSF4D6mqbK5JI-hV0Jn0P006P1MFlg)

  ### Part 4 - Communicating the Results

  In addition to the readme markdown file that communicated the work done and methodologies implemented, I prepared a brief management report with the capstone findings.  You can find below some of the relationships found in the data.


* **Class Size and Test Scores**
  * "Class Size" is calculated as the number of unique students per unique classroom id
  * As you can see below, as the final test scores increase as the class size increased from 5 to ~35

<!-- <center>
<img src="/home/david/galvanize/2.0_dsi-immersive-boulder-g53ds/09_WEEK_Final and Projects/common_lit/plots/class_size_scatter_cls5_1stu2000_ca4.png" width="500">
</center> -->

<!-- <img src="/plots/class_size_scatter_cls5_1stu2000_ca4.png" width="700"> -->


* What are the **takeaways** from this information?
    * We often assume that smaller class size will result in better student performance but perhaps...
      * There is a synergistic effect between more students per class
      * The teachers who create bigger classes are more committed to implementing the CommonLit curriculum




<br><br>
* **Sign-In Count and Test Scores**
  * "Sign-In" is calculated by the total number of times the a unique student logged on to the website
  * As you can see below, as the sign-in count increased so did the final test scores

<!-- <center>
<img src="/home/david/galvanize/2.0_dsi-immersive-boulder-g53ds/09_WEEK_Final and Projects/common_lit/plots/sign-in_count_scatter_cls5_1stu2000_ca10.png" width="500">
</center> -->

<img src="/plots/sign-in_count_scatter_cls5_1stu2000_ca10.png" width="700">

  * What are the **takeaways** from this information?
      * The more often a student logs in, the more likely they are to benefit from the curriculum

<br><br>
* **Productivity and Test Scores**
  * Student Productivity is the measure of completed assignments per week
  * One would think that the more 'intensely' a student works, the better their final score

<!-- <center>
<img src="/home/david/galvanize/2.0_dsi-immersive-boulder-g53ds/09_WEEK_Final and Projects/common_lit/plots/productivity_scatter.v2_15 assgmts_ and_within_3std.png" width="500">
</center> -->

<img src="/plots/productivity_scatter.v2_15 assgmts_ and_within_3std.png" width="700">

  * What are the **takeaways** from this information?
      * Is there something hidden that requires further information or calculation?
        * Does not take into consideration academic calender breaks, time off, etc.
      * Are multiple students using the same login?
      * Are teachers asking students to work in pairs or groups on an assignment?


<br><br>
* **Completed Assignments and Test Scores**
  * Completed Assignments is the count of completed assignments per student
  * It is logical to see a relationship between more assignments and higher final scores

<!-- <center>
<img src="/home/david/galvanize/2.0_dsi-immersive-boulder-g53ds/09_WEEK_Final and Projects/common_lit/plots/completed_assignments.png" width="500">
</center> -->

<img src="/plots/completed_assignments.png" width="700">

* What are the **takeaways** from this information?
    * High performing students tend to continue to do well
    * There seems to be a strong initial correlation but a subsequent downward trend


<br><br>
* **First Score and Test Scores**
  * The "First Score" is calculated by taking the average of the first two test scores of a student
  * While not an actionable item by teachers, this is a clear indicator of a student final test scores

<!-- <center>
<img src="/home/david/galvanize/2.0_dsi-immersive-boulder-g53ds/09_WEEK_Final and Projects/common_lit/plots/first_score_scatter_cls5_1stu2000_ca10.png" width="500">
</center> -->

<img src="/plots/first_score_scatter_cls5_1stu2000_ca10.png" width="700">

  * What are the takeaways from this information?
      * High performing students tend to continue to do well and visa versa
      * It is also observable that if you have a lower first score, you will show more improvement.  However, if you have a higher initial score, you do not tend to improve
<br><br>







![screenshot](http://ecampusnews.eschoolmedia.com/files/2015/02/Upace.jpg)

  ### Part 5 - Final Thoughts

  #### What did we learn?

All in all, it is a difficult task to predict a students outcome based on limited information.  Some data proved to be informative and expected, more informative and unexpected, while a lot of the data was not informative at all.


## Recommendations & Next Steps

* Additional feature engineering
* Further Model Optimization - parameter tuning
* Visualization - D3

acknowledgments
common lit data staff
sklearn, wordnet, make more technical, fewer results, reference code via link.

<!-- * While our SVC model performed okay, we would like to improve this
* Additional feature engineering
* Further Model Optimization - parameter tuning
* Optimize Threading
* Dashboard Improvement - Form Request with Time Period, Location and other Factors
* Visualization - D3
* Further steps we might have taken if we were to continue the project
* Additional feature engineering
* Further Model Optimization
* Optimize Threading
* Visualization - D3 -->
