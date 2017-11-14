# CommonLit

![screenshot](https://s3.amazonaws.com/owler-image/logo/commonlit-org_owler_20170203_094053_original.png)

*Mission: to develop informed and engaged citizens by improving students' reading, writing and speaking skills*

## Predicting Student Success Through Advanced Machine Learning

### Goals & Objectives

* Define **"success"** in the context of CommonLit's mission

  *--How can we predict what improves a students ability to read and write using the CommonLit methodology and resources--*

* **Data preprocessing**

* **Feature Engineering**

* **Build a Model** that is **efficient** and **effective** at predicting **student success**
  * Apply machine learning methodologies and advanced statistical analytics to provide insights to the data.  


* Provide **clear recommendations** in the report for **action items** management can implement

![screenshot](https://d1e2bohyu2u2w9.cloudfront.net/education/sites/default/files/website_review_-_commonlit.png)

  ### Defining the problem:  What is "student success"?

  * How do we measure it?

  * What factors influence it?

  * How can we influence it?

  Success in the context of CommonLit's mission is the improvement of a given students reading and writing ability measured by their assignment score improvement over time.

  We want to

  any instance of user activity using company infrastructure that results in a liability to the company.  For example, this could include fake tickets that are sold on the company website, fake events that the company hosts, unauthorized user activity of other people's accounts, and falsified credit card and checking account information used to successfully purchase event tickets.

  Our analysis focused on all the information *"acct_type"* labeled as fraud in any way.  This included *"fraudster_event"*, *"fraudster"*, and *"fraudster_att"*.  With these assumptions and classifications, we can then focus on building the most effective tool.

  <center>*Initial Dataset => Fraud: 1293 Not Fraud: 13044*</center>

![screenshot](https://datonauts.com/wp-content/uploads/2016/10/Statistics-Word-Cloud.png)

  ### Part 1 - Scoping the problem: Preprocessing and building an appropriate Feature Matrix

  We started by performing a column-by-column analysis on the data looking for easy to incorporate features for our MVP.  We wanted to tackle this in steps to ensure we could get an MVP as quickly as possible.

* **Initial Features** We identified columns with null values and missing or incorrect data.  We looked at histograms and a scatter matrix to get an idea of the data's distribution keeping an eye out for possible relationships.

  Initial Features for MVP: *body_length, fb_published, has_analytics, has_logo, name_length, user_age*

* **Columns requiring minimal preprocessing - one hot encoding, scaling:** These features were tagged to work on next as they required little additional work in order to be incorporated into the models.

  Minimal Preprocessing: *channels, currency, delivery_method, venue_country/country, listed, user_type, gts, payout_type*

* **Columns requiring NLP processing** We wanted to keep these features for further investigation at a later date.  While there may be opportunities to discover additional insights form these features, it would require

  NLP Features: *name, description, org_desc, org_name, payee_name, venue_address*

* **Columns requiring extensive further investigation (out of scope)** We wanted to keep these features for further investigation at a later date.  While there may be opportunities to discover additional insights form these features, it would require

  Out-of-Scope Features: *approx_payout_date, fb_published, has_analytics, ticket_types, venue_address, venue_country*

* **Drop Columns** Finally, we designated certain columns too sparse, irrelevant for consideration, or having too much missing or corrupt data.  We eliminated these from the analysis for the time being.

  Drop Features: *event_created, event_end, event_published, user_created, venue_latitude, venue_longitude, venue_name, venue_state, has_header, previous_payouts, org_twitter*


  ### Part 2 - Building the Model

  We built an SkLearn Pipeline to facilitate the rapid iteration and testing of various models and their respective parameters. We read in our processed DataFrame and tested the following models, looking for the highest **f1_score**:
  * SVM
  * Logistic Regression
  * RandomForestClassifier
  * kNN
  * AdaBoostClassifier

  ```
  SVM Cross Validation Results:

  In [4]: cross_val_score(tmt.pipeline, tmt.X, tmt.y, scoring="f1_macro")
  Out[4]: array([ 0.67204029,  0.68970558,  0.68097463])

  Model Features:

  ['body_length', 'channels', 'currency', 'fb_published', 'has_analytics', 'has_logo', 'name_length', 'user_age', 'payout_type', 'acct_type']
  ```

    We decided on our “optimal” model by following the Process Flow below:
   * Preprocessing - calculate our **"y"** from *acct_type*
     * Create MVP with initial features requiring no preprocessing
     * Add to model features needing only one hot encoding, scaling
     * Work on more involved feature engineering like **NLP**
     * Drop remaining
   * Accuracy metrics selected - **f1_score**

* Initial Confusion Matrix
   ```
           -----------
           | TP | FP |
           -----------
           | FN | TN |
           -----------

           -----------
           | 1093  | 196 |
           -----------
           | 914 | 12134 |
           -----------
   ```
   * Validation and testing methodology - **sklearn cross val score - f1_score: macro**
   * Parameter tuning involved in generating the model - (in process)
   * Further steps we might have taken if we were to continue the project
     * Additional feature engineering
     * Further Model Optimization
     * Optimize Threading
     * Dashboard Improvement - Form Request with Time Period, Location and other Factors
     * Visualization - D3



![screenshot](https://dev.codetrick.net/wp-content/uploads/2017/05/oEGZ93DzRw2JMGZStUKE_flask-crud-part-one1.jpg)

  ### Part 3 - Building the App

  Wrote a 'GET' / 'POST' function which made the necessary server requests. This function is called every five seconds.

  Running the app:
  * From the command line in folder ```web_app/```

      ```
    python final_model.py  # creates a pickle file of model
    python app.py  # sets up flask site (keep running: tmux)
    ```

  * From the command line in folder FROM A NEW TERMINAL in folder ```web_app/```

      ```python request_data.py  # gets data from source and posts to flask site (keep running: tmux)
      ```

  Once the data starts streaming in, click on the button ‘View Data’ to navigate to the show the data presented from the associated database. Additionally, ‘View Readme’ takes you to this document on github.


![screenshot](https://realpython.com/images/blog_images/python-and-mongo//python-and-mongo-logos.png)

  ### Part 4 - MongoDB - Storing the Predicted Fraud Probabilities

  MongoDB was used to store the information. After the data is uploaded to the web app, the predictions, time the predictions were made, and the original data is all stored in a MongoDB database in a JSON like format. The database is continuously updates as the web app runs. On the web app, when the ‘view predictions’ button is clicked, the web app takes the data from the MongoDB database and displays the case id, prediction, and time predicted.


## Results and Recommendations

* We break the results into categorical risks because is imperfect
* While our SVC model performed okay, we would like to improve this
* Be Alert for Fraud!



## Next Steps

* Additional feature engineering
* Further Model Optimization - parameter tuning
* Optimize Threading
* Dashboard Improvement - Form Request with Time Period, Location and other Factors
* Visualization - D3
