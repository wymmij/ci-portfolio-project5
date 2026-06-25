# Heritage Housing Issues

Heritage Housing Issues is a Predictive Analytics project that uses data from the Ames, Iowa housing market to help a fictional client estimate the likely sale price of inherited properties.

The project includes data analysis, data visualisation, and a supervised machine learning regression model deployed in a Streamlit dashboard.

[Live App](https://YOUR_APP_NAME.herokuapp.com/)

[GitHub Repository](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME)

---

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). A fictional user story has been created to show how predictive analytics can be applied in a real business context.

The dataset contains almost 1.5 thousand housing records from Ames, Iowa. Each row represents one house sale. The columns describe house attributes such as floor area, basement features, garage features, kitchen quality, lot size, porch area, year built, and sale price.

The target variable for the machine learning task is `SalePrice`.

| Variable      | Meaning                                     | Units / Values                     |
| ------------- | ------------------------------------------- | ---------------------------------- |
| 1stFlrSF      | First floor square feet                     | 334 - 4692                         |
| 2ndFlrSF      | Second floor square feet                    | 0 - 2065                           |
| BedroomAbvGr  | Bedrooms above grade                        | 0 - 8                              |
| BsmtExposure  | Basement exposure                           | Gd, Av, Mn, No, None               |
| BsmtFinType1  | Rating of basement finished area            | GLQ, ALQ, BLQ, Rec, LwQ, Unf, None |
| BsmtFinSF1    | Type 1 finished basement square feet        | 0 - 5644                           |
| BsmtUnfSF     | Unfinished basement square feet             | 0 - 2336                           |
| TotalBsmtSF   | Total basement square feet                  | 0 - 6110                           |
| GarageArea    | Garage size in square feet                  | 0 - 1418                           |
| GarageFinish  | Interior finish of garage                   | Fin, RFn, Unf, None                |
| GarageYrBlt   | Year garage was built                       | 1900 - 2010                        |
| GrLivArea     | Above-grade living area in square feet      | 334 - 5642                         |
| KitchenQual   | Kitchen quality                             | Ex, Gd, TA, Fa, Po                 |
| LotArea       | Lot size in square feet                     | 1300 - 215245                      |
| LotFrontage   | Linear feet of street connected to property | 21 - 313                           |
| MasVnrArea    | Masonry veneer area in square feet          | 0 - 1600                           |
| EnclosedPorch | Enclosed porch area in square feet          | 0 - 286                            |
| OpenPorchSF   | Open porch area in square feet              | 0 - 547                            |
| OverallCond   | Overall condition rating                    | 1 - 10                             |
| OverallQual   | Overall material and finish rating          | 1 - 10                             |
| WoodDeckSF    | Wood deck area in square feet               | 0 - 736                            |
| YearBuilt     | Original construction date                  | 1872 - 2010                        |
| YearRemodAdd  | Remodel date                                | 1950 - 2010                        |
| SalePrice     | Sale price                                  | 34900 - 755000                     |

---

## Business Requirements

Lydia Doe has inherited four houses in Ames, Iowa. She has a good understanding of property prices in her own country, but she is unfamiliar with the local housing market in Ames. She wants to avoid inaccurate price estimates before deciding how to sell the inherited houses.

The project has two business requirements:

1. The client is interested in discovering how house attributes correlate with sale price. Therefore, the client expects data visualisations of the most relevant variables against sale price.

2. The client is interested in predicting the sale price of her four inherited houses and any other house in Ames, Iowa.

---

## Hypothesis and How to Validate

### Hypothesis 1

Houses with higher overall quality and larger above-grade living area tend to have higher sale prices.

### Validation Method

This hypothesis will be validated by:

* calculating correlation values between `SalePrice` and numerical house attributes;
* visualising the relationship between `SalePrice`, `OverallQual`, and `GrLivArea`;
* reviewing whether the strongest correlations and plots support the hypothesis.

### Conclusion

This section will be completed after the data analysis notebooks have been run.

---

## The Rationale to Map the Business Requirements to the Data Visualisations and ML Tasks

### Business Requirement 1: Sale Price Study

The first business requirement will be answered through data analysis and data visualisation.

The project will:

* inspect the housing dataset;
* identify the variables most strongly related to `SalePrice`;
* use correlation analysis and visual plots to study relationships between house attributes and sale price;
* display the most useful findings in the Streamlit dashboard.

### Business Requirement 2: Sale Price Prediction

The second business requirement will be answered through a supervised machine learning regression task.

The project will:

* train a regression model using historical Ames housing data;
* evaluate the model using R² score and actual-vs-predicted plots;
* save the trained machine learning pipeline;
* use the saved pipeline in the Streamlit dashboard to predict sale prices for Lydia’s inherited houses and user-provided house profiles.

---

## ML Business Case

### Predict Sale Price

The project includes one machine learning task: predicting house sale price.

* **ML task:** Regression
* **Learning method:** Supervised learning
* **Target variable:** `SalePrice`
* **Model output:** Predicted sale price in US dollars
* **Training data:** Historical house sale records from Ames, Iowa
* **Business use:** The prediction helps Lydia estimate the likely sale value of inherited houses and other Ames properties.

### Success Metrics

The model will be evaluated using R² score on the train and test sets.

The model will be considered successful if:

* the test set R² score is at least 0.75;
* the train and test scores do not suggest severe overfitting;
* the dashboard can use the saved pipeline to return sale price predictions.

### Failure Conditions

The model will be considered unsuccessful if:

* the test set R² score is below 0.75;
* the model performs well on the train set but poorly on the test set;
* the saved pipeline cannot be used reliably in the deployed dashboard.

### Heuristics

Without a machine learning model, the client has no data-driven method to estimate sale price for inherited houses in Ames, Iowa.

---

## Dashboard Design

The Streamlit dashboard will contain five pages.

### Page 1: Project Summary

This page will include:

* a summary of the client situation;
* a description of the dataset;
* the two business requirements;
* an explanation of the dashboard pages.

### Page 2: Sale Price Study

This page will answer Business Requirement 1.

It will include:

* a summary of the sale price study;
* visualisations showing the strongest relationships between house attributes and `SalePrice`;
* written interpretations of each plot or group of plots;
* conclusions about which house attributes appear most relevant to sale price.

### Page 3: Project Hypothesis and Validation

This page will include:

* the project hypothesis;
* the validation method;
* the final conclusion after the data analysis has been completed.

### Page 4: Predict Sale Price

This page will answer Business Requirement 2.

It will include:

* prediction results for Lydia’s four inherited houses;
* total predicted sale value for the inherited properties;
* input widgets allowing the user to provide house attributes;
* a button to run the predictive analysis;
* the predicted sale price for the user-provided house profile.

### Page 5: ML Performance

This page will include:

* a summary of the machine learning pipeline;
* the model performance results;
* the train and test R² scores;
* actual-vs-predicted plots;
* feature importance information, where available;
* a statement explaining whether the model met the business requirement.

---

## Project Structure

```text
.
├── app.py
├── app_pages
├── inputs
├── jupyter_notebooks
├── outputs
├── src
├── README.md
├── requirements.txt
├── Procfile
├── runtime.txt
└── setup.sh
```

---

## Testing

This section will be completed after the project has been implemented.

Testing will include:

* checking that all notebooks run from top to bottom;
* checking that the Streamlit app runs locally;
* checking that all dashboard pages load correctly;
* checking that prediction widgets return a sale price prediction;
* checking that the deployed Heroku app works as expected.

---

## Unfixed Bugs

This section will be completed before submission.

At present, there are no known unfixed bugs.

---

## Deployment

### Heroku

The app will be deployed to Heroku.

The live app link is:

https://YOUR_APP_NAME.herokuapp.com/

The project includes the following deployment files:

* `requirements.txt`
* `Procfile`
* `runtime.txt`
* `setup.sh`
* `.python-version`

Deployment steps:

1. Log in to Heroku.
2. Create a new Heroku app.
3. In the Deploy tab, select GitHub as the deployment method.
4. Connect the GitHub repository.
5. Select the deployment branch.
6. Click **Deploy Branch**.
7. After deployment, click **Open App** to confirm the app is running.

---

## Main Data Analysis and Machine Learning Libraries

The main libraries used in this project will include:

* **pandas**: for loading, cleaning, and analysing tabular data.
* **NumPy**: for numerical operations.
* **Matplotlib**: for static data visualisations.
* **Seaborn**: for statistical data visualisations.
* **Plotly**: for interactive visualisations.
* **scikit-learn**: for machine learning pipeline creation, model training, and model evaluation.
* **Feature-engine**: for feature engineering steps.
* **Streamlit**: for building and deploying the dashboard.
* **joblib**: for saving and loading the fitted machine learning pipeline.

This section will be updated as the project develops.

---

## Credits

### Content

* The dataset was provided by Code Institute through Kaggle: [Housing Prices Data](https://www.kaggle.com/codeinstitute/housing-prices-data).
* The project structure and assessment expectations are based on Code Institute’s Predictive Analytics project materials.
* Code Institute walkthrough projects were used as guidance for project structure, dashboard organisation, and README organisation.

### Code

* Code Institute walkthrough project materials were used as structural guidance.
* Additional code credits will be added here where external documentation, examples, or tutorials are used.

### Acknowledgements

* Code Institute for providing the project brief, assessment materials, and dataset.
