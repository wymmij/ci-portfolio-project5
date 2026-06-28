# Heritage Housing Issues

Heritage Housing Issues is a predictive analytics dashboard built with Streamlit.

The project uses the Ames, Iowa housing dataset to investigate which house attributes are most associated with sale price and to predict sale prices for four inherited houses.

The dashboard was built for a fictional client, Lydia, who has inherited four houses in Ames, Iowa. Lydia does not know the Ames housing market and wants data-driven support before deciding how to value the inherited properties.

[View the live app](https://wymmij-heritage-housing-issues-4e260e88ecde.herokuapp.com/)

[View the GitHub repository](https://github.com/wymmij/ci-portfolio-project5)

---

## Dataset Content

The dataset is sourced from the Code Institute Heritage Housing Issues project dataset on Kaggle.

The dataset contains housing records from Ames, Iowa. Each record describes property attributes such as living area, basement area, garage size, kitchen quality, lot size, construction year, remodelling year, porch area, and sale price.

The project uses two source datasets:

* `house_prices_records.csv`: historical house sale records, including `SalePrice`;
* `inherited_houses.csv`: four inherited houses requiring sale price predictions.

The target variable for the machine learning task is `SalePrice`.

A full feature table is included below to explain the dataset variables and category codes.

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

The client has two business requirements.

### Business Requirement 1

The client wants to understand which house attributes are most strongly associated with sale price in Ames, Iowa.

This is answered through exploratory data analysis, correlation analysis, and dashboard visualisations showing the strongest relationships between house attributes and `SalePrice`.

### Business Requirement 2

The client wants to predict the sale price of four inherited houses and any other Ames house with similar attribute data.

This is answered through a supervised regression model embedded in the Streamlit dashboard.

---

## Project Hypothesis and Validation

### Hypothesis

Houses with higher overall quality `OverallQual`, larger above-ground living area `GrLivArea`, and newer construction dates `YearBuilt` tend to have higher sale prices.

### Validation Method

The hypothesis was validated through the Sale Price Study notebook and dashboard page.

The validation process used:

* Pearson correlation;
* Spearman correlation;
* box plots;
* scatter plots;
* feature importance from the final regression model.

### Hypothesis Result

The hypothesis was supported.

The strongest positive relationships with `SalePrice` included:

* overall quality `OverallQual`;
* above-ground living area `GrLivArea`;
* construction year `YearBuilt`;
* garage area `GarageArea`;
* total basement area `TotalBsmtSF`.

The analysis does not prove causation. It shows that these attributes are meaningfully associated with sale price in this dataset.

---

## Rationale: Mapping Business Requirements to Data Visualisations and ML Tasks

### Business Requirement 1: Sale Price Study

To answer the first business requirement, the project uses data analysis and visualisation.

The Sale Price Study investigates correlations between house attributes and `SalePrice`. It identifies the features most strongly associated with sale price and visualises the most important relationships.

Dashboard evidence:

* top correlated features table;
* bar chart of strongest Spearman correlations;
* box plot of `SalePrice` by `OverallQual`;
* scatter plot of `SalePrice` by `GrLivArea`;
* scatter plot of `SalePrice` by `YearBuilt`.

### Business Requirement 2: Sale Price Prediction

To answer the second business requirement, the project uses a supervised machine learning regression model.

The model predicts `SalePrice` from house attributes. The final fitted model is used in the dashboard to:

* predict sale prices for Lydia's four inherited houses;
* calculate the total predicted value of the inherited houses;
* allow the user to enter custom house attributes and generate a sale price prediction.

---

## ML Business Case

### Objective

The objective is to build a regression model that predicts house sale price from property attributes.

### ML Task

This is a supervised regression task.

### Target Variable

The target variable is:

* `SalePrice`

### Model Inputs

The model uses numerical and ordinal-encoded property attributes, including:

* quality and condition ratings;
* living area;
* basement area;
* garage area;
* lot area;
* porch and deck area;
* construction and remodelling age features.

### Model Output

The model outputs a predicted house sale price in US dollars.

### Success Metric

The main success metric is R² on the test set.

The model is considered successful if it achieves:

* test set R² >= 0.75

Additional metrics used for evaluation are:

* Mean Absolute Error;
* Root Mean Squared Error.

### Model Selection

Several regression models were compared:

* Linear Regression;
* Decision Tree Regressor;
* Random Forest Regressor;
* Extra Trees Regressor;
* Gradient Boosting Regressor.

Gradient Boosting was selected as the final model because it gave the strongest test-set performance and a better train/test balance than the more heavily overfitted tree ensemble alternatives.

The final Gradient Boosting model achieved:

* Train R²: 0.965
* Test R²: 0.880

The final model therefore met the success threshold of test R² >= 0.75.

---

## Dashboard Design

The Streamlit dashboard contains five pages.

### Page 1: Project Summary

This page introduces the project, the dataset, the client problem, and the two business requirements.

It helps the user understand the purpose of the dashboard before exploring the analysis or prediction tools.

### Page 2: Sale Price Study

This page answers Business Requirement 1.

It shows:

* the top features correlated with `SalePrice`;
* feature meanings for abbreviation-heavy variables;
* a bar chart of the strongest correlations;
* visualisations of key relationships between sale price and important house attributes.

### Page 3: Project Hypothesis

This page states the project hypothesis and explains how it was validated.

It shows the correlation evidence for the hypothesis features and states whether the analysis supports the hypothesis.

### Page 4: Predict Sale Price

This page answers Business Requirement 2.

It shows:

* predicted sale prices for Lydia's four inherited houses;
* the total predicted value of the inherited houses;
* a custom prediction form where users can enter house attributes and generate a sale price prediction.

The prediction form includes explanations for dataset abbreviations and category codes.

### Page 5: ML Performance

This page shows evidence for the final model's performance.

It includes:

* model comparison results;
* final model performance metrics;
* the final test-set R² score;
* actual-vs-predicted plots for train and test sets;
* feature importance table and chart.

---

## Testing

Testing was carried out throughout the project while developing the notebooks, machine learning pipeline, and Streamlit dashboard.

### Notebook Testing

Each notebook was run from top to bottom after development.

| Notebook                           | Test                                                                                                                                               | Result |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `01 DataCollection.ipynb`          | Downloaded the Kaggle dataset and saved project copies of the raw CSV files                                                                        | Pass   |
| `02 SalePrice Study.ipynb`         | Loaded collected data, created correlation analysis, and saved correlation summary files                                                           | Pass   |
| `03 DataCleaning.ipynb`            | Loaded collected datasets, investigated missing values, cleaned data, and saved cleaned train/test/inherited datasets                              | Pass   |
| `04 FeatureEngineering.ipynb`      | Loaded cleaned datasets, encoded categorical variables, created engineered features, and saved featured datasets and feature-engineering artifacts | Pass   |
| `05 Modeling and Evaluation.ipynb` | Trained and evaluated regression models, saved final model and performance artifacts, and predicted inherited house prices                         | Pass   |

### Data Cleaning Testing

The data cleaning process checked:

* duplicated rows;
* missing values before and after cleaning;
* whether missing values represented structural absence or unknown values;
* remaining missing values after imputation;
* train/test/inherited dataset shape consistency.

Final checks confirmed that no missing values remained in the cleaned datasets.

### Feature Engineering Testing

Feature engineering checks confirmed that:

* categorical variables were encoded successfully;
* no categorical values failed mapping;
* no missing values remained after feature engineering;
* all final model input columns were numerical;
* train, test, and inherited datasets used the same selected feature set.

### Model Testing

Several regression models were compared before selecting the final model.

The final model was evaluated using:

* R²;
* Mean Absolute Error;
* Root Mean Squared Error;
* actual-vs-predicted plots for train and test sets;
* feature importance.

The final Gradient Boosting model achieved:

| Dataset | R²    |
| ------- | ----- |
| Train   | 0.965 |
| Test    | 0.880 |

The model met the project success threshold of test R² >= 0.75.

The saved model was reloaded and tested in the modelling notebook to confirm that it could be used after serialisation.

### Streamlit App Testing

The Streamlit app was tested locally using:

```bash
streamlit run app.py
```

The following dashboard pages were checked manually:

| Page               | Test                                                                                          | Result |
| ------------------ | --------------------------------------------------------------------------------------------- | ------ |
| Project Summary    | Page loads and shows business requirements and page descriptions                              | Pass   |
| Sale Price Study   | Page loads correlation table, feature meanings, and charts                                    | Pass   |
| Project Hypothesis | Page loads hypothesis, validation table, and conclusion                                       | Pass   |
| Predict Sale Price | Page loads inherited house predictions and custom prediction form                             | Pass   |
| Predict Sale Price | Custom form submits successfully and returns a sale price prediction                          | Pass   |
| ML Performance     | Page loads model comparison, final metrics, actual-vs-predicted plots, and feature importance | Pass   |

### Code Validation

Python files were checked using:

```bash
python -m compileall app.py app_pages src
```

The command completed successfully.

### Browser and Responsiveness Testing

The dashboard was tested locally in a desktop browser.

The prediction form was adjusted so that it has a maximum width on wide screens while still adapting to narrower screens. This improved readability without making the form unnecessarily cramped.

---

## Bugs

### Fixed Bugs

#### Pandas reading `"None"` as missing data

The cleaned datasets use the string `"None"` to represent structural absence, such as no basement or no garage.

When the cleaned CSV files were loaded again, pandas interpreted `"None"` as a missing value. This caused issues during categorical encoding.

The issue was fixed by loading the cleaned CSV files with:

```python
pd.read_csv(file_path, keep_default_na=False)
```

#### Incorrect assumption about missing categorical values

Initial cleaning logic assumed that missing values in garage and basement category columns always meant no garage or no basement.

Further investigation showed this was not always true. Some missing `GarageFinish` values occurred where `GarageArea` was greater than 0, meaning the house appeared to have a garage but the finish value was unknown.

The cleaning strategy was updated to separate:

* structural absence;
* unknown categorical values;
* unknown numerical values.

#### Arrow warning in inherited houses display table

After transposing the inherited houses predictions table, Streamlit produced an Arrow serialisation warning because table columns contained mixed numeric and string values.

The issue was fixed by formatting the transposed display table as strings for presentation only.

### Unfixed Bugs

There are no known unfixed bugs.

---

## Deployment

The project is designed to be deployed to Heroku.

The repository contains the files required for Heroku deployment:

| File               | Purpose                                                 |
| ------------------ | ------------------------------------------------------- |
| `Procfile`         | Tells Heroku how to run the Streamlit app               |
| `setup.sh`         | Creates Streamlit server configuration for Heroku       |
| `requirements.txt` | Lists Python dependencies                               |
| `runtime.txt`      | Specifies the Python runtime required by the assessment |
| `.python-version`  | Specifies the Python version used locally               |

### Local Deployment

To run the project locally:

1. Clone the repository:

```bash
git clone https://github.com/wymmij/ci-portfolio-project5
````

2. Change into the project directory:

```bash
cd ci-portfolio-project5
```

3. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

4. Install requirements:

```bash
python -m pip install -r requirements.txt
```

5. Run the Streamlit app:

```bash
streamlit run app.py
```

### Heroku Deployment Steps

The app can be deployed to Heroku using the following process:

1. Create a new Heroku app.
2. Connect the Heroku app to the GitHub repository.
3. Ensure the deployment branch is selected.
4. Deploy the branch manually or enable automatic deployment.
5. Open the deployed app from the Heroku dashboard.

### Deployment Files

The `Procfile` contains:

```text
web: sh setup.sh && streamlit run app.py
```

The `setup.sh` file creates the Streamlit configuration needed for Heroku.

The project uses Python 3.12.

---

## Main Data Analysis and Machine Learning Libraries

The project uses the following main libraries:

| Library      | Purpose                                                              |
| ------------ | -------------------------------------------------------------------- |
| pandas       | Data loading, cleaning, analysis, and dataframe handling             |
| numpy        | Numerical operations                                                 |
| matplotlib   | Static visualisations in notebooks                                   |
| seaborn      | Statistical visualisations in notebooks                              |
| plotly       | Interactive dashboard charts                                         |
| scikit-learn | Machine learning models, model evaluation, and hyperparameter tuning |
| joblib       | Saving and loading fitted models and project artifacts               |
| Streamlit    | Interactive dashboard application                                    |

---

## Project Files

The main project files and folders are:

| Path                                             | Purpose                                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------------------- |
| `app.py`                                         | Main Streamlit app entry point                                               |
| `app_pages/`                                     | Streamlit dashboard page modules                                             |
| `src/`                                           | Reusable project code                                                        |
| `src/data_management.py`                         | Data and model artifact loading functions                                    |
| `src/feature_metadata.py`                        | Feature labels, help text, and category metadata                             |
| `src/machine_learning/predictive_analysis_ui.py` | Prediction form and prediction-preparation logic                             |
| `jupyter_notebooks/`                             | Project notebooks                                                            |
| `outputs/datasets/collection/`                   | Collected project datasets                                                   |
| `outputs/datasets/cleaned/`                      | Cleaned train, test, and inherited-house datasets                            |
| `outputs/datasets/featured/`                     | Feature-engineered datasets                                                  |
| `outputs/ml_pipeline/predict_sale_price/v1/`     | Saved model, model performance, feature importance, and prediction artifacts |

---

## Credits

### Dataset

The dataset was provided by Code Institute through the Heritage Housing Issues project dataset on Kaggle.

### Code and Project Structure

This project was built as part of the Code Institute Full Stack Software Development Diploma.

The project structure, Streamlit dashboard approach, notebook workflow, and machine learning project expectations were informed by Code Institute course materials and walkthrough projects.

### Documentation and Learning Resources

The following resources were used during development:

* Code Institute course materials;
* Code Institute Heritage Housing Issues project brief and dataset;
* Code Institute walkthrough project materials;
* scikit-learn documentation;
* Streamlit documentation;
* pandas documentation;
* Plotly documentation;
* Heroku deployment documentation.

### AI Assistance

AI assistance was used during project development for explanation, code review, and refactoring suggestions.

---

## Acknowledgements

Thank you to Code Institute for the project brief, learning materials, and assessment structure.
