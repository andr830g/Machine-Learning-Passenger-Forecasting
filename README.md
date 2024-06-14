# From Black-Box to Explainable Insights: Forecasting Passenger Demand using APC Data
Bachelor Thesis: 6. semester, Aarhus University

Andreas Hyldegaard Hansen 202106123,
Andreas Skiby Andersen 202107332

<img src="assets/Passenger_Count" width="400"/> <img src="assets/XGB_Forecast" width="400"/>

## Abstract
Implementation of Automatic Passenger Counting (APC) systems in public transportation allows
public transport providers to collect and analyze Big Data, which is crucial for modern demand
planning and scheduling. APC data enables time series forecasting of future passenger demand.
Danish research, regarding the analysis of bus passenger behavior and time series forecasting of
bus passenger demand, is very limited, which highlights the importance of research.

The field of time series is rapidly expanding with "black-box" machine learning and neural net-
work methods being favored above traditional linear methods. "Black-box" methods may increase
accuracy, but limits transparency in the training and inference process, resulting in models with
learning patterns that cannot fully be explained.
This project evaluates a Seasonal Naive model (SNaive), an autoregressive Lasso, Random For-
est Regression (RF), XGBoost (XGB), Vanilla Reccurent Neural Network (RNN) and Long Short-
Term Memory (LSTM) on Midttrafik’s APC data from June 2021 to December 2023 in Aarhus,
Denmark. Data is aggregated on 60-, 30- and 15-minute intervals.
We show that Fourier Transformations are efficient for analyzing seasonal patterns, and find that
including daily lags for the past week is optimal for all non-differenced models. We also show
that the Diebold-Mariano test is effective for testing the statistical significance of performance
between forecasts from different models.

Our results show that XGBoost has the highest accuracy, between 85.4%-90.5% nMAE, with a 2
day forecast horizon on the test data, for 60-minute aggregation level, compared to 76.6%-84.8%
nMAE for the benchmark.
While XGBoost is shown to be the most robust method for lower aggregation levels, we find that
increasing the forecast horizon from 2 days to 2 weeks only gave up to around 26% relative in-
crease in nMAE for Random Forest, XGBoost, Vanilla RNN and LSTM, and a 40% relative in-
crease for the benchmark model.
A compelling aspect of the project is centered around explaining "Black-Box" models using Perfor-
mance Based Shapley Values (PBSV); a new variant of SHAP values. For XGBoost, PBSV shows
that including daily and weekly lags, information about peak hour and type of bus schedule has
the highest importance, while weather variables have no significant importance.

<img src="assets/XGB_SHAP" width="400"/> <img src="assets/XGB_PBSV" width="400"/>

## GitHub
### assets
+ Contains the final report of the thesis in the "Report.pdf" file.
+ Contains images for "README.md"

### hyperparameterSearch
Contains output files from hyperparameter search for optimal hyperparameters for each variant of each model.

### predictions
Contains forecast predictions.

### utils
Contains code for functions used throughout the analysis files. Notably, it contains the code for implementation of recursive forecasting for scikit-learn and PyTorch models.
+ The default implementation of recursive forecasting with skforecast using scikit-learn models demands the use of lags. Our implementation can handle the use of lags as well as the exclusion of lags.
+ We found no implementation of recursive forecasting using PyTorch models. This has been implemented in this project in the "Pytorch.py" file in the "utils" folder.
