# **Predicting Groundwater Levels in Spokane, Washington**
The purpose of this project is to use a neural network modeling approach to predict groundwater levels in Spokane, Washington using weather and surface water data. The final outcome can be viewed in the Streamlit App.
# **Description**

## **Overview**

Groundwater is a critical source of water for human survival. A significant percentage of both drinking and crop irrigation water is drawn from groundwater sources through wells. In the US, overuse of groundwater could have major implications for the future.

Machine learning, deep learning, and time series techniques have shown promise for helping understand and address these problems. In this project  we implemented Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) to predict Groundwater Levels (GWL) in selected wells located in Columbia Plateau basaltic-rock and basin-fill aquifers: one of the principal aquifers of the U.S  using  historical well, weather and surface data such as  temperature, precipitation,  humidity, wind speed , river height and discharge rate etc.

Ultimately, our goal is to be able to provide a forecast of groundwater levels while accounting for uncertainties that can be used to help inform water use policy decisions for stakeholders. An interactive web application using Streamlit dashboard was also implemented to provide end-users friendly graphical interface for predicting and forecasting groundwater levels.

**Stakeholders:** General Public, Agriculture and Farming businesses, Real estate and Property developers, Local Government Agencies such as Spokane County Water Resources, Spokane Water Department and Washington State Department of Ecology.

**KPIs for Groundwater levels prediction and forecast:** Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) between the true and predicted groundwater levels.

## **Dataset**

* Data sets required for the project: water well, surface water, and weather data.
  * Groundwater monitoring well data obtained from USGS, including [AEK201](https://cida.usgs.gov/ngwmn/provider/WAECY/site/100018881/), [APK309](https://cida.usgs.gov/ngwmn/provider/WAECY/site/100080103/), [APK310](https://cida.usgs.gov/ngwmn/provider/WAECY/site/100080102/), and [AFL259](https://cida.usgs.gov/ngwmn/provider/WAECY/site/100079507/) (used as the test set due to similar lithography to [AEK201](https://cida.usgs.gov/ngwmn/provider/WAECY/site/100018881/)).
  * Weather data sourced from OpenWeather.com and NOAA, covering precipitation, temperature, humidity, pressure, and wind speed.
  * Surface water data acquired from USGS.
* Data processing started with pickling each data set.
  * Monitoring well data was sorted by date/time, and measurements from 2006-2017 averaged for the same day.
  * Weather and surface water data time frame aligned with monitoring well data.
  * Correlation coefficients were calculated for weather features; only average values utilized.
  * NOAA was used for precipitation data with a 45-day lag.
  * OpenWeather.com weather data contained NaN values replaced with zeros.
  * Surface water data had missing values, filled with the value before the gap.

## **Approach and Results**

Using both models we successfully predicted GWL for the case study wells with high accuracy. Strategies implemented and results achieved are summarized below:
* Baseline: We wanted a simple baseline against which we could compare more later models, so we chose to start with a simple model that predicts a constant output based on the average of the target values in the training set.
* Linear Regression: As a second sort of baseline option, we used Scikit-Learn’s built in Linear Regression module to train a linear model on the training data.
* CNN: While CNN’s are not typically associated with time series data, we did see examples where they had been applied to problems with similarities to ours. We were curious to see whether we could construct a model with this structure that could perform comparably to other models.
* LSTM: A neural network typically used for time series data that was compared to CNN.

## **Future Iterations**
* Seasonal ARIMA
* Additional parameter tuning
* Model serialization
* Wells at other locations
* More interactive web Interface

# **Authors**
* [Marcos Ortiz](https://www.linkedin.com/in/passpassthemath/)
* [Meredith Sargent](https://www.linkedin.com/in/meresar/)
* [Riti Bahl](https://www.linkedin.com/in/ritibahl/)
* [Chelsea Gary](https://www.linkedin.com/in/chelseargary/)
* [Anireju Emmanuel Dudun](https://www.linkedin.com/in/anireju-emmanuel-dudun-78359153/)
