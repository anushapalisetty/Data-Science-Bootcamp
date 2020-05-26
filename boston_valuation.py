from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Gather data
boston_dataset=load_boston()
data=pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)
features=data.drop(["INDUS","AGE"],axis=1)
target=pd.DataFrame(data=np.log(boston_dataset.target),columns=["PRICE"])

RM_IDX=4
PT_RATIO_IDX=8
CHAS_IDX=2

ZILLOW_MEDIAN_PRICE=583.3
SCALE_FACTOR=ZILLOW_MEDIAN_PRICE/np.median(boston_dataset.target)

property_stats=features.mean().values.reshape(1,11)
regr=LinearRegression().fit(features,target)
fitted_vals=regr.predict(features)

MSE=mean_squared_error(target,fitted_vals)
RMSE=np.sqrt(MSE)

def get_log_estimate(nrooms, students_per_class,next_to_river=True,high_confidence=True):
    
    #configure property
    property_stats[0][RM_IDX]=nrooms
    property_stats[0][PT_RATIO_IDX]=students_per_class
    
    if next_to_river:
        property_stats[0][CHAS_IDX]=1
    else:
        property_stats[0][CHAS_IDX]=0
    
    #Make prediction
    log_estimate=regr.predict(property_stats)[0][0]
        
    if high_confidence:
        upper_bound= log_estimate + 2 * RMSE
        lower_bound= log_estimate - 2 * RMSE
        interval=95
        
    else:
        upper_bound= log_estimate + RMSE
        lower_bound= log_estimate - RMSE
        interval=68
    
    return log_estimate,lower_bound,upper_bound,interval


def get_dollar_estimate(rm,ptratio,chas=False,large_range=True):
    """Estimate the price o fpropety in BOSTON

    """
    if rm <1 or ptratio <1:
        print("Unrealistic")
    else:
        log_est,low,upper,conf=get_log_estimate(rm,ptratio,chas,large_range)
        dollar_est=round(np.e**log_est * 1000 * SCALE_FACTOR,-3)
        dollar_hi=round(np.e**upper * 1000 * SCALE_FACTOR,-3)
        dollar_low=round(np.e**low * 1000 * SCALE_FACTOR,-3)

        print(f"Estimated property {dollar_est}")
        print(f"At {conf} confidence % the valuation range is")
        print(f"USD  {dollar_low} at lower end to USD {dollar_hi} at high end" )