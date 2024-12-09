# Python| Supervised Machine Learning| Ecommerce | Customer Churn
Using Python to build ML model to predict customer churn rate from different factors (Supervised).

## Context Overview
One ecommerce company has a project on predicting churned users in order to offer potential
promotions. 

## EDA
After importing the dataset into Google Colab, we will conduct an exploratory data analysis (EDA) to deepen our understanding of the data. This process is likely to yield valuable insights that could enhance the performance of our supervised learning model in subsequent steps.

Let's ".describe()" the dataset a bit

![image](https://github.com/user-attachments/assets/127e02ef-cee0-429c-aaa7-b1078856d19a)

Column Analysis

CustomerID
*   Count: 5,630 unique customers.
Note: This variable is not predictive of churn and primarily serves as an identifier.

Churn
*   Mean: a relatively low mean (0.168), indicating that about 16.8% of the customers are considered churned.
*   Implication: Important for modeling to understand customer retention.

Tenure
*   Mean: Approximately 10.19 months, with a wide range of tenure (0 to 61 months).
*   Std Dev: 8.56 months, indicating significant variability in customer tenure.
*   Implication: Longer tenure may correlate with retention; investigate its effect on churn.

CityTier
*   Mean: Approximately 1.65, suggesting that most customers are from urban areas (typically tier 1).
*   Max: The maximum value is 3, suggesting three categories of city tiers.
*   Implication: May relate to service or product accessibility, worth exploring in relation to churn.

WarehouseToHome
*   Mean Distance: About 15.64 (in unspecified units, likely km).
*   Range: From 5 to 127, indicating substantial variance in customer locations.
*   Implication: Distance may affect delivery perceptions; analyze its relationship with churn.

HourSpendOnApp
*   Mean: About 2.93 hours spent on the app, indicating moderate engagement.
*   Min: Equal to 0, indicating misclick? Might remove zero to access the true min hours.
*   Max: Up to 5 hours, suggesting a ceiling in user engagement.
*   Implication: Higher engagement might be correlated with lower churn—an area for targeted marketing.
NumberOfDeviceRegistered
*   Mean: Approximately 3.69 devices per customer.
*   Max: 6 devices, indicating some users engage across multiple platforms, which may enhance interaction.
*   Implication: Device diversity could correlate with retention; explore further.
SatisfactionScore
*   Mean: About 3.07 on a likely 5-point scale, indicating moderate satisfaction.
*   Std Dev: 1.38 suggests moderate variability in customer experiences, not reaching the max score.
*   Implication: Satisfaction may influence churn; further investigation is warranted, especially given some satisfied users still churned.

NumberOfAddress
*   Mean: About 4.21 addresses per customer, indicating customer complexity in their transactions.
*   Implication: Multiple addresses may be linked to higher order counts,  spending patterns, or errors in delivery, worth exploring.
Complain
*   Mean: 0.28 complaints per customer, indicating around 28% of customers raised complaints.
*   Implication: Complaints are a potential churn indicator; investigate their impact on retention.

OrderAmountHikeFromLastYear
*   Mean: Approximately 15.71% increase in order amount, suggesting growth among returning customers.
*   Implication: Higher increases may reduce churn; track this metric closely.

CouponUsed
*   Mean: About 1.75 coupons used per customer, reflecting moderate use of discounts.
*   Implication: Coupon usage could be a churn mitigator; explore its correlation with retention.

OrderCount
*   Mean: Approximately 3.01 orders per month, with significant variance (max 16).
*   Implication: More orders could correlate with lower churn—critical for developing retention strategies.

DaySinceLastOrder
*   Mean: Approximately 4.54 days since the last order, suggesting regular ordering patterns.
*   Std Dev: 3.65, indicating variability.
*   Implication: Longer gaps could indicate churn risk—target campaigns to re-engage inactive users.

CashbackAmount
*   Mean: About 177.22, with a maximum of 324.99.
*   Implication: Cashback amounts are linked to retention; higher cashback might attract repeat customers.

The dataset appears to contain valuable predictors of churn, with significant variability across key metrics. Factors such as tenure, engagement (hours spent on the app), satisfaction scores, and order behavior are particularly promising for modeling customer retention. Additionally, the presence of missing values in several columns suggests that data cleaning and imputation strategies may be necessary before further analysis. Finally, integrating insights from EDA into predictive modeling will enhance our ability to identify at-risk customers effectively.

Another point that I was curious about when investigating the dataframe are the collected values of the HourSpendonApp column. There is some uncertainty regarding the rounding method used in the HourSpendonApp data. It is important to consider that customers may spend varying amounts of time on apps, such as just a few minutes or a full hour plus additional minutes. The presence of '0' hours in the data could be misleading, as it might represent different usage patterns. Therefore, it is necessary to discuss these nuances with the data collector to ensure accurate interpretation of the data. However, in this context, it's not possible to do so, which I would note as a limitation of this project.  

![image](https://github.com/user-attachments/assets/77e7b4ee-f7d0-4b4f-bb5d-e29ec3af5d29)

df.info() reveals the availability of some missing values in such numeric columns as Tenure, WarehouseToHome, HourSpendOnApp, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder. Thus, I would need to handle them with an imputing method, using mean or median. To determine whether to use the mean or median as a measure of central tendency, we first need to examine the data for skewness and the presence of outliers. The median is less influenced by outliers and provides a more accurate reflection of the central tendency in skewed distributions. Therefore, assessing the data's characteristics will help us make the appropriate choice. 

Firstly, I check the distribution of these columns to see if the values are normally distributed.

![image](https://github.com/user-attachments/assets/5bcb66d1-5088-4ee2-a680-74f3d6b7e541)
![image](https://github.com/user-attachments/assets/0b060f44-f63a-4b9a-94d1-bc65a7252e4c)

The distributions exhibit significant skewness. To confirm this observation, I will apply the Interquartile Range (IQR) method to identify outliers in these columns.

![image](https://github.com/user-attachments/assets/3593c1f9-b30d-42b6-b6cd-0826c19f4f14)

Then we should impute median instead. To accomplish this imputation, I would use SimpleImputer from sklearn.impute.

![image](https://github.com/user-attachments/assets/de1d7ff7-cbf0-431e-b388-1450771c07ea)


