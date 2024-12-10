# Python| Supervised Machine Learning| Ecommerce | Customer Churn
Using Python to build ML model to predict customer churn rate from different factors (Supervised).

## Context Overview
In the competitive world of e-commerce, customer retention is a crucial determinant of long-term success. Our company has observed an alarming trend in user behavior: an increasing number of customers are churning—abandoning their carts, failing to complete purchases, and ceasing interactions altogether. To address this challenge, our initiative focuses on predicting churned users and offering strategic promotions to re-engage them. By understanding the factors that lead to churn, we can not only recover lost revenue but also enhance customer relationships and foster loyalty.

Why Use Machine Learning?

Traditional methods of analyzing churn rates, while valuable, often fall short in capturing the complexities of customer behavior. Machine learning offers a powerful alternative, enabling us to uncover hidden patterns and relationships within large datasets. By leveraging predictive algorithms, we can automate the identification of at-risk customers, allowing us to implement timely and targeted engagement strategies. This proactive approach not only enhances our agility in responding to churn signals but also provides a scalable solution that evolves alongside changing customer behaviors.

## EDA
After importing the dataset into Google Colab, we will conduct an exploratory data analysis (EDA) to deepen our understanding of the data. This process is likely to yield valuable insights that could enhance the performance of our supervised learning model in subsequent steps.

Let's ".describe()" the dataset a bit

![image](https://github.com/user-attachments/assets/127e02ef-cee0-429c-aaa7-b1078856d19a)

### Numerical Columns Analysis

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
### Categorical Columns Analysis
![image](https://github.com/user-attachments/assets/7bfd6719-ca0a-4fcf-9bb4-e195ca1647b8) ![image](https://github.com/user-attachments/assets/24b9c859-f381-4bd0-ab71-e50b12bf4707)


Preferred Login Device:
- Insights: The majority of customers (49%) prefer to log in using a mobile phone, followed by computers and phones. Note, phone might overlap with mobile phone, which inidcates the domination of phone users.
- Implications: Churning could be related to the device used for logging in. Understanding this pattern may reveal that mobile users face unique challenges or preferences that lead to higher churn rates. This insight could inform the development of mobile-specific features or improvements to enhance user satisfaction.

Preferred Payment mode:
- Insights: The majority of customers (41%) prefer to use debit cards for payment, followed by credit cards, e-wallets, and UPI.
- Implications: Payment preference could indicate customer loyalty and purchase behavior. Understanding which payment methods are associated with high-value transactions can help tailor promotions or offers. Additionally, analyzing churn patterns could uncover whether customers who prefer certain payment methods are more likely to leave, providing an avenue for targeted retention efforts. Besides, payment methods with more churned customers could be targeted for improvements.

Gender
- Insights: The majority of customers (60%) are male, while 40% are female.
- Implications: Gender demographics may influence purchasing behaviors and preferences. Insights into gender-related tendencies can help optimize marketing strategies and product recommendations. Furthermore, understanding any churn correlations within gender groups could reveal opportunities for targeted retention campaigns designed to meet the interests of different customer segments.
Preferred Order Category:
- Insights: The majority of customers (36%) prefer to order laptops and accessories, followed by mobile phones, fashion products, and mobile-related products.
- Implications: Product preferences can guide inventory management and targeted marketing campaigns. Recognizing that a significant portion of customers prefers laptops and accessories indicates the necessity to tailor promotions or features towards these products. Analyzing order patterns may also help identify if churn is linked to dissatisfaction with certain product categories.
Marital Status
- Insights: The majority of customers (53%) are married, followed by single customers (32%), and divorced customers (15%).
- Implications: Marital status may impact purchasing behavior and lifecycle stages, such as family-oriented product needs. Understanding these dynamics can lead to personalized marketing strategies and product offerings. It’s also worth examining whether marital status correlates with churn rates to inform retention strategies that address the specific needs of different customer segments.
### Correlation Analysis
![image](https://github.com/user-attachments/assets/dbcabdc0-c26f-498b-8454-eb6b98949794)

Strong correlations (> 0.5)
- CashbackAmount and Tenure (0.467986): There's a moderate positive correlation between the cashback amount and the tenure of a customer. This suggests that customers who have been with the company longer tend to receive more cashback.
- OrderCount and CouponUsed (0.641178): There's a strong positive correlation between the number of orders and the usage of coupons. This implies that customers who use coupons tend to make more orders. This indicates a good incentive to retain customers.
Moderate correlations (0.3 to 0.5)
- HourSpendOnApp and CouponUsed (0.187166): There's a moderate positive correlation between the time spent on the app and the usage of coupons. This suggests that customers who spend more time on the app are more likely to use coupons.
- DaySinceLastOrder and CashbackAmount (0.316568): There's a moderate positive correlation between the time since the last order and the cashback amount. This implies that customers who have a longer time since their last order tend to receive more cashback. Is this detrimental in a way?
Weak correlations (< 0.3)
- WarehouseToHome and HourSpendOnApp (0.064069): There's a weak positive correlation between the warehouse-to-home time and the time spent on the app. This suggests that there's not a strong relationship between these two variables.
- OrderAmountHikeFromlastYear and CouponUsed (0.024482): There's a weak positive correlation between the order amount hike from last year and the usage of coupons. This implies that there's not a strong relationship between these two variables.
Insights and potential actions
- Targeted marketing: Identify customers who have been with the company longer and offer them more cashback opportunities to increase loyalty.
- Coupon strategy: Analyze the coupon usage pattern and consider offering targeted coupons to customers who are more likely to use them (e.g., those who spend more time on the app).
- App engagement: Investigate ways to increase engagement on the app, as it's correlated with coupon usage and potentially other desirable behaviors. Gamification for coupon is a good example for this case.

### Checking distribution balance of the Churn column

Balancing an imbalanced distribution in a supervised learning project is crucial for reasons:
- Model Performance:Bias Toward Majority Class: In imbalanced datasets, models often become biased toward the majority class. For example, if 84% of your data is from class 0 (no churn), the model might predict class 0 for most cases. This can lead to high accuracy but poor performance in identifying the minority class.
- Underperformance in Minority Class Prediction: The model might completely miss recognizing examples of the minority class, which could be critical in applications (like fraud detection or churn prediction).
Evaluation Metrics:
- Misleading Accuracy: With imbalanced classes, accuracy alone is a poor metric. A model could achieve high accuracy by simply predicting the majority class most of the time, neglecting the minority class.
- Importance of Other Metrics: Metrics such as Precision, Recall, and F1 Score are more informative in imbalanced contexts because they specifically measure the model's ability to predict the minority class correctly.

Real-World Consequences: In many cases, failing to predict minority class instances can have significant consequences. For instance, in a churn prediction scenario, missing a customer who is about to leave could lead to revenue loss.

![image](https://github.com/user-attachments/assets/c344147d-4fd0-4d3e-b936-8ddc0d3f6d8f)

The data distribution is considered balanced when the percentage of value 1 falls between 20% and 40%. Currently, this is not the case, so I'll need to adjust the data accordingly to achieve a more balanced distribution.


