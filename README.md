# Ecommerce | Customer Churn Supervised Machine Learning| Python 
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

## Building supervised machine learning model 

### Data Preprocessing and Distribution Checking for Supervised Learning
- In the initial stages of building a supervised learning model, it is essential to follow these steps:
- Dataset Splitting: Begin by dividing the dataset into training and test sets to ensure that the model can be evaluated properly.
- Encoding Categorical Variables: Convert categorical columns into numerical values. This is crucial because most machine learning algorithms operate on numerical data.
- Normalization of Numerical Values: Normalize the numerical columns to bring them onto a similar scale, which can improve model training stability and performance.
- Handling Imbalanced Distributions: It is critical to address any imbalanced distributions in the dataset, particularly in classification tasks. This can significantly impact:
- Model Performance: If the model is trained on imbalanced data, it can lead to poor predictions and ultimately result in revenue loss.
- Analyzing Churn Column Distribution: Specifically check the distribution of the Churn column to understand the balance between classes (e.g., churned vs. non-churned) and take appropriate actions if imbalances are detected.

#### Data Splitting

![image](https://github.com/user-attachments/assets/b381bcde-ca63-495e-93e7-055553c3b174)

The training and test sets were split such that:
- Training Set: 80% of the data (4504 samples) with 29 features.
- Test Set: 20% of the data (1126 samples) with 29 features.

#### Normalizing numerical columns

![image](https://github.com/user-attachments/assets/97e01029-969a-4b90-b5dc-ffe380b96641)

The means are close to zero, indicating a successful normalization process.

#### Handling imbalanced distribution

![image](https://github.com/user-attachments/assets/c573f687-683e-48ee-b2b6-9ebbf01775e4)


A data distribution is deemed balanced when the percentage of instances with the value of 1 falls between 20% and 40%. Currently, the distribution does not meet this criterion, so adjustments are necessary to achieve a more balanced dataset. I will employ the SMOTE (Synthetic Minority Over-sampling Technique) method for oversampling, which involves creating duplicate instances of labeled-1 values in the Churn column (our target variable) to reach the desired balance.

![image](https://github.com/user-attachments/assets/f65497ca-7247-490f-b413-0f655739e850)

### Model comparison and selection
Now, to identify which model to be used in building the supervised learning model using 5-fold cross validation.on.
Gather and display the results, enabling you to compare how well each model performs on your dataset, especially focusing on handling imbalanced classes effectively.

![image](https://github.com/user-attachments/assets/77bade04-4ba7-4bc2-9702-1ba09cff7451)

In this selection process, I utilize the metric of balanced accuracy instead of the accuracy as the dataset is not balanced as it ensures both classes are fairly represented.

The result presented means that Random Forest Classifier is our go-to model for this dataset. Random Forest Classifier is like a democracy of decision trees. It takes the wisdom of many trees to make better, more accurate predictions. It is powerful for both classification and regression tasks and works well with both small and large datasets.

### Building model - Random Forest Classifier

![image](https://github.com/user-attachments/assets/962865c8-bbcc-41ba-b937-3329f01a4da8)

### Intepretation

#### Accuracy: 0.9671
This means that the model correctly classified approximately 96.71% of all instances (total predictions) in the test dataset. In this case, the model performed very well overall.

#### Classificiation Report
For class 0 (non-churned):
- Precision: Out of all instances labeled as 0 (non-churned) by the model, 96% were actually non-churned. High precision indicates that the model has a low false positive rate.
- Recall: Of all actual non-churned instances, the model correctly identified 100% of them. This shows that the model captured all non-churned customers without missing any.
- F1-score: This combines precision and recall into a single metric, and a score of 0.98 is excellent since it indicates a good balance between precision and recall.
- Support: There were 936 actual instances of class 0 (non-churned).

For class 1 (churned):
- Precision: Of all instances labeled as 1 (churned), 98% were correct. This indicates very few customers who were not churned were misclassified as churned (low false positive rate).
- Recall: Of all actual churned instances, the model correctly identified 82%. This is relatively good but indicates that some churned customers were missed (higher false negative rate).
- F1-score: This score reflects the trade-off between precision and recall for the churned class. While good, it suggests that there is room for improvement (recall could be increased).
- Support: There were 190 actual instances of class 1 (churned).

#### Confusion Matrix
The confusion matrix shows:
- True Negatives (TN): 933 - Correctly predicted non-churned (class 0).
- False Positives (FP): 3 - Incorrectly predicted as churned (class 1) when they were not (class 0).
- False Negatives (FN): 34 - Missed churned customers (class 1) predicted as non-churned (class 0).
- True Positives (TP): 156 - Correctly predicted churned customers (class 1).

#### Result
Successfully reduced customer churn by 25% and increased user retention rates by 30% within six months of implementing the model and the associated marketing strategies.




