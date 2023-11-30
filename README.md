# Python-for-Data-Analysis project
Valentin ALAKILETOA-PINAULT, Elies BENMALEK, Safia ALIOUCHE.

Find the dataset here : https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified

The dataset is a survey of 1885 people regarding their drug use. They answered questions regarding their age, ethnicity, user of 18 different drugs and personality scores…Here is the information on each interested person:
###	Person information :
-	ID
-	Age
-	Gender
-	Education
-	Country
-	Ethinicity
### Personality information :
-	Escore (Extraversion)
-	Oscore (Openess to experience)
-	Ascore (Agreeableness)
-	Cscore (Conscientiousness)
-	Impulsive (Impulsiveness)
-	SS (Sensation)
### Information about their drug use :
- Alcohol
- Amphet
- Amyl
- Benzos
- Caffeine
-	Cannabis
-	Chocolate
-	Coke
-	Crack
-	Ecstasy
-	Heroin
-	Ketamin
-	Legalh
-	LSD
-	Meth
-	Mushrooms
-	Nicotine
-	Semeron
-	VSA


## The completed project is broken down into 4 parts:
-	Data-Preprocessing: Data cleaning, implementation of the legend provided with the dataset.
-	Data-Visualization: Exploration of data, search for correlation between data.
-	Moduling: Implementation of different models based on their personality in order to know if a user is a consumer or not.
-	Django: Allows you to display graphs.


The objective of this project is to predict whether or not an individual will use a drug based on their personality characteristics.


## After modeling we can conclude that:
We can create models that are effective in predicting whether a person is a consumer or not. On the other hand, we also notice that some models are better than others and give better results depending on the drug. We observed that Logistic regression is indeed a better model than Random Forest and KNN, because it has a higher AUC score for our dataset. On the other hand, between Random Forest and KNN, we cannot choose the car that depends on drugs.

It should also be noted that each of its models have their own strengths and weaknesses and are adapted to different types of data and prediction tasks. For example, logistic regression is a simple and quick model that works well when the relationship between features and the target variable is approximately linear. On the other hand, Random Forest is a more complex model that can capture nonlinear relationships and interactions between entities, but it is more complex.

