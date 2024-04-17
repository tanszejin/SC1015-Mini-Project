# SC1015-Mini-Project

## Problem: 
What do aspiring musicians need to look out for to make a popular song? We explore a dataset of Taylor Swift's discography to determine the best predictors for a song's popularity.

## Data cleaning: 
To clean up the dataset, we will add an additional column to indicate whether a song is a Taylor's version of the song, as this attribute might be a potential predictor of popularity. We will also remove extraneous non-alphanumeric symbols from song titles, and rescale the data for 'instrumentalness' to values between 0 and 1, making it easier to train on.

## Exploratory analysis and data visualisation:
We will plot the data in heatmaps, box plots, violin plots, scatter plots, multi-variate scatter plots and more.

## Pattern recognition: 
We will then find the correlation coefficients of each pair of attributes and plot them in a correlation matrix.

## Machine learning: 
We will use the dataset to train linear regression, classification tree and etc machine learning models, then analyse the goodness of fit and prediction accuracies to determine the best predictors of popularity.

## Intelligent decision: 
Finally, with the results obtained, we will conclude what makes a song popular.

## Random Forest: 
Random Forest is a versatile and powerful ensemble machine learning method that operates by constructing multiple decision trees during training and outputs the consensus of these trees for prediction tasks. Primarily used for classification and regression, Random Forest improves on the performance and accuracy of single decision trees by building a forest of them, each trained on random subsets of the data and features. The intuition behind Random Forest is to benefit from the collective decision-making of multiple models, mitigating the risk of errors associated with any single tree, particularly overfitting. When predicting, in classification tasks, each tree votes for a class, and the class receiving the majority of votes is chosen as the final output. In regression, it takes the average of the outputs across all trees. This averaging helps to reduce variance and improve the robustness of the model. When using random forest for our musical instrument classification, it achieved a high accuracy across all 4 instruments as shown below: <br />
Cello accuracy rate = 99.2%<br />
Flute accuracy rate = 89.6%<br />
Piano accuracy rate = 99.1%<br />
Violen accuracy rate = 100%
