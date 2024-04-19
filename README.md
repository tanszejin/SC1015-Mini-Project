# SC1015-Mini-Project

## Problem: 
Can we make use of machine learning to predict what instrument is being played from an audio file? We explore the musicnet dataset from Kaggle to train our various machine learning methods.

## Data cleaning: 
To clean up the dataset, we will implement a cleaning function on the audio files from the dataset to remove any possible noise from the background.

## Exploratory analysis and data visualisation:
We will explore the daset's metadata and plot them in boxplots, histograms and violin plots. We will then use audio analysis techniques to explore the audio data, such as waveforms, spectrograms, MFCC heatmaps, scatterplots and more.

## Pattern recognition: 
We will then find the correlation coefficients of each pair of attributes and plot them in a correlation matrix.

## Machine learning: 
We will use the dataset to train a neural network, classification tree and random forest, then analyse the prediction accuracies.

## Intelligent decision: 
Finally, with the results obtained, we will determine which machine learning method is the most effective for this task

## Random Forest: 
Random Forest is a versatile and powerful ensemble machine learning method that operates by constructing multiple decision trees during training and outputs the consensus of these trees for prediction tasks. Primarily used for classification and regression, Random Forest improves on the performance and accuracy of single decision trees by building a forest of them, each trained on random subsets of the data and features. The intuition behind Random Forest is to benefit from the collective decision-making of multiple models, mitigating the risk of errors associated with any single tree, particularly overfitting. When predicting, in classification tasks, each tree votes for a class, and the class receiving the majority of votes is chosen as the final output. In regression, it takes the average of the outputs across all trees. This averaging helps to reduce variance and improve the robustness of the model. When using random forest for our musical instrument classification, it achieved a high accuracy across all 4 instruments as shown below: <br />
Cello accuracy rate = 99.2%<br />
Flute accuracy rate = 89.6%<br />
Piano accuracy rate = 99.1%<br />
Violen accuracy rate = 100%
