# SC1015-Mini-Project

## !Important!:
>As the dataset on [Kaggle](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset) is very large in nature, please download the subset of the dataset used.<br>
>This dataset has completed the pre-processing stage. Users will not have to run **preparation.ipynb** again!<br>
>[Train Data](https://entuedu-my.sharepoint.com/:f:/g/personal/hhan012_e_ntu_edu_sg/Em9MKOz90vJKskR3ACUK_aUBIc4j5DbqKClyHc5sqMU17g?e=6P6FRu)<br>
>[Test Data](https://entuedu-my.sharepoint.com/:f:/g/personal/hhan012_e_ntu_edu_sg/EkAdQcu1KpJLtR8PBBXO5ZsBKQkZPSeYMuLhfawrca25CQ?e=p43DB3)<br>

<br>

>If you experience difficulties in download the model file, please download it using this link:
>[model](https://entuedu-my.sharepoint.com/:u:/g/personal/hhan012_e_ntu_edu_sg/EW31jJzivGdHs0ZwauhULU4BROMxnMiSmRf2dEUUgd-YYA?e=nw5zxh)<br>

<br>

>This is how your directory should look like:<br>
![image](https://github.com/tanszejin/SC1015-Mini-Project/assets/127087818/131efee9-d206-4c11-a1ee-e2517a51255c)


<br>

>Install necessary libraries using **requirements.txt**
>>pip install -r requirements.txt

---
# Outline

### Problem: 
Can we make use of machine learning to predict what instrument is being played from an audio file? We explore the MusicNet dataset from Kaggle to train our various machine learning methods.

### Dataset:
The dataset was sourced from https://www.kaggle.com/datasets/imsparsh/musicnet-dataset

### Pre-Processing: 
Pre-processing is done by selecting solo music files and segmenting them into 1-second audio.

### Exploratory analysis and data visualisation:
We will explore the dataset's metadata and plot them in boxplots, histograms and violin plots. We will then use audio analysis techniques to explore the audio data, such as waveforms, spectrograms, MFCC heatmaps, scatterplots and more.

### Pattern recognition: 
We will then find the correlation coefficients of each pair of attributes and plot them in a correlation matrix.

### Machine learning: 
We will use the dataset to train a neural network, classification tree and random forest, then analyse the prediction accuracies.

### Intelligent decision: 
Finally, with the results obtained, we will determine which machine learning method is the most effective for this task

---
# Results

## Classification Tree:
As the identification of the audio samples' instrument is a classic classification problem, we used a classification tree, also known as a decision tree. We extracted four features from the audio dataset which we have found to be useful for classification during the EDA, and trained the classification tree using these features.   
> Classification accuracy for tree of max depth 2 = 83.97%  
> Classification accuracy for tree of max depth 3 = 85.81%  
> Classification accuracy for tree of max depth 4 = 94.07%  

## Random Forest: 
Random Forest is a versatile and powerful ensemble machine learning method that operates by constructing multiple decision trees during training and outputs the consensus of these trees for prediction tasks. Primarily used for classification and regression, Random Forest improves on the performance and accuracy of single decision trees by building a forest of them, each trained on random subsets of the data and features. The intuition behind Random Forest is to benefit from the collective decision-making of multiple models, mitigating the risk of errors associated with any single tree, particularly overfitting. When predicting, in classification tasks, each tree votes for a class, and the class receiving the majority of votes is chosen as the final output. In regression, it takes the average of the outputs across all trees. This averaging helps to reduce variance and improve the robustness of the model. When using random forest for our musical instrument classification, it achieved a high accuracy across all 4 instruments as shown below: <br />
>Cello accuracy rate = 99.2%<br />
>Piano accuracy rate = 99.1%<br />
>Violen accuracy rate = 100%

## Wav2Vec2:
Using Hugging Face's transformers library, Wav2Vec2 is a popular tool for automatic speech recognition (ASR). However, for our project, we will be training it to classify classical instruments instead.<br>
>Validation Accuracy: 99.85%

---
# Contributions
**Lee Yi Ming, Bennett:** ideation, random forest analysis, slides, presentation, video editing  
**Han Xin Yi, Heidi:** ideation, data preprocessing, wav2vec2, slides, presentation  
**Tan Sze Jin:** ideation, exploratory data analysis, classification tree, slides, presentation  

---
# References
Ghantiwala, A. (2022, April 26). Visualizing Audio Data and Performing Feature Extraction. Medium. https://towardsdatascience.com/visualizing-audio-data-and-performing-feature-extraction-e1a489046000  
SriVinayA. (2023, November 2). 🎙️ Exploratory Data Analysis and Processing of an Audio Dataset 🎵. Medium. https://medium.com/@SriVinayA/%EF%B8%8F-exploratory-data-analysis-and-processing-of-an-audio-dataset-4ff47a0e815a  
