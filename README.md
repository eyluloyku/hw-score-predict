# hw-score-predict

This project aims to predict the homework grades of students from their ChatGPT histories. Each student was asked to complete their homework by using ChatGPT and share their chat histories beforehand. After manual grading of the homeworks, different ML models were trained to predict the grade of a student from the history. You can find the details about the dataset, approaches and results below. Feel free to reach out to me for further discussion!

## Dataset Properties
- #HTML files: 127
- #scores: 145
Some students did not upload ChatGPT conversations or the conversation is empty, hence the inconsistency in number of files and scores. We still need to address this problem while predicting grades.

## Repository Overview
- combined.ipybn: This notebook contains 5 major approaches (each defined below) and the final ensembling of the best 3 of them.
  - Approach 1: Predicting scores from features extracted from text such as number of certain words, average chars, total predicted score based on prompt matching etc.
  - Approach 2: Clusterin using K-means and Hierarchical Clustering approaches.
  - Approach 3: Classification of grade bins and regression from predicted bins for score prediction.
  - Approach 4: Regression from text models using BoW, Word2Vec and TF-IDF on both prompts and answers.
  - Approach 5: Merging features with text vectors, applying PCA and Simulated Annealing for feature subset selection.
- oversampling.ipynb: Oversampling using SMOTE. Did not improve performance of the methods.
- ranking.ipynb: Comparing pair of students to determine who got higher grades than others. Not finished.
- softmax_two_class.ipynb: Treating students getting grade 100 as a class and others as a seperate class. Determining the scores based on probability of belonging to class 1. Did not work well.

## Methodology
### Feature Mapping
We started by extracting features from prompts and answers. We added the number of certain words that are used when getting an error such as "why" or "error", as well as words specific to the homework context such as "hyperparameters". After mapping the questions to the prompts asked, we calculated the average and standart dev of mapping scores for a students along with the total grade predicted by multiplying points of question by mapping score of question for that student. They all proved to be correlated with the grade.
####  Missing Data
As some students did not provide ChatGPT histories or the provided ones were empty, we filled the values of these students with the average of K-nearest neighbors based on grades. 
#### Preprocessing
We tried to employ text preprocessing techniques like lemmatization, stemming, removing punctuation but they decreased the metrics significantly. We believed that this is due to the feature of the assignment, as code blocks are given and taken as input to/from ChatGPT. That is why the notebooks only apply converting to lower case. 
### Clustering
After getting the features for all students we applied K-means and Hierarchical Clustering on the data with elbow curve approach. After clustering the students we tried two approaches: training a regressor for each cluster and using the cluster as a feature on whole data. We also tried to cluster the data based on vector representations of prompts and answers, but we abandoned that approach as it performed worse.
### Classification of Grade Bins 
Based on the distribution of the grades, we divided the students into grade bins. Then we used both the vector representations of the texts and the features to predict the bin that student falls into. For the former approach, we used simple neural networks. Simultaneously we trained a model to predict the grade with the assigned bin. However, when testing the grade predictions we changed the assigned bin to predicted bin. Again, we used two approaches while predicting grade: training a regressor for each bin and using the bin as a feature on whole data.
### Regression from Text
Rathen than using features extracted from the texts seperately, we also tried to train regressor on vector representations of the texts. We fit CountVectorizer, TF-IDFVectorizer and Word2Vec on prompts and answers seperately. Then we predicted the grade from each of the 6 combinations and used soft-voting and weighted-soft-voting approaches to ensemble the results of them. 

Later on, we concatanated the vector representation with the features extracted in the beginning. We applied PCA on the data to get the best performing features. Also, we used a heuristic called Simulated Annealing to determine the best combinations of features and predicted results with them. 
### Ensemble
Lastly, we gathered the best predictions together and ensembled them using weighted-soft-voting. You can find the detailed results below for each method and final prediction.

## Results
AAA
