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
- 
