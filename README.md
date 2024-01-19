# hw-score-predict

This project aims to predict the homework grades of students from their ChatGPT histories. Each student
was asked to complete their homework by using ChatGPT and share their chat histories beforehand.
After manual grading of the homeworks, different ML models were trained to predict the grade of a
student from the history. You can find the details about the dataset, approaches and results below. Feel
free to reach out to me for further discussion!

## Dataset Properties
- #HTML files: 127
- #scores: 145
Some students did not upload ChatGPT conversations or the conversation is empty, hence the
inconsistency in number of files and scores. We still need to address this problem while predicting
grades.

## Repository Overview
- combined.ipybn: This notebook contains 5 major approaches (each defined below) and the final
ensembling of the best 3 of them.
- Approach 1: Predicting scores from features extracted from text such as number of certain words,
average chars, total predicted score based on prompt matching etc.
- Approach 2: Clusterin using K-means and Hierarchical Clustering approaches.
- Approach 3: Classification of grade bins and regression from predicted bins for score prediction.
- Approach 4: Regression from text models using BoW, Word2Vec and TF-IDF on both prompts and
answers.
- Approach 5: Merging features with text vectors, applying PCA and Simulated Annealing for feature
subset selection.
- oversampling.ipynb: Oversampling using SMOTE. Did not improve performance of the methods.
- ranking.ipynb: Comparing pair of students to determine who got higher grades than others. Not
finished.
- softmax_two_class.ipynb: Treating students getting grade 100 as a class and others as a seperate class.
Determining the scores based on probability of belonging to class 1. Did not work well.
- regressor_from_text.ipynb: Seperate regressor from text, not merged with other features. Gives 0.46 R2 score.
## Methodology

### Feature Mapping
We started by extracting features from prompts and answers. We added the number of certain words
that are used when getting an error such as &quot;why&quot; or &quot;error&quot;, as well as words specific to the homework
context such as &quot;hyperparameters&quot;. After mapping the questions to the prompts asked, we calculated
the average and standart dev of mapping scores for a students along with the total grade predicted by
multiplying points of question by mapping score of question for that student. They all proved to be
correlated with the grade.
#### Missing Data
As some students did not provide ChatGPT histories or the provided ones were empty, we filled the
values of these students with the average of K-nearest neighbors based on grades.
#### Preprocessing
We tried to employ text preprocessing techniques like lemmatization, stemming, removing punctuation
but they decreased the metrics significantly. We believed that this is due to the feature of the
assignment, as code blocks are given and taken as input to/from ChatGPT. That is why the notebooks
only apply converting to lower case.
### Clustering
After getting the features for all students we applied K-means and Hierarchical Clustering on the data
with elbow curve approach. After clustering the students we tried two approaches: training a regressor
for each cluster and using the cluster as a feature on whole data. We also tried to cluster the data based
on vector representations of prompts and answers, but we abandoned that approach as it performed
worse.
### Classification of Grade Bins
Based on the distribution of the grades, we divided the students into grade bins. Then we used both the
vector representations of the texts and the features to predict the bin that student falls into. For the
former approach, we used simple neural networks. Simultaneously we trained a model to predict the
grade with the assigned bin. However, when testing the grade predictions we changed the assigned bin
to predicted bin. Again, we used two approaches while predicting grade: training a regressor for each bin
and using the bin as a feature on whole data.

### Regression from Text
Rathen than using features extracted from the texts seperately, we also tried to train regressor on vector
representations of the texts. We fit CountVectorizer, TF-IDFVectorizer and Word2Vec on prompts and
answers seperately. Then we predicted the grade from each of the 6 combinations and used soft-voting
and weighted-soft-voting approaches to ensemble the results of them.

Later on, we concatanated the vector representation with the features extracted in the beginning. We
applied PCA on the data to get the best performing features. Also, we used a heuristic called Simulated
Annealing to determine the best combinations of features and predicted results with them.

### Oversampling
Since there is not enough data available, both for the classification results and clusters, we had to create
synthetic samples. To accomplish that we have used SMOTE and created synthetic instances. Until the
classes or clusters become same size to have equally distributed classes. However, this approach did not
created value.

### SoftMax 2 Class
The purpose of using 2 classes was to assign probability to everyone to indicate their probability of being
100. It was assumed that this probability multiplied by 100 should indicate their grades. However, this
approach did not work as well as assumed.

### Learning to Rank Approach
It is checked pairwise which one is higher than the other for each pair. This task was completed as a
classification task with 3 classes. Classes were “higher”, “same” and “lower”. The accuracy of the
predictions was not good enough, so the regression part is not completed.

### Ensemble
Lastly, we gathered the best predictions together and ensembled them using weighted-soft-voting. You
can find the detailed results below for each method and final prediction.

## Results

### Feature Mapping
The below picture shows the correlations between variables. From these features we extracted the ones that have high correlation with grade and eliminated the others.

<img width="339" alt="image" src="https://github.com/eyluloyku/hw-score-predict/assets/116841987/40ab2a70-3b34-448a-a2a8-4727f4683483">

### Clustering Using K-Means

K-means clustering has been applied to clearly identify the differences and similarities between different points. As seen in the visual, even though there is no clear elbow, 6 is chosen as the best value. Similarly, thanks to having a small dataset, hierarchical clustering runs properly. 6 is also seen as a good value. These clusters are formed from features. Even though clustering based on text was tried, it did not give better results. 

<img width="320" alt="image" src="https://github.com/eyluloyku/hw-score-predict/assets/116841987/b08087f0-6594-439b-9950-f0aa5b75c399">
<img width="338" alt="image" src="https://github.com/eyluloyku/hw-score-predict/assets/116841987/355ca3d2-1acf-4279-922b-8f65563f639e">

Clustering and training different regressors for each cluster worked good for some clusters while others performed bad as seen below. The reason behind that is having a highly unbalanced data. It seems like there is no linear correlations among variables, but the error is low.


| Metric    | Train Set Score      | Test Set Score       |
|-----------|----------------------|----------------------|
| MSE       | 9.3909 (Model 1)     | 50.4272 (Model 1)    |
|           | 51.8259 (Model 2)    | 55.3250 (Model 2)    |
| R2 Score  | 0.9643 (Model 1)     | -1.1301 (Model 1)    |
|           | 0.8030 (Model 2)     | -1.3370 (Model 2)    |


Since cluster based regressors did not work well, we assigned clusters as a new feature to dataset before running a seperate regressor. It performed relatively well but still not the best.


| Metric    | Train Set Score      | Test Set Score       |
|-----------|----------------------|----------------------|
| MSE       | 39.0207              | 97.6039              |
| R2 Score  | 0.7612               | 0.1306               |

### Classification for Grade Bins
We binned data according to their grades. After that developed a classifier to predict the grade bins. Ensembled accuracy reached to 64% thanks to neural network models.

**Overall Accuracy**: 0.64

### Soft Voting

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.00      | 0.00   | 0.00     | 1       |
| 1     | 0.00      | 0.00   | 0.00     | 1       |
| 2     | 0.00      | 0.00   | 0.00     | 7       |
| 3     | 0.64      | 1.00   | 0.78     | 16      |

- **Macro Average**: Precision: 0.16, Recall: 0.25, F1-Score: 0.20
- **Weighted Average**: Precision: 0.41, Recall: 0.64, F1-Score: 0.50


However, since the bins are dominated by the higher classes and SMOTE did not work as well as expected, this approach also did not work as good as assumed. Most of the instances are in 2 class since data is not distributed evenly. However it should be noted that, if we were able to do that with 100% accuracy, R2 reaches to 90%. 

![image](https://github.com/eyluloyku/hw-score-predict/assets/116841987/049055c8-8fae-4b76-96b4-3e2467620191)

### Regressor from Text 
The regressors ran from text vectors were ensembled to get combine best features of all. Best of Bag of Words gave 0.19, TF-IDF gave 0.22 and Word2Vec gave 0.21 R2m score, while their ensemble gave much better results.

| Metric | Score          |
|--------|----------------|
| MSE    | 79.9514        |
| R2     | 0.2701         |

But a better result came from using Simulated Annealing from feature selection. It was performed on the concatenation of features and text vector matrices. The best subset gave the result of 0.57 R2score.

### Oversampling

<img width="285" alt="image" src="https://github.com/eyluloyku/hw-score-predict/assets/116841987/f737c259-2e93-442f-a3a5-f15e35d08165">

For classes that we have acquired, using SMOTE, synthetic samples were generated according to the following rule: Use SMOTE for bins that have less instance than the biggest bin. Generate their grades according to their bins’ distributions. Underlying distributions assumed uniform using the visuals. Above there exist a bin’s distribution.

However, predictions using this approach, could not reach a successful result.

<img width="302" alt="image" src="https://github.com/eyluloyku/hw-score-predict/assets/116841987/c364923d-46f5-4b68-a10b-252191e18976">

### Ensemble Best of All

Lastly we ensembled the best of all worlds: results from text regression and simulated annealing on all features. Below is the final result.

| Metric | Score          |
|--------|----------------|
| R2     | 0.5648         |
| MSE    | 48.8635        |

![image](https://github.com/eyluloyku/hw-score-predict/assets/116841987/cd4516fe-bc9c-4d6e-85bc-81c1069e7cf7)

## Conclusions

Even though we achieved relatively high R2 score and low mean squared error in the end, there are some points of improvement. First of all, the dataset is quite imbalanced and better oversampling techniques should be developed to combat this issue. This affects most of the approaches used from clustering to binning. For instance in the bin classification and grade regression approach, we tested that if the bins were predicted correctly, R2 score would be around 0.9. However class imbalance makes it impossible as can be seen from recall and precision of classes even though the prediction accuracy is 64%. Hence, we can conclude that most of the approached used are promising but more data would have brought more value. 

## Contributions of Team Members
We collaborated on all tasks and discussed the approaches together. None of the models are implemented by one person. But you can find below the files that corresponding team member spent more effort.

Eylül: 
- combined.ipynb
- regressor_from_text.ipynb
- softmax_two_class.ipynb

Arda: 
- combined.ipynb
- oversampling.ipynb
- ranking.ipynb
