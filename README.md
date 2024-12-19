# Pregnancy-Risk-Level-Prediction-using-machine-learning
### Abstract:
Maternal health is a critical aspect of healthcare, particularly in rural areas where access to medical facilities and timely interventions can be limited. This project aims to predict maternal health risks using machine learning algorithms, by analyzing and training different ML models including Decision Tree Learning, Random Forests, K-Nearest Neighbors (KNN), Naïve Bayes, Support Vector Machines (SVM), and Artificial Neural Networks (ANN), on a dataset that contains information gathered from multiple health facilities in rural Bangladesh using special internet-connected devices that track health risks. The dataset comprises 1013 instances with six key features: Age, Systolic Blood Pressure (SystolicBP), Diastolic Blood Pressure (DiastolicBP), Blood Sugar (BS), Body Temperature (BodyTemp), and Heart Rate.
Each algorithm was evaluated based on its predictive performance and accuracy. This comparative analysis provides insights into the strengths and weaknesses of each method, helping to identify the most effective approach for early detection and intervention of maternal health risks. Our findings aim to contribute to the broader goal of reducing maternal mortality, aligning with the Sustainable Development Goals (SDGs) set by the United Nations

### Introduction
Maternal health is a pivotal element of global healthcare, critically impacting the well-being of both mothers and their children. Despite remarkable progress, maternal mortality remains a significant concern, particularly in poor and underserved communities, where medical resources, access to critical healthcare services, and robust monitoring systems are very limited. Effectively addressing maternal health risks in these contexts is essential to achieving the Sustainable Development Goals (SDGs) established by the United Nations, particularly the goal of reducing global maternal mortality rates.
The problem addressed here is the complex issue of ensuring effective monitoring and management of maternal health in remote and rural areas, where such efforts are often fraught with difficulties. Traditional healthcare infrastructure often falls short in providing timely and accurate risk assessments, leading to preventable complications and fatalities. The development and implementation of robust risk prediction models are imperative to mitigate these issues. This study aims to analyze the data in depth, and train various machine learning models (Decision Tree, Random Forests, K-Nearest Neighbors, Naïve Bayes, Support Vector Machines, and Artificial Neural Networks) to predict maternal health risks using a dataset collected from rural Bangladesh. By comparing the performance of different models, the project seeks to identify the most effective algorithm for early risk detection. This can ultimately contribute to improved healthcare interventions, better resource allocation, and reduced maternal mortality rates in remote areas.
### Dataset Description
#### Source
The dataset is titled "Maternal Health Risk," is sourced from the UCI Machine Learning Repository and can be accessed at https://archive.ics.uci.edu/dataset/863/maternal+health+risk. It is published by Marzia Ahmed (2020) .
The data has been collected from health facilities in the rural areas of Bangladesh using an IoT-based risk monitoring system.

#### Dataset Characteristics
● Type:: Multivariate

● Subject Area:: Health and MedicineAssociated

● Tasks:: Classification

● Features Type:: Real, Integer

● Instances:: 1013

● Features:: 6


1. Data Preprocessing:

for robust analysis, we conducted this data preprocessing pipeline:

● Data Cleaning: We confirmed the absence of missing values and identified outliers in the Blood Sugar and Body Temperature features, opting to monitor their impact during model training rather than addressing them immediately. Data cleaning rectified inconsistencies and duplicates, removing 50% of the data identified as duplicates while preserving a clean copy for potential future use.

● Feature Engineering: we transformed BodyTemp into a binary attribute named HighFever (0 if equal 98 1 if greater) and encoded the target feature levels into numeric (0 for low risk, 1 for mid risk and 2 for high risk). We also employed data augmentation techniques to address potential biases using SMOTE.

● Data Scaling: We applied standard scaling on a copy of the dataset, to ensure all features were on a comparable scale.
● Data Splitting: We strategically split both datasets (scaled and non scaled) into training(80%) and testing(20%) sets for model development and evaluation.
2. Model Selection and Training:

● Decision Tree & Random Forest: DT is a simple model that creates decision rules by splitting data based on feature values. While RF aime to improve accuracy and reduce overfitting by averaging multiple decision trees. Since scaling did not impact the performance of the models, we chose to train them on the non-scaled dataset. Initially, we fine-tuned hyperparameters via grid search for DT and RF models. Further experiments refined parameters for better performance. We then employed AdaBoost, which iteratively trains models on modified data distributions, focusing on examples where prior models underperformed, here again, we used fine-tuning to find the best ensemble model.

● K-Nearest Neighbors: A distance-based algorithm that classifies data points based on their proximity to their neighbors. We performed a grid search and other fine tuning methods, on both scaled and non scaled datasets, to identify optimal hyperparameters, including the k value. Additionally, we fine-tuned the KNN model using bagging, which is training multiple KNN models on data subsets with replacement and combining their predictions.

● Naive Bayes: A probabilistic classifier based on Bayes' theorem, assuming feature independence. We evaluated the Naïve Bayes model on various forms of the dataset, including the original dataset (with no augmentation or changes), the scaled original dataset, dataset with and without duplicates, dataset with binned features, and dataset with transformed features. Using different fine-tuning techniques, we identified the form of the dataset with which Naïve Bayes performs best. To enhance robustness and potentially reduce variance, we employed ensemble methods, including bagging and boosting. Additionally, we explored calibration techniques to improve the reliability of the model's predicted probabilities, ensuring they better reflect the true class likelihoods.

● SVM : A classifier that finds the optimal hyperplane to separate different classes in the feature space. Similar to Naïve Bayes, we evaluated the SVM model on different forms of the dataset, including the original dataset, the scaled dataset, and datasets with and without duplicates. After identifying the form of the dataset where the model performs best using grid search, we conducted further hyperparameter tuning to find the best parameters. This involved searching for the model with the highest accuracy.

● ANN : A network of interconnected nodes (neurons) organized in layers. Here, we worked with the scaled dataset. First we used one hot encoding for the target feature. Then we did several trials to choose the network architecture and other hyperparameters, activation functions for hidden and output layers, compiler, and regularization technique. We also used different approaches and tests to fine tune batch_size and number of epochs.

3. Models Evaluation:

The models were evaluated using various performance metrics:

● Train and Test Accuracy: We calculated the accuracy on both training and testing sets to assess the general performance of the models and check for overfitting.

● Confusion Matrix: We used confusion matrices on both training and testing sets to visualize the performance of the classification models across different risk levels. The confusion matrix provides a clear picture of true positives, false positives, true negatives, and false negatives.

● Classification Report: We generated detailed classification reports on bothtraining and testing sets, including precision, recall, F1-score, and support for each class. Additionally, we reported macro and weighted averages along with overall accuracy to provide a comprehensive view of model performance.

● Comparative Analysis: We used different graphs including Bar plots for different accuracies, Receiver Operating Characteristic (ROC) and Precision-Recall curve to plot and compare the effectiveness of the five classification algorithms in classifying the target variable with three levels.

### Results and Analysis
1. Decision Tree:
The initial grid search achieved a best score of 0.838798 but with a high max_depth.To improve, we fixed two combinations with different max_features and traced accuracy vs. depth, the optimal model yielded 0.9137577 training accuracy and 0.852459 testing accuracy Using this model with AdaBoost we traced accuracy vs. n_estimators, the best result achieved 0.93429 training accuracy and 0.89344 testing accuracy.

2. Random Forest:
The initial grid search achieved a best score of 0.81928, but the resulting model has a slightly high n_estimator, so we try to find smaller n_estimator that might give a better score, by tracing the test and train accuracies vs n_estimator, then we trace the accuracies vs depths, the accuracy is 0.9322 for train and 0.8852 for test. Using this model with AdaBoost we traced accuracy vs. n_estimators, the best result achieved 0.935318 training accuracy and 0.901639 testing accuracy.

3. KNN:
For this model, as said earlier, we tried to run a grid search to find the best parameters on scaled and unscaled dataset, we got a best score 0.84089. We then ran various techniques for fine tuning, we ended up with a model with 0.8663 test accuracy and 0.9365 train accuracy


4. Naive bayes:
On the first attempt we ran a grid search on all possible preprocessed data (scaled, no duplicates…) the best score was 0.64835, which is too low. Looking for better results, we tried to bin the features which gave training accuracy of 0.72853 and testing accuracy of 0.72527. Further fine tuning did not improve our result so we will stick with this latter.

5. SVM:
Similarly to NB, we first ran a grid search on all possible preprocessed data (scaled, no duplicates…) the best score was 0.701575. In order to improve it we performed further fine tuning using graphs, the best model we managed to fit gave us 0.72022 as train accuracy and 0.703 as test accuracy, which can be considered slightly low in comparison to the previous models we saw.

6. ANN:
Initially, without data augmentation, our model achieved a high train accuracy of 0.782 but a lower test accuracy of 0.653. Through hyperparameter tuning, we further enhanced the model's performance, significantly improving both train and test accuracies to 0.88 and 0.751, respectively. After implementing data augmentation, the train accuracy went up to 0.90, with the test accuracy increasing to 0.86, reducing overfitting.

### General Analysis & Comparison:

● Generalization: all models generalize well (with variant scores) due to effective preprocessing and representative training data.

● Misclassifications: Across models, misclassification of low-risk instances as mid-risk is observed frequently, which may be acceptable given the priority of identifying high-risk cases, this could be due to small dataset size, algorithm sensitivity …

● Model Robustness: Random Forest with AdaBoost and Artificial Neural Network exhibit robust performance with high accuracy and balanced precision, recall, and F1-scores across all classes.

● Prioritizing High-risk: All models demonstrate reasonable performance in identifying high-risk cases, which is crucial for maternal health risk classification tasks.

### Conclusion

After evaluating our models, it is clear that the Random Forest model provides the highest testing accuracy at 0.901639, making it the best performer for this dataset. This model's ability to achieve perfect precision for the high-risk class confirms its suitability for maternal health risk classification tasks. Other models, such as the Decision Tree, KNN, and ANN, also showed strong performance, making them viable alternatives depending on specific requirements and computational resources.
While models like Naive Bayes and SVM performed less well, particularly in terms of generalization and class-specific metrics, their insights were still valuable in understanding the dataset's complexity.

#### Future Work:

● Advanced Techniques: Exploring more advanced models and further fine-tuning
existing ones could lead to better results.

● Clinical Implementation: Practical considerations such as model interpretability, ease of use, and integration with existing healthcare systems should be evaluated for real-world application.

By adopting these preprocessing steps, we ensured that each model received the data in the format that maximized its performance, leading to the most accurate and reliable predictions formaternal health risk classification. This careful attention to preprocessing and model-specific requirements has resulted in high accuracy, especially for the high-risk class, which is paramount for ensuring maternal health safety.

