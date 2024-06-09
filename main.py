import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, learning_curve, validation_curve


class ModelRunner:
    def __init__(self, data, label, model):
        self.data = data
        self.label = label
        self.model = model
        self.CV = 5

    def train(self, title, filename):
        # Start with learning curve
        sizes, training_scores, testing_scores = learning_curve(self.model, self.data,
                                                                self.label, cv=self.CV,
                                                                scoring='accuracy')

        self.learning_curve(sizes, training_scores, testing_scores,
                            title,
                            "Training Set Size",
                            "Accuracy Score",
                            filename)

    def validation(self, param_name, param_range, filename):
        train_score, test_score = validation_curve(self.model, self.data, self.label,
                                                   param_name=param_name,
                                                   param_range=param_range,
                                                   cv=self.CV, scoring='accuracy')

        if 'layer_sizes' in filename:
            self.validation_curve(train_score, test_score,
                                  f"Validation Curve with for {param_name}",
                                  f'Different {param_name}',
                                  'Accuracy', [583 * 10, 583 * 20, 583 * 30, 583 * 40, 583 * 50], filename)
        else:
            self.validation_curve(train_score, test_score,
                                  f"Validation Curve with for {param_name}",
                                  f'Different {param_name}',
                                  'Accuracy', param_range, filename)

    def train_nn(self, filename):
        data_train, data_test, label_train, label_test = train_test_split(self.data, self.label, test_size=0.2)
        self.model.fit(data_train, label_train)

        plt.figure()
        plt.plot(self.model.loss_curve_, label='Training')
        plt.xlabel('Epochs')
        plt.ylabel("Error")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f'outputs/{filename}.jpg')
        plt.close()

    @staticmethod
    def learning_curve(sizes, train, test, Title, xlabel, ylabel, filename):
        # Mean and Standard Deviation of training scores
        mean_training = np.mean(train, axis=1)

        # Mean and Standard Deviation of testing scores
        mean_testing = np.mean(test, axis=1)

        # dotted blue line is for training scores and green line is for cross-validation score
        plt.plot(sizes, mean_training, '--', color="b", label="Training score")
        plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

        # Drawing plot
        plt.title(Title)
        plt.xlabel(xlabel), plt.ylabel(ylabel), plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(f'outputs/{filename}.jpg')
        plt.close()

    @staticmethod
    def validation_curve(train, test, Title, xlabel, ylabel, parameter, filename):
        # Calculating mean of training score
        mean_train_score = np.mean(train, axis=1)

        # Calculating mean of testing score
        mean_test_score = np.mean(test, axis=1)

        # Plot mean accuracy scores for training and testing scores
        plt.plot(parameter, mean_train_score,
                 label="Training Score", color='b')
        plt.plot(parameter, mean_test_score,
                 label="Cross Validation Score", color='g')

        # Creating the plot
        plt.title(Title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.legend(loc='best')
        plt.savefig(f'outputs/{filename}.jpg')
        plt.close()


if __name__ == '__main__':
    # Loading Diabetes dataset
    diabetes = pd.read_csv('./dataset/diabetes_prediction_dataset.csv')

    # Preprocessing Diabetes dataset
    diabetes['gender'] = diabetes['gender'].map({'Female': 1, 'Male': 0})
    diabetes = diabetes.drop('smoking_history', axis=1)
    diabetes = diabetes.dropna()

    # Split datasets to features and classes
    features = diabetes.iloc[:, 0:-1]
    classes = diabetes.iloc[:, -1]

    print('Creating KNN Model')
    model = ModelRunner(features, classes, KNeighborsClassifier())

    print('Training Curve')
    model.train('Learning Curve for KNN on Diabetes Dataset','Diabetes_Prediction_KNN_Learning_Curve')

    print('Validation Curve for Leaf Size')
    model.validation('leaf_size', range(30, 80, 5), 'Diabetes_prediction_different_leaf_size')

    print('Validation Curve for K')
    model.validation('n_neighbors', range(2, 22, 2), 'Diabetes_prediction_different_k_values')

    print('Creating SVM Model')
    model = ModelRunner(features, classes, SVC())

    print('Training Curve')
    model.train('Learning Curve for SVM on Diabetes Dataset', 'Diabetes_Prediction_SVM_Learning_Curve')

    print('Validation Curve for Regularization')
    model.validation('C', [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], 'Diabetes_prediction_different_C_values')

    print('Validation Curve for Kernels')
    model.validation('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], 'Diabetes_prediction_different_kernels')

    print('Creating Neural Network Model')
    model = ModelRunner(features, classes, MLPClassifier(solver='adam', max_iter=200, early_stopping=True))

    print('Epoch Curve')
    model.train_nn('Diabetes_prediction_epochs')

    print('Training Curve')
    model.train('Learning Curve for NN on Diabetes Dataset', 'Diabetes_Prediction_NN_Learning_Curve')

    print('Validation Curve for Layer Sizes')
    model.validation('hidden_layer_sizes', [(583, 10), (583, 20), (583, 30), (583, 40), (583, 50)],
                     'Diabetes_prediction_different_layer_sizes')

    print('Validation Curve for Alpha')
    model.validation('alpha', [0.1, 0.01, 0.001, 0.0001], 'Diabetes_prediction_different_alpha')

    # Loading Diabetes dataset
    heart = pd.read_csv('./dataset/heart_attack_prediction_dataset.csv.csv')

    # Preprocessing Diabetes dataset
    heart = heart.dropna()

    # Split datasets to features and classes
    features = heart.iloc[:, 1:]
    classes = heart.iloc[:, 0]

    print('Creating KNN Model')
    model = ModelRunner(features, classes, KNeighborsClassifier())

    print('Training Curve')
    model.train('Learning Curve for KNN on Heart Dataset', 'Heart_Attack_Prediction_KNN_Learning_Curve')

    print('Validation Curve for Leaf Size')
    model.validation('leaf_size', range(30, 80, 5), 'Heart_Attack_prediction_different_leaf_size')

    print('Validation Curve for K')
    model.validation('n_neighbors', range(2, 22, 2), 'Heart_Attack_prediction_different_k_values')

    print('Creating SVM Model')
    model = ModelRunner(features, classes, SVC())

    print('Training Curve')
    model.train('Learning Curve for SVM on Heart Dataset', 'Heart_Attack_Prediction_SVM_Learning_Curve')

    print('Validation Curve for Regularization')
    model.validation('C', [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], 'Heart_Attack_prediction_different_C_values')

    print('Validation Curve for Kernels')
    model.validation('kernel', ['linear', 'poly', 'rbf', 'sigmoid'], 'Heart_Attack_prediction_different_kernels')

    print('Creating Neural Network Model')
    model = ModelRunner(features, classes, MLPClassifier(solver='adam', max_iter=200, early_stopping=True))

    print('Epoch Curve')
    model.train_nn('Heart_Attack_prediction_epochs')

    print('Training Curve')
    model.train('Learning Curve for NN on Heart Dataset', 'Heart_Attack_Prediction_NN_Learning_Curve')

    print('Validation Curve for Layer Sizes')
    model.validation('hidden_layer_sizes', [(583, 10), (583, 20), (583, 30), (583, 40), (583, 50)],
                     'Heart_Attack_prediction_different_layer_sizes')

    print('Validation Curve for Alpha')
    model.validation('alpha', [0.1, 0.01, 0.001, 0.0001], 'Heart_Attack_prediction_different_alpha')

    print('Done')
