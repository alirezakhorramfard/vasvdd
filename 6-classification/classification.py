import pandas as pd
from sklearn.model_selection import StratifiedKFold 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from imblearn.metrics import specificity_score
from sklearn.utils import shuffle



class classification:
   

    def load_data(csv_file):
    
        df2 = pd.read_csv(csv_file)
    
        df_1 = df2[df2['label'] == 1]
        df_0 = df2[df2['label'] == 0]

        n_1 = len(df_1)  
        n_0 = len(df_0)
    
        n_train_1 = int(n_1 * 0.8)
        n_train_0 = int(n_0 * 0.8)
    
        df_train_1 = df_1.sample(n=n_train_1)
        df_train_0 = df_0.sample(n=n_train_0)   
        
        df_test_1 = df_1.drop(df_train_1.index)
        df_test_0 = df_0.drop(df_train_0.index)
        
        df_train_1 = shuffle(df_train_1)
        df_train_0 = shuffle(df_train_0)

        df_test_1 = shuffle(df_test_1) 
        df_test_0 = shuffle(df_test_0)
                
        train = []
        train.append(df_train_1)
        train.append(df_train_0)
        train = pd.concat(train)
        
        test = []  
        test.append(df_test_1)
        test.append(df_test_0)
        test = pd.concat(test)
        
        X_train = train.drop('label', axis=1)
        y_train = train['label']

        X_test = test.drop('label', axis=1)  
        y_test = test['label']
        
        return X_train, y_train, X_test, y_test
    
    def xgboost(X_train, y_train , X_test):
        
        kfold = StratifiedKFold(n_splits=5)
        
        clf = make_pipeline(StandardScaler(), 
                            GradientBoostingClassifier())
                        
        scores = cross_val_score(clf, X_train, y_train, cv=kfold)
        
        print(scores)
        print("std : ",scores.std())
        print("Training cross val score:", scores.mean())
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        return y_pred
    
    def mlp(X_train, y_train,X_test):

        kfold = StratifiedKFold(n_splits=5)

        clf = make_pipeline(StandardScaler(), 
                            MLPClassifier(hidden_layer_sizes=100,activation='relu'))

        scores = cross_val_score(clf, X_train, y_train, cv=kfold)

        
        print(scores)
        print("std : ",scores.std())
        print("Training cross val score:", scores.mean())
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        return y_pred
    
    def random_forest(X_train, y_train,X_test):

        kfold = StratifiedKFold(n_splits=5)

        clf = make_pipeline(StandardScaler(),  
                            RandomForestClassifier(n_estimators=100))

        scores = cross_val_score(clf, X_train, y_train, cv=kfold)

        print(scores)
        print("std : ",scores.std())
        print("Training cross val score:", scores.mean())

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        return y_pred
    def svm(X_train, y_train,X_test):

        kfold = StratifiedKFold(n_splits=5)

        clf = make_pipeline(StandardScaler(),  
                            SVC(gamma='auto' , kernel="rbf"))

        scores = cross_val_score(clf, X_train, y_train, cv=kfold)

        print(scores)
        print("std : ",scores.std())
        print("Training cross val score:", scores.mean())

        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        return y_pred
    def KNN(X_train, y_train):
        
        kfold = StratifiedKFold(n_splits=10)

        clf = make_pipeline(StandardScaler(),  
                            KNeighborsClassifier(n_neighbors=3))

        scores = cross_val_score(clf, X_train, y_train, cv=kfold)

        print(scores)
        print("Training cross val score:", scores.mean())
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        return y_pred
    def vae():
        pass
    

    def accuracy(y_test, y_pred):

        accuracy_all = accuracy_score(y_test, y_pred)
        print(f'Accuracy for label 0 and 1: {accuracy_all}')

    def f1_score(y_test, y_pred):
        f1_allClass=f1_score(y_test,y_pred)
        print(f'Overall F1 score: {f1_allClass}')
    
    def conf_matrix(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm
    def conf_matrix_plot(y_test, y_pred):
        cm = classification.conf_matrix(y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Non-Interaction', 'Intraction'])

        cm_display.plot()
        plt.show()
    def spe(y_test, y_pred):
        cm = classification.conf_matrix(y_test, y_pred)
        TN = cm[0,0]
        FP = cm[0,1]

        specificity = TN / (TN + FP)

        print(f'Specificity(SPE): {specificity}')
    

    def sen(y_test, y_pred):
        cm = classification.conf_matrix(y_test, y_pred)

        TP = cm[1,1] 
        FN = cm[1,0]

        sensitivity = TP / (TP + FN)

        print(f'Sensitivity(SEN): {sensitivity}')


    def aupr(y_test, y_pred):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        aupr = average_precision_score(y_test, y_pred)

        print(f'AUPR: {aupr}')


        plt.step(recall, precision, color='b', alpha=0.2, 
                where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                        color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(aupr))
        plt.show()

    
    def auc_roc(y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        print(f'AUC-ROC: {auc}')
        plt.plot(fpr, tpr, color='orange', label='ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()


path = "DTI\\5-preparation-label\\df_NR_ADEFG\\df_NR_ADEFG_1_5.csv"


X_train, y_train, X_test, y_test = classification.load_data(path)

y_pred=classification.xgboost(X_train, y_train, X_test)
classification.accuracy(y_test, y_pred)
classification.f1_score(y_test,y_pred)
classification.conf_matrix(y_test,y_pred)
classification.conf_matrix_plot(y_test, y_pred)
classification.spe(y_test,y_pred)
classification.sen(y_test, y_pred)
classification.aupr(y_test, y_pred)
classification.auc_roc(y_test, y_pred)
