def plot(criteria):
    survivors_gender = survivors_data.groupby([criteria]).size().values
    non_survivors_gender = non_survivors_data.groupby([criteria]).size().values
    totals = survivors_gender + non_survivors_gender
    data1_percentages = (survivors_gender/totals)*100 
    data2_percentages = (non_survivors_gender/totals)*100
    if(criteria=='Sex'):
        categories = ['Female', 'Male']
    if(criteria=='Pclass'):
        categories=['Lower class','Middle class','Upper class']
    if(criteria=='age_group'):
        categories=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79','80-89']
    if(criteria=='Embarked'):
        categories=['Cherbourg','Queenstown','Southampton']
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    
    # plot chart for count of survivors by class
    ax1.bar(range(len(survivors_gender)), survivors_gender, label='Survivors', alpha=0.5, color='g')
    ax1.bar(range(len(non_survivors_gender)), non_survivors_gender, bottom=survivors_gender, label='Non-Survivors', alpha=0.5, color='r')
    plt.sca(ax1)
    if(criteria=='Sex'):
        plt.xticks([0,1], categories)
    if(criteria=='Pclass'):
        plt.xticks([0,1,2], categories)
    if(criteria=='Embarked'):
        plt.xticks([0,1,2], categories)
    if(criteria=='age_group'):
        tick_spacing = np.array(range(len(categories)))
        plt.xticks(tick_spacing,categories)
    plt.legend(loc='best')
    ax1.set_ylabel("Count")
    ax1.set_xlabel("")
    ax1.set_title("Count of survivors by "+criteria,fontsize=14)
    

    # plot chart for percentage of survivors
    ax2.bar(range(len(data1_percentages)), data1_percentages, alpha=0.5, color='g')
    ax2.bar(range(len(data2_percentages)), data2_percentages, bottom=data1_percentages, alpha=0.5, color='r')
    plt.sca(ax2)
    plt.pause(1)
    if(criteria=='Sex'):
        plt.xticks([0,1], categories)
    if(criteria=='Pclass'):
        plt.xticks([0,1,2], categories)
    if(criteria=='Embarked'):
        plt.xticks([0,1,2], categories)
    if(criteria=='age_group'):
        tick_spacing = np.array(range(len(categories)))
        plt.xticks(tick_spacing,categories)
    ax2.set_ylabel("Percentage")
    ax2.set_xlabel("")
    ax2.set_title("% of survivors by "+criteria,fontsize=14)


def two():
    male_data = tdf[tdf.Sex == "male"].groupby('age_group').Survived.mean().values
    female_data = tdf[tdf.Sex == "female"].groupby('age_group').Survived.mean().values
    categories=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79','80-89']
    f,ax = plt.subplots(1,1,figsize=(10,5))
    male_plt_position = np.array(range(len(categories)))
    female_plt_position = np.array(range(len(categories)))+0.4
    ax.bar(male_plt_position, male_data,width=0.4,label='Male',color='b')
    ax.bar(female_plt_position, female_data,width=0.4,label='Female',color='r')
    plt.sca(ax)
    plt.xticks(np.array(range(len(categories)))+0.2 ,categories)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Age Group")
    ax.set_title("Proportion of survivors by age group / Gender",fontsize=14)
    plt.legend(loc='best')
    

def process_data(df):
    new_df=df.copy()
    le=preprocessing.LabelEncoder()
    new_df['Sex']=le.fit_transform(new_df['Sex'])
    new_df['Pclass']=le.fit_transform(new_df['Pclass'])
    new_df['Embarked']=le.fit_transform(new_df['Embarked'])
    new_df=new_df.drop(['SibSp','Parch','FamilySize','age_group'],axis=1)
    return new_df

def removal_NA(row):
    
    if pd.isnull(row['Age']):
        return average_ages[row['Sex'],row['Pclass']]
    else:
        return row['Age']



def map_data(df):
    df['Age'] =df.apply(removal_NA, axis=1)
    df['Embarked'].fillna('S',inplace=True)
    df=df.drop(['Cabin','Name','Ticket','PassengerId','Fare',],axis=1)
    survived_map={0:False,1:True}
    df['Survived']= df['Survived'].map(survived_map)
    pclass_map = {1: 'Upper Class', 2: 'Middle Class', 3: 'Lower Class'}
    df['Pclass'] = df['Pclass'].map(pclass_map)
    port_map = {'S': 'Southampton', 'C': 'Cherbourg','Q':'Queenstown'}
    df['Embarked'] = df['Embarked'].map(port_map)
    df['FamilySize'] = df['SibSp'] + df['Parch']
    age_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79','80-89']
    df['age_group'] = pd.cut(df['Age'], range(0, 91, 10), right=False, labels=age_labels)
    return df


def authenticate(criteria):
    table = pd.crosstab([tdf['Survived']], tdf[criteria])
    chi2, p, dof, expected = stats.chi2_contingency(table.values)
    print("Chi square value for "+criteria+" is ",chi2)
    print("P value for "+criteria+" is ",p)
    print("\n")


def ML(a,b,c,d):
    X=ndf.drop(['Survived'],axis=1).values
    Y=ndf['Survived'].values
    feature_train,feature_test,label_train,label_test=train_test_split(X,Y,test_size=0.2)
    clf=DecisionTreeClassifier(min_samples_split=100)
    clf.fit(feature_train,label_train)
    pred=clf.predict(feature_test)
    print(clf.predict(np.array([a,b,c,d]).reshape(1,-1)))     
    print("Accuracy Level is : ",accuracy_score(pred,label_test))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 


tdf=pd.read_csv('titanic_data.csv')
average_ages = tdf.groupby(['Sex','Pclass'])['Age'].mean()
tdf=map_data(tdf)
survivors_data = tdf[tdf['Survived']==True]
non_survivors_data = tdf[tdf['Survived']==False]
ndf=process_data(tdf)

while(1):
    print(" ")
    print("1-Preview of the Dataset")
    print("2-Factors Affecting Survival Chances of Passengers")
    print("3-Graphs depicting the count and percentage of survivors for such factors")
    print("4-Proportion of survivors by age group/gender")
    print("5-Predicting the chances of survival of a passenger given the details")
    choice=input("Enter your choice : ")
    if choice=='1':
        print(tdf.head(10))
    elif choice=='2':
        print("Factors are :")
        print("\n")
        print("Passenger's Class")
        authenticate('Pclass')
        print("Passenger's Age")
        authenticate('age_group')
        print("Gender of the Passenger")
        authenticate('Sex')
        print("Embarkation Point")
        authenticate('Embarked')
        print("For more relevance consider graphs also")
    elif choice=='3':
        fact_choice=input("Enter the factor ")
        plot(fact_choice)
    elif choice=='4':
        two()
    elif choice=='5':
        Pcl=int(input("Enter the class of Passenger [Upper class(2) Middle class(1) or Lower class(0)]: "))
        gen=int(input("Enter the Gender of Passenger [Male(1) or Female(0)] : "))
        ag=int(input("Enter the Age of Passenger : "))
        em=int(input("Enter the Embarkation point [Chebourg(0) Queenstown(1) or Southampton(2)]"))
        print(" ")
        ML(Pcl,gen,ag,em)
    else:
        print("Invalid choice")
    print("\n")
    print("do u want to continue(y/n)")
    dep=input()
    if dep=='y' or dep=='Y':
        print("\n")
        continue
    elif dep=='n' or dep=='N':
        break
    else:
        print("Invalid Choice")
        print("\n")















