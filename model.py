import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.set_style('whitegrid')
sns.set_palette('dark')
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("UCI_Credit_Card.csv",index_col='ID')
data.drop(['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','default.payment.next.month'],axis=1).describe()
data.columns

columns=['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','default.payment.next.month']
#checking the categorical values 
for col in columns:
    values=data[col].value_counts()
    print("The number categorical values in {} are\n{} " .format(col,values))

#Grouping categories 4:others, 5:unknown, and 6:unknown to a single class '4' in Education variable
data['EDUCATION'].replace([0,5,6],4,inplace=True)

#Grouping category 0:unlabeled to single class '3'
data['MARRIAGE'].replace(0,4,inplace=True)

#Renaming label column
data.rename(columns={'default.payment.next.month':'DEFAULT'},inplace=True)    


#replacing the integers with the names in the categorical variables
#SEX variable
sex={'SEX':{1:"male",2:'female'}}
data.replace(sex,inplace=True)

#EDUCATION variable
education={'EDUCATION':{1 :'graduate school',2 : 'university',3 : 'high school',4 : 'others'}}
data.replace(education,inplace=True)

#MARRIEGE variable
marriage={'MARRIAGE':{1:"married", 2:"single", 3:"others"}}
data.replace(marriage,inplace=True)


#Binning AGE variable(Grouping the data into specific range/Categories)
data['AGE']=pd.cut(data['AGE'],6,labels=['20-30','30-40','40-50','50-60','60-70','70-80'])
data['AGE'].value_counts()

#dividing the dataframe into label and independent variables
label=data['DEFAULT']
predData=data.drop('DEFAULT',axis=1)

##Creating Dummy Variables
dummyData = pd.get_dummies(predData[['SEX',"AGE",'EDUCATION', 'MARRIAGE']],drop_first=False)
print(dummyData)

#creating a new dataframe by dropping 'SEX', 'EDUCATION', 'MARRIAGE' from the old dataframe and concating it with the dataframe containing dummy variables
finalData=pd.concat([dummyData,predData.drop(['SEX', 'EDUCATION', 'MARRIAGE','AGE'],axis=1)],axis=1)
finalData.head()

# Am going to Normalize my data 
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
normData=scaler.fit(finalData)
normData=scaler.transform(finalData)
#cobverting the results to Pandas dataframe
newData=pd.DataFrame(normData,columns=finalData.columns)
newData.head()


#split the data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(newData,label,test_size=0.3,random_state=11)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#importing the model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

pred=model.predict(x_test)

#Evaluatin the model
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,pred))

print(classification_report(y_test,pred))

# importing necessary library for
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, pred)  
print('The AUC of the model is: %.2f' % auc)


#Saving the model: Serialization
from sklearn.externals import joblib
import joblib
joblib.dump(model,'model.pkl')
lr=joblib.load('model.pkl')
