
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
import warnings
import pickle
warnings.filterwarnings("ignore")

'#Read File'
supply_chain_dataset= pd.read_csv('Processed.csv')

'# Data Preparation'
supply_chain_dataset = supply_chain_dataset.drop('DeliveryStatus',axis=1)
label_map = {'PackageStatus': {'DELIVERED': 0, 'DISPATCHED': 0, 'RETURN_ACKNOWLEDGED': 1,'RETURNED': 1}}
supply_chain_dataset = supply_chain_dataset.replace(label_map)
supply_chain_dataset_processed = supply_chain_dataset.copy()
supply_chain_dataset_processed = supply_chain_dataset_processed.drop(['FA_Time', 'order_delivery_date','dispatch_date'], axis=1)

label_encoder = LabelEncoder()
le = LabelEncoder()


supply_chain_dataset_processed['PackageID'] = le.fit_transform(supply_chain_dataset_processed['PackageID'])
supply_chain_dataset_processed['Category'] = le.fit_transform(supply_chain_dataset_processed['Category'])
supply_chain_dataset_processed['District'] = le.fit_transform(supply_chain_dataset_processed['District'])
supply_chain_dataset_processed['Taluka'] = le.fit_transform(supply_chain_dataset_processed['Taluka'])
supply_chain_dataset_processed['Shipper'] = le.fit_transform(supply_chain_dataset_processed['Shipper'])
supply_chain_dataset_processed['Source'] = le.fit_transform(supply_chain_dataset_processed['Source'])

supply_chain_dataset_processed = pd.get_dummies(supply_chain_dataset_processed, prefix=['State'], columns=['State'])

'#Data Splitting'
X = supply_chain_dataset_processed.drop('PackageStatus',axis=1) 
Y = supply_chain_dataset_processed['PackageStatus']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=7)


'#Naive Bayes'
model = CategoricalNB()
print(model.__class__)
model.fit(X_train,Y_train)
'#Y_predict = model.predict(X_test)'

pickle.dump(model, open('model.pk1', 'wb'))
model = pickle.load(open('model.pk1', 'rb'))
