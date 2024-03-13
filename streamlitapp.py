import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

model = pickle.load(open('boston.pkl', 'rb'))
encoder_dict = pickle.load(open('scaling.pkl', 'rb')) 
cols=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']    
  
def main(): 
    st.title("Boston House Pricing App")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Boston Housing Price Prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    crim = st.text_input("CRIM - per capita crime rate by town") 
    zn = st.text_input("ZN - proportion of residential land zoned for lots over 25,000 sq.ft") 
    indus = st.text_input("INDUS - proportion of non-retail business acres per town") 
    chas = st.text_input("CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)") 
    nox = st.text_input("NOX - nitric oxides concentration (parts per 10 million)") 
    rm = st.text_input("RM - average number of rooms per dwelling") 
    age = st.text_input("AGE - proportion of owner-occupied units built prior to 1940") 
    dis = st.text_input("DIS - weighted distances to five Boston employment centres") 
    rad = st.text_input("RAD - index of accessibility to radial highways") 
    tax = st.text_input("TAX - full-value property-tax rate per $10,000") 
    ptratio = st.text_input("PTRATIO - pupil-teacher ratio by town") 
    b = st.text_input("B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
    lstat = st.text_input("LSTAT - per lower status of the population") 
    
    if st.button("Predict"): 
        #features = [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX,PTRATIO, B, LSTAT]]
        data = {'crim': float(crim), 'zn': float(zn), 'indus': float(indus), 'chas': float(chas), 'nox': float(nox), 'rm': float(rm), 'age': float(age), 'dis': float(dis), 'rad': float(rad), 'tax': float(tax), 'ptratio': float(ptratio), 'b': float(b), 'stat':float(lstat)}
        print(data)
        df=pd.DataFrame([list(data.values())])
        # category_col =['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
        # for cat in encoder_dict:
        #     for col in df.columns:
        #         le = preprocessing.LabelEncoder()
        #         if cat == col:
        #             le.classes_ = encoder_dict[cat]
        #             for unique_item in df[col].unique():
        #                 if unique_item not in le.classes_:
        #                     df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
        #             df[col] = le.transform(df[col])
        
        df1 = scaler.fit_transform(df)
        #features_list = df1.values.tolist()      
        prediction = model.predict(df1)
    
        output = int(prediction[0])
        st.success('Median value of owner-occupied homes is {:.2f}'.format(output*1000))
      
if __name__=='__main__': 
    main() 
    