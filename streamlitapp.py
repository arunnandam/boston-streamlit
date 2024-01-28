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
    <h2 style="color:white;text-align:center;">Income Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    crim = st.text_input("CRIM",0) 
    zn = st.text_input("ZN",0) 
    indus = st.text_input("INDUS",0) 
    chas = st.text_input("CHAS",0) 
    nox = st.text_input("NOX",0) 
    rm = st.text_input("RM",0) 
    age = st.text_input("AGE",0) 
    dis = st.text_input("DIS",0) 
    rad = st.text_input("RAD",0) 
    tax = st.text_input("TAX",0) 
    ptratio = st.text_input("PTRATIO",0) 
    b = st.text_input("B",0)
    lstat = st.text_input("LSTAT",0) 
    
    if st.button("Predict"): 
        #features = [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX,PTRATIO, B, LSTAT]]
        data = {'crim': int(crim), 'zn': int(zn), 'indus': int(indus), 'chas': int(chas), 'nox': int(nox), 'rm': int(rm), 'age': int(age), 'dis': int(dis), 'rad': int(rad), 'tax': int(tax), 'ptratio': int(ptratio), 'b': int(b), 'stat':int(lstat)}
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
        st.success('Price is {}'.format(output))
      
if __name__=='__main__': 
    main() 
    