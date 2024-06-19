#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


# In[12]:


# Load the trained model
with open("insurance.sav", "rb") as model_file:
    loaded_model = pickle.load(model_file)


# In[ ]:


# Load the scaler
filename = 'minmax_scaler.pkl'
loaded_scaler = pickle.load(open(filename, 'rb'))


# In[14]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
with open("insurance.sav", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Load the scaler
filename = 'minmax_scaler.pkl'
loaded_scaler = pickle.load(open(filename, 'rb'))

def insurance_pred(input_data):
    # Predict using the loaded model
    prediction = loaded_model.predict(input_data)
    print(f"Prediction array: {prediction}")  # Debugging output

    if prediction[0] == 0:
        return 'The Insurance is Cheap'
    elif prediction[0] == 1:
        return 'The Insurance is Expensive'
    else:
        return f'Unexpected prediction value: {prediction[0]}'

def main():
    st.title("Insurance Prediction System")

    # Define background image URL (local file path)
    bg_image = "background.jpg"

    # Set background image using CSS
    st.markdown(
        f"""
        <style>
            .reportview-container {{
                background: url(data:image/jpeg;base64,{bg_image}) !important;
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Input fields
    age = st.text_input("Please enter your age")
    sex = st.selectbox("Select your sex", ["male", "female"])
    bmi = st.text_input("(Body Mass Index) Please enter your BMI.")
    children = st.number_input("Enter the number of children", min_value=0, step=1)
    smoker = st.selectbox("Are you a smoker?", ["yes", "no"])
    region = st.selectbox("Select your region", ["northeast", "northwest", "southeast", "southwest"])

    # Convert inputs to appropriate format
    try:
        age = int(age)
        bmi = float(bmi)
    except ValueError:
        st.error("Please enter valid numeric values for age and BMI.")
        return
    
    # Create a dictionary for the new data
    new_data = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }

    # Convert categorical inputs to numerical values
    sex_mapping = {"male": 0, "female": 1}
    smoker_mapping = {"yes": 1, "no": 0}
    region_mapping = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}

    new_data["sex"] = sex_mapping[new_data["sex"]]
    new_data["smoker"] = smoker_mapping[new_data["smoker"]]
    new_data["region"] = region_mapping[new_data["region"]]

    # Convert new_data to DataFrame and then to numpy array
    input_data = pd.DataFrame([new_data]).values
    
    # Scale the input data using the loaded scaler
    input_data_scaled = loaded_scaler.transform(input_data.reshape(1, -1))

    if st.button("Insurance test result"):
        result = insurance_pred(input_data_scaled)
        if result == "The Insurance is Cheap":
            st.success(result)
        else:
            st.error(result)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:




