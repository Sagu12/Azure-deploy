#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, request, jsonify, render_template
import joblib
from pyforest import *


# In[3]:


app = Flask(__name__)


# In[4]:


model = joblib.load(open('model.pkl', 'rb'))


# In[5]:


@app.route('/')
def home():
    return render_template('index.html')


# In[6]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='flower is {}'.format(output))


# In[ ]:


if __name__ == "__main__":
    app.run(debug=True)

