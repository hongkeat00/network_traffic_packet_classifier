import numpy as np
from flask import Flask, request, render_template
import pickle

# Create an app object using Flask class
app = Flask(__name__)

# Load in the trained model (in pickle file format)
model = pickle.load(open('models/classifier.pkl', 'rb'))

# Define the routes for the app

#use the route() decorator to tell Flask 
# what url should trigger the assiociated defined functions

# route '/' as the homepage of the app
# which redner the html template
@app.route('/')
def home():
    return render_template('index.html')

# route '/predict' as the predict page with
# the predicted output
@app.route('/predict', methods=['POST'])
def predict():
    
    # Convert the string inputs into floating point numbers
    int_features = [float(x) for x in request.form.values()]
    
    # Convert int_features into array form [[a, b]]
    features = [np.array(int_features)]
    
    # begining the prediction using model
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(features)
    features = sc.transform(features)
    prediction = model.predict(features)
    
    # output the predicted output array into string
    for i in prediction:
        output = i
        
    # navigate to render_template with the output string
    return render_template('index.html', prediction_text='The packet is predicted to be originated from {} .'.format(output))

# When Python interpreter defines variable '__name__'
# is equal to '__main__', execute the app
if __name__ == "__main__":
    app.run()
           