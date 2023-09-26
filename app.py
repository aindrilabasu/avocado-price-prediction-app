from datetime import datetime
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

model = pickle.load(open('avocado_regression_model.pkl', 'rb'))
scaler = pickle.load(open('scaler_file.pkl', 'rb'))
le1 = LabelEncoder()
le2 = LabelEncoder()
type_labels = ['conventional', 'organic']
region_labels = ['Albany', 'Atlanta', 'BaltimoreWashington', 'Boise', 'Boston', 'BuffaloRochester', 'California',
                 'Charlotte', 'Chicago', 'CincinnatiDayton', 'Columbus', 'DallasFtWorth', 'Denver', 'Detroit',
                 'GrandRapids', 'GreatLakes', 'HarrisburgScranton', 'HartfordSpringfield', 'Houston', 'Indianapolis',
                 'Jacksonville', 'LasVegas', 'LosAngeles', 'Louisville', 'MiamiFtLauderdale', 'Midsouth', 'Nashville',
                 'NewOrleansMobile', 'NewYork', 'Northeast', 'NorthernNewEngland', 'NorthernNewEngland', 'Philadelphia',
                 'PhoenixTucson', 'Pittsburgh', 'Plains', 'Portland', 'RaleighGreensboro', 'RichmondNorfolk',
                 'Roanoke', 'Sacramento', 'SanDiego', 'SanFrancisco', 'Seattle', 'SouthCarolina', 'SouthCentral',
                 'Southeast', 'Spokane', 'StLouis', 'Syracuse', 'Tampa', 'TotalUS', 'West', 'WestTexNewMexico']
le1.fit(type_labels)
le2.fit(region_labels)


@app.route('/')
def home():
    return render_template('index.html', avg_price=-1)


@app.route('/predict', methods=['POST'])
def predict():
    date = datetime.strptime(request.form.get('Date'), "%Y-%m-%d").date()
    num_4046 = float(request.form.get('4046'))
    num_4225 = float(request.form.get('4225'))
    num_4770 = float(request.form.get('4770'))
    small_bags = float(request.form.get('Small Bags'))
    large_bags = float(request.form.get('Large Bags'))
    xlarge_bags = float(request.form.get('XLarge Bags'))
    av_type = request.form.get('type')
    year = date.year
    region = request.form.get('region')
    month = date.month
    day = date.day

    # Converting the data as per model requirements
    transformed_num_4046 = stats.yeojohnson(np.array([num_4046]), lmbda=0.05)[0]
    transformed_num_4225 = stats.yeojohnson(np.array([num_4225]), lmbda=0.05)[0]
    transformed_num_4770 = stats.yeojohnson(np.array([num_4770]), lmbda=0.05)[0]
    transformed_small_bags = stats.yeojohnson(np.array([small_bags]), lmbda=0.05)[0]
    transformed_large_bags = stats.yeojohnson(np.array([large_bags]), lmbda=0.05)[0]
    transformed_xlarge_bags = stats.yeojohnson(np.array([xlarge_bags]), lmbda=0.05)[0]

    # Encoding categorical variables
    encoded_type = le1.transform([av_type])[0]
    encoded_region = le2.transform([region])[0]

    input_variables = np.array([10000, transformed_num_4046, transformed_num_4225, transformed_num_4770, 10000,
                                transformed_small_bags, transformed_large_bags, transformed_xlarge_bags,
                                encoded_type, year, encoded_region, month, day])

    # Scaling the input variables
    scaled_input = scaler.transform(input_variables.reshape(1, -1))
    scaled_input_list = scaled_input.tolist()[0]
    del scaled_input_list[0]
    del scaled_input_list[3]

    # Making the prediction
    prediction = model.predict(np.array([scaled_input_list]))
    return render_template('index.html', avg_price=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
