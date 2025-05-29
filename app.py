import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and scaler from pickle files
with open('XGB_pkl', 'rb') as model_file:
    pickled_model = pickle.load(model_file)

with open('scaling.pkl', 'rb') as scaler_file:
    scalar = pickle.load(scaler_file)

# Initialize SHAP explainer
explainer = shap.TreeExplainer(pickled_model)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json.get('data')
    if data is None:
        return jsonify({'error': 'No data provided'}), 400

    input_data = np.array(list(data.values())).reshape(1, -1)
    scaled_data = scalar.transform(input_data)
    prediction = pickled_model.predict(scaled_data)[0]

    shap_values = explainer.shap_values(scaled_data)
    shap_values_list = shap_values.tolist()[0]

    return jsonify({
        'prediction': prediction,
        'explanation': shap_values_list
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
    except ValueError:
        return render_template('home.html', prediction_text="Invalid input. Please enter numeric values.")
    
    scaled_data = scalar.transform(np.array(data).reshape(1, -1))
    prediction = pickled_model.predict(scaled_data)[0]
    shap_values = explainer.shap_values(scaled_data)

    # Generate SHAP force plot as image
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[0], matplotlib=True, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    shap_plot_data = base64.b64encode(buf.read()).decode('utf-8')
    shap_plot_uri = f"data:image/png;base64,{shap_plot_data}"

    return render_template("home.html",
                           prediction_text=f"The shear capacity is {prediction:.2f} KN",
                           shap_plot_uri=shap_plot_uri)

if __name__ == "__main__":
    app.run(debug=True)
