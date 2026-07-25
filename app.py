
import pickle
from typing import Collection
from flask import Flask, Response, abort, jsonify,render_template, redirect, request, url_for,send_file
import numpy as np
import urllib3
from urllib.parse import urlencode
import json
import base64
import pandas as pd
import sklearn 
from google import genai
from dotenv import load_dotenv
import os
import textwrap
import requests


app = Flask(__name__)



@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/mainpages/weather')
def weather():
    return render_template('/mainpages/weather.html')


@app.route('/mainpages/crop_predict')
def crop_predict():
    return render_template('mainpages/crop_predict.html')


@app.route('/mainpages/fertilizer')
def fertilizer():
    return render_template('mainpages/fertilizer.html')


#weather    _____________________________________________________________
def get_icon_class(icon_code):
    icon_mapping = {
        '01d': 'wu-sunny',
        '01n': 'wu-clear',
        '02d': 'wu-partlycloudy',
        '02n': 'wu-partlycloudy',
        '03d': 'wu-cloudy',
        '03n': 'wu-cloudy',
        '04d': 'wu-mostlycloudy',
        '04n': 'wu-mostlycloudy',
        '09d': 'wu-rain',
        '09n': 'wu-rain',
        '10d': 'wu-rain',
        '10n': 'wu-rain',
        '11d': 'wu-tstorms',
        '11n': 'wu-tstorms',
        '13d': 'wu-snow',
        '13n': 'wu-snow',
        '50d': 'wu-fog',
        '50n': 'wu-fog',
    }
    return icon_mapping.get(icon_code, 'wu-unknown')

@app.route('/forecast', methods=['GET'])
def get_weather():
    city = request.args.get('city')
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    pincode = request.args.get('pincode')

    if city:
        data = {
            'q': city,
            'appid': '34fce2119d382f143879ab0c444cf31d',
            'units': 'metric'
        }
    elif lat and lon:
        data = {
            'lat': lat,
            'lon': lon,
            'appid': '34fce2119d382f143879ab0c444cf31d',
            'units': 'metric'
        }
    elif pincode:
        data = {
            'zip': pincode,
            'appid': '34fce2119d382f143879ab0c444cf31d',
            'units': 'metric'
        }
    else:
        abort(400, 'Missing argument city, coordinates, or postal code')

    url_values = urlencode(data)
    url = 'http://api.openweathermap.org/data/2.5/forecast'
    full_url = f'{url}?{url_values}'
    
    http = urllib3.PoolManager()
    response = http.request('GET', full_url)

    if response.status != 200:
        abort(response.status, 'Error fetching weather data')
    
    weather_data = json.loads(response.data.decode('utf8'))
    
    return render_template('/mainpages/weather.html', title='Weather App', data=weather_data, get_icon_class=get_icon_class)



#________________________________________________________________________________________________________
#_________________________________________________________________________________________________________
with open('models/RandomForest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

crop_images = {
    'rice': 'images/Rice.jpg',
    'maize': 'images/Maize.jpg',
    'chickpea': 'images/chickpea.jpg',
    'Kidneybeans': 'images/Kidneybeans',
    'Chickpea': 'images/Chickpea',
    'Pigeonpeas': 'images/Pigeonpeas',
    'Mothbeans': 'images/Mothbeans',
    'Mungbean': 'images/Mungbean',
    'Blackgram': 'images/Blackgram',
    'Lentil': 'images/Lentil',
    'Pomegranate': 'images/Pomegranate',
    'Banana': 'images/Banana',
    'Mango': 'images/Mango',
    'Grapes': 'images/Grapes',
    'Watermelon': 'images/Watermelon',
    'Muskmelon': 'images/Muskmelon',
    'Apple': 'images/Apple',
    'Orange': 'images/Orange',
    'Papaya': 'images/Papaya',
    'Coconut': 'images/Coconut',
    'Cotton': 'images/Cotton',
    'Jute': 'images/Jute',
    'Coffee': 'images/Coffee.jpg'
    # Add paths for other crops here
}

# Load crop names
crop_names = ['Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas', 'Mothbeans',
              'Mungbean', 'Blackgram', 'Lentil', 'Pomegranate', 'Banana', 'Mango', 'Grapes',
              'Watermelon', 'Muskmelon', 'Apple', 'Orange', 'Papaya', 'Coconut', 'Cotton',
              'Jute', 'Coffee']

# Suggestions for each crop
crop_suggestions = {
    'Rice': ["- Rice requires plenty of water, so ensure proper irrigation.",
             "- Keep the fields flooded during the growing season.",
             "- Maintain the pH level of the soil between 5.5 to 6.5."],
    'Maize': ["- Maize grows well in fertile, well-drained soils with plenty of sunlight.",
              "- Ensure proper spacing between plants for optimal growth.",
              "- Provide nitrogen-rich fertilizers for better yield."],
    'Chickpea': ["- Chickpeas thrive in well-drained soils with a pH between 6.0 to 7.0.",
                 "- Avoid waterlogging, as chickpeas are susceptible to root rot.",
                 "- Inoculate seeds with nitrogen-fixing bacteria for improved nitrogen availability."],
      'Kidneybeans': ["- Plant kidney beans in well-drained soil with full sun exposure.",
                "- Provide support such as trellises or stakes for the plants to climb.",
                "- Keep the soil consistently moist but not waterlogged."],


    'Pigeonpeas': ["- Grow pigeon peas in loamy, well-drained soil with full sun.",
               "- Water the plants regularly, especially during flowering and pod formation.",
               "- Control weeds and pests to ensure optimal growth and yield."],


    'Mothbeans': ["- Moth beans prefer sandy, well-drained soil and warm temperatures.",
              "- Provide support for the plants to climb as they grow.",
              "- Avoid overwatering, as moth beans are sensitive to waterlogging."],


    'Mungbean': ["- Mung beans thrive in well-drained, sandy loam soil with a pH of 6.0 to 7.2.",
             "- Plant in full sun and water regularly, especially during flowering and pod development.",
             "- Use organic mulch to retain soil moisture and suppress weeds."],

    'Blackgram': ["- Black gram requires well-drained loamy soil and warm temperatures for optimal growth.",
               "- Provide support for the plants to climb as they grow.",
               "- Ensure adequate irrigation, especially during flowering and pod development."],


    'Lentil': ["- Lentils grow best in well-drained sandy loam soil with a pH of 6.0 to 7.5.",
           "- Plant in full sun and water regularly, especially during flowering and pod development.",
           "- Inoculate seeds with nitrogen-fixing bacteria to improve soil fertility."],


    'Pomegranate': ["- Pomegranates thrive in well-drained soil with a pH of 5.5 to 7.0.",
                "- Plant in full sun and provide regular irrigation, especially during fruit development.",
                "- Prune the trees regularly to promote airflow and reduce disease incidence."],


    'Banana': ["- Bananas prefer well-drained, fertile soil with a pH of 5.5 to 6.5.",
           "- Plant in full sun and provide ample water, especially during the growing season.",
           "- Mulch around the plants to conserve moisture and suppress weeds."],


    'Mango': ["- Mango trees require well-drained soil with good water retention.",
          "- Plant in full sun and provide regular watering, especially during flowering and fruiting.",
          "- Prune the trees to remove dead or diseased branches and promote air circulation."],


    'Grapes': ["- Grapes thrive in well-drained soil with good sunlight exposure.",
           "- Provide support such as trellises for the vines to climb.",
           "- Prune the vines regularly to promote air circulation and improve fruit quality."],

   
    'Watermelon': ["- Watermelons require well-drained, sandy loam soil and full sun.",
                "- Provide ample water during fruit development, but avoid overwatering.",
                "- Mulch around the plants to retain soil moisture and suppress weeds."],


    'Muskmelon': ["- Muskmelons prefer well-drained, sandy loam soil and full sun.",
                "- Provide regular watering, especially during fruit development.",
                "- Mulch around the plants to retain soil moisture and suppress weeds."],

 
    'Apple': ["- Apples require well-drained soil with a pH of 6.0 to 7.0.",
            "- Plant in full sun and provide regular pruning to shape the tree and promote airflow.",
            "- Apply organic mulch around the base of the tree to retain soil moisture and suppress weeds."],


    'Orange': ["- Oranges prefer well-drained, sandy loam soil and full sun.",
            "- Provide regular watering, especially during fruit development.",
            "- Apply organic mulch around the base of the tree to retain soil moisture and suppress weeds."],

 
    'Papaya': ["- Papayas prefer well-drained soil with good water retention.",
            "- Plant in full sun and provide regular watering, especially during fruit development.",
            "- Apply balanced fertilizer regularly to promote healthy growth and fruit production."],


    'Coconut': ["- Coconuts thrive in well-drained, sandy soil with good water retention.",
                "- Plant in full sun and provide regular watering, especially during the dry season.",
                "- Apply organic mulch around the base of the tree to retain soil moisture and suppress weeds."],

   
    'Cotton': ["- Cotton prefers well-drained, loamy soil with good water retention.",
            "- Plant in full sun and provide regular irrigation, especially during flowering and boll development.",
            "- Control weeds and pests to ensure optimal growth and yield."],

   
    'Jute': ["- Jute grows best in well-drained, fertile soil with full sun exposure.",
            "- Provide regular watering, especially during the growing season.",
            "- Harvest jute fibers before the plants start flowering for best quality."],

  
    'Coffee': ["- Coffee plants prefer well-drained, acidic soil with good water retention.",
            "- Plant in partial shade and provide regular watering, especially during dry periods.",
            "- Mulch around the plants to retain soil moisture and suppress weeds."],
}

# Load fertilizer recommendations
fertilizer_dic = {
    'NHigh': """<center><h2>Apply nitrogen-rich fertilizer in moderation.</h3><br>
               <h3>Suggestions:</h3></center>
               <ol>
                   <li><i>Manure:</i> Adding Manure is one of the simplest ways to amend your soil with nitrogen. <br>Be careful as there are various types of manures with varying degrees of nitrogen.</li>
                   <br> <li><i>Coffee grinds:</i> Use your morning addiction to feed your gardening habit! Coffee grounds in soil will be fed with nitrogen. An added benefit to including coffee grounds to your soil is while it will compost, it will also help provide increased drainage to your soil.</li>
                   <br> <li><i>Plant nitrogen fixing plants:</i> Planting vegetables that are in Fabaceae family like peas, beans, and soybeans have the ability to increase nitrogen in your soil.</li>
                   <br><li><i>Plant ‘green manure’ crops:</i> Like cabbage, corn, and broccoli.</li>
                   <br><li><i>Use mulch (wet grass) while growing crops:</i> Mulch can also include sawdust and scrap softwoods.</li>
               </ol>""",
    'PHigh': """<center><h2>Apply phosphorus-rich fertilizer in moderation.</h2><br>
                <h3>Suggestions:</h2><center>
                <ol>
                    <li><i>Avoid adding manure:</i> Manure contains high levels of phosphorus.</li>
                    <br><li><i>Use only phosphorus-free fertilizer:</i> Find a fertilizer with no phosphorus.</li>
                    <br><li><i>Water your soil:</i> Soaking your soil liberally will aid in driving phosphorus out of the soil.</li>
                </ol>""",
    'KHigh': """<center><h2>Apply potassium-rich fertilizer in moderation.</h2><br>
                <h3>Suggestions:</h3></center>
                <ol>
                    <li><i>Loosen the soil:</i> Deeply with a shovel, and water thoroughly to dissolve water-soluble potassium.<br/> Allow the soil to fully dry, and repeat digging and watering the soil two or three more times.</li>
                    <br><li><i>Sift through the soil:</i> And remove as many rocks as possible, using a soil sifter. Minerals occurring in rocks such as mica and feldspar slowly release potassium into the soil slowly through weathering.</li>
                    <br><li><i>Stop applying potassium-rich commercial fertilizer:</i> Apply only commercial fertilizer that has a '0' in the final number field. Commercial fertilizers use a three-number system for measuring levels of nitrogen, phosphorus, and potassium. The last number stands for potassium.</li>
                    <br><li><i>Mix crushed eggshells, crushed seashells, wood ash, or soft rock phosphate into the soil:</i> To add calcium. Mix in up to 10 percent of organic compost to help amend and balance the soil.</li>
                </ol>""",
    'Nlow': """<center><h2>Increase nitrogen in soil.</h2><br>
               <h3>Suggestions:</h3></center>
               <ol>
                   <li><i>Add Sawdust or fine woodchips to your soil:</i> The carbon in the sawdust/woodchips loves nitrogen and will help absorb and soak up any excess nitrogen.</li>
                   <br><li><i>Plant heavy nitrogen-feeding plants:</i> Like tomatoes, corn, broccoli, cabbage, and spinach.</li>
                   <br><li><i>Water your soil:</i> Soaking your soil with water will help leach the nitrogen deeper into your soil, effectively leaving less for your plants to use.</li>
                   <br><li><i>Sugar:</i> In limited studies, it was shown that adding sugar to your soil can help potentially reduce the amount of nitrogen in your soil. Sugar is partially composed of carbon, an element that soaks up nitrogen in the soil.</li>
                   <br><li><i>Add composted manure to the soil.</i></li>
                   <br><li><i>Plant Nitrogen-fixing plants:</i> Like peas or beans.</li>
                   <br><li><i>Use MPK fertilizers with high N value.</i></li>
                   <br><li><i>Do nothing:</i> While it may seem counter-intuitive, avoid over-fertilization as it can lead to nutrient imbalances, environmental pollution, and long-term soil degradation.</li>
               </ol>""",
    'Plow': """<center><h2>Increase phosphorus in soil.</h2><br>
               <h3>Suggestions:</h3></center>
               <ol>
                   <li><i>Bone meal:</i> A fast-acting source rich in phosphorus.</li>
                   <br><li><i>Rock phosphate:</i> A slower-acting source that needs to be converted by the soil.</li>
                   <br><li><i>Phosphorus fertilizers:</i> Apply fertilizer with a high phosphorus content.</li>
                   <br><li><i>Organic compost:</i> Adding quality organic compost to your soil will increase phosphorus content.</li>
                   <br><li><i>Manure:</i> Manure can be an excellent source of phosphorus for your plants.</li>
                   <br><li><i>Clay soil:</i> Introducing clay particles into your soil can help retain phosphorus.</li>
                   <br>li><i>Ensure proper soil pH:</i> Maintain a pH in the 6.0 to 7.0 range for optimal phosphorus uptake.</li>
               </ol>""",
    'Klow': """<center><h2>Increase potassium in soil.</h2><br>
               <h3>Suggestions:</h3></center>
               <ol>
                   <li><i>Mix in muricate of potash or sulphate of potash.</i></li>
                   <br><li><i>Try kelp meal or seaweed.</i></li>
                   <br><li><i>Try Sul-Po-Mag.</i></li>
               </ol>"""
}


@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Sample input data provided by the user
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Predict the main crop label directly using the machine learning model (RF)
    main_crop_label = model.predict(user_input)[0]

    # Get the probabilities for all classes using the machine learning model (RF)
    predicted_probabilities = model.predict_proba(user_input)[0]

    # Get the index of the main predicted crop label
    main_crop_index = np.argmax(predicted_probabilities)

    # Exclude the main predicted crop label and its corresponding probability
    predicted_probabilities[main_crop_index] = 0.0

    # Sort the probabilities in descending order
    sorted_indices = np.argsort(predicted_probabilities)[::-1]

    # Retrieve the top alternative crop options and their probabilities
    n = 5  # Number of top crops to consider
    top_alternatives = [(crop_names[i], predicted_probabilities[i]) for i in sorted_indices[:n] if predicted_probabilities[i] > 0]

    # Image URL for the main predicted crop
    main_crop_image_url = url_for('static', filename='images_crop/{}.jpg'.format(main_crop_label.lower()))

    # Capitalize the first character of the main predicted crop label
    main_crop_label_capitalized = main_crop_label.capitalize()

    # Get suggestions for the main crop
    suggestions = crop_suggestions.get(main_crop_label_capitalized, [])

    # Check if there are other crop options with probability greater than zero
    other_crops_available = any(prob > 0 for _, prob in top_alternatives)

    # Render the result template with prediction, main crop label, main crop probability, main crop image URL, other crop options, and suggestions
    return render_template('/mainpages/result.html', prediction="Prediction", main_crop_label=main_crop_label_capitalized,
                           main_crop_probability=1.0, main_crop_image_url=main_crop_image_url,
                           other_crop_options=top_alternatives, suggestions=suggestions,
                           other_crops_available=other_crops_available)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    # Preprocess crop name
    crop_name = request.form['crop'].strip().lower()

    # Read fertilizer data from a CSV file
    df = pd.read_csv('Data/fertilizer.csv')

    # Convert crop names in the DataFrame to lowercase and remove extra whitespace
    df['Crop'] = df['Crop'].str.strip().str.lower()

    # Filter the DataFrame based on the preprocessed crop name
    filtered_df = df[df['Crop'] == crop_name]

    if not filtered_df.empty:
        # Get recommended N, P, K values for the specified crop
        nr = filtered_df['N'].iloc[0]
        pr = filtered_df['P'].iloc[0]
        kr = filtered_df['K'].iloc[0]

        # User input for N, P, K
        N = int(request.form['N'])
        P = int(request.form['P'])
        K = int(request.form['K'])

        # Calculate differences between user-input and recommended N, P, K values
        n_diff = nr - N
        p_diff = pr - P
        k_diff = kr - K

        # Identify the nutrient with the largest difference
        temp = {abs(n_diff): "N", abs(p_diff): "P", abs(k_diff): "K"}
        max_value = temp[max(temp.keys())]

        # Select fertilizer recommendation based on the identified nutrient
        if max_value == "N":
            key = 'NHigh' if n_diff < 0 else 'Nlow'
        elif max_value == "P":
            key = 'PHigh' if p_diff < 0 else 'Plow'
        else:
            key = 'KHigh' if k_diff < 0 else 'Klow'

        # Get the fertilizer recommendation
        recommendation = fertilizer_dic[key]

        return render_template('/mainpages/fert_Result.html', recommendation=recommendation, title=title)
    else:
        # Handle the case where no rows match the condition
        return render_template('/mainpages/fert_Result.html', recommendation="No fertilizer recommendation found for the selected crop.", title=title)

#____________________________________________________________________________________________________________
#demobuddy
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_gemini_response(question):
    try:
        prompt = f"""
You are FarmBuddy AI, an intelligent agriculture assistant.

Rules:
1. Answer only agriculture and farming-related questions.
2. Reply in the same language as the user's question.
3. Do NOT use Markdown.
4. Do NOT use #, ##, ###, **, *, _, or backticks.
5. Use simple plain text.
6. When listing information, use bullets like:
   • Point 1
   • Point 2
   • Point 3
7. Leave a blank line between sections for readability.
8. Keep answers practical and concise.
9. If the question is unrelated to farming or agriculture, politely reply:

   "I'm FarmBuddy AI. I can only help with agriculture and farming-related questions."

Question:
{question}
"""

        response = client.models.generate_content(
            model="models/gemini-3.6-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        print("Gemini Error:", e)
        return f"Error: {str(e)}"

@app.route('/mainpages/farm_talk')
def farm_talk():
    return render_template('/mainpages/farm_talk.html')

@app.route('/generate_content', methods=['POST'])
def generate_content():
    question = request.json.get("question", "")
    generated_content = get_gemini_response(question)
    return generated_content


@app.route("/scheme/<int:scheme_id>")
def scheme_details(scheme_id):

    with open("Data/government_schemes.json", "r") as file:
        schemes = json.load(file)

    scheme = next(
        (item for item in schemes if item["id"] == scheme_id),
        None
    )

    if not scheme:
        return "Scheme Not Found", 404

    return render_template(
        "mainpages/scheme_details.html",
        scheme=scheme
    )

@app.route("/mainpages/government_schemes")
def government_schemes():

    with open("data/government_schemes.json", "r") as file:
        schemes = json.load(file)

    return render_template(
        "mainpages/government_schemes.html",
        schemes=schemes
    )


#________________________________________________________________________________________________________________
if __name__ == '__main__':
    app.run(debug=True)
