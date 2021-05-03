import numpy as np 
import joblib
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import catboost



#load model
app = Flask(__name__)
api = Api(app)

model = joblib.load("restaurant_model.sav")

class MakePrediction(Resource):
    @staticmethod
    def post():
        data = request.form.to_dict()
        rec_name = data["recipe_name"]
	prod_type = data["product_type"]
        cal = data["calories"]
        carb = data["carbs"]
        time = data["cooking_time"]
	cus = data["cuisine"]
	diff = data["difficulty"]
	desc = data["description"]
        dsh = data["dish_type"]
        heat = data["heat_level"]
        fat = data["fat"]
	classic = data["is_classic"]
	ingrd = data["number_of_ingredients_per_recipe"]
	pref = data["preferences"]
	carbs_content = data["carbs_content"]
	seasons = data["seasons"]
        protein = data["proteins"]
	pro_type = data["protein_types"]
        course_type = data["course_type"]
	meta = data["meta_tags"]
	year = data["year"]
	week = data["week"]
	dishtyp = data["dish_typ"]

	answer = model.predict([rec_name, prod_type, cal, carb, time, cus, diff, desc, dsh, heat, fat, classic, ingrd, 
					pref, carbs_content, seasons, protein, pro_type, course_type, meta, year, week, dishtyp])
        answer = np.exp(answer)

        return jsonify({
            'Prediction': answer
        })
    
    
api.add_resource(MakePrediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
