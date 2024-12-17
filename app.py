from flask import Flask, render_template, request
import pandas as pd
from pulp import *

# Initialize the Flask app
app = Flask(__name__)

# Predefined ingredients data: [Protein, Fat, Fibre, Salt, Sugar, Cost]
predefined_data = {
    "Chicken": [0.100, 0.080, 0.001, 0.002, 0.000, 0.095],
    "Beef": [0.200, 0.100, 0.005, 0.005, 0.000, 0.150],
    "Mutton": [0.150, 0.110, 0.003, 0.007, 0.000, 0.100],
    "Rice": [0.000, 0.010, 0.100, 0.002, 0.000, 0.002],
    "Wheat bran": [0.040, 0.010, 0.150, 0.008, 0.000, 0.005],
    "Corn": [0.0329, 0.0128, 0.028, 0.000, 0.045, 0.012],
    "Peanuts": [0.258, 0.492, 0.085, 0.001, 0.047, 0.013],
    "Soya": [0.2434, 0.2352, 0.029, 0.0015, 0.050, 0.1],
    "Egg": [0.100, 0.040, 0.0005, 0.001, 0.000, 0.025],
    "Milk": [0.170, 0.010, 0.0005, 0.002, 0.005, 0.010],
}

@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        # Collect predefined ingredients from form
        selected_ingredients = request.form.getlist("predefined")

        # Nutrition and cost data
        nutrition = pd.DataFrame(columns=["Protein", "Fat", "Fibre", "Salt", "Sugar"])
        dict_costs = {}
        variables = []

        # Add selected predefined ingredients
        for ingredient in selected_ingredients:
            data = predefined_data[ingredient]
            nutrition.loc[ingredient] = data[:5]
            dict_costs[ingredient] = data[5]
            variables.append(ingredient)

        # Collect custom ingredients from form
        custom_names = request.form.getlist("custom_ingredient_name")
        custom_proteins = request.form.getlist("custom_protein")
        custom_fats = request.form.getlist("custom_fat")
        custom_fibres = request.form.getlist("custom_fibre")
        custom_salts = request.form.getlist("custom_salt")
        custom_sugars = request.form.getlist("custom_sugar")
        custom_costs = request.form.getlist("custom_cost")

        # Add custom ingredients to nutrition and cost data
        for i in range(len(custom_names)):
            name = custom_names[i]
            protein = float(custom_proteins[i])
            fat = float(custom_fats[i])
            fibre = float(custom_fibres[i])
            salt = float(custom_salts[i])
            sugar = float(custom_sugars[i])
            cost = float(custom_costs[i])

            nutrition.loc[name] = [protein, fat, fibre, salt, sugar]
            dict_costs[name] = cost
            variables.append(name)

        # Ensure there's at least one ingredient
        if not variables:
            return render_template("index.html", results={"error": "Please select at least one ingredient or add a custom ingredient."})

        # Optimization model
        model = LpProblem("Optimize_Protein_Bar", LpMinimize)
        x = LpVariable.dicts("Qty", variables, lowBound=0, cat="Continuous")

        # Objective Function: Minimize Cost
        model += lpSum([dict_costs[i] * x[i] for i in variables])

        # Constraints
        model += lpSum([x[i] for i in variables]) == 120, "Total_Quantity"
        model += lpSum([x[i] * nutrition.loc[i, "Protein"] for i in variables]) >= 22, "Min_Protein"
        model += lpSum([x[i] * nutrition.loc[i, "Fat"] for i in variables]) <= 22, "Max_Fat"
        model += lpSum([x[i] * nutrition.loc[i, "Fibre"] for i in variables]) >= 6, "Min_Fibre"
        model += lpSum([x[i] * nutrition.loc[i, "Salt"] for i in variables]) <= 3, "Max_Salt"
        model += lpSum([x[i] * nutrition.loc[i, "Sugar"] for i in variables]) <= 20, "Max_Sugar"

        # Solve the model
        status = model.solve()

        # Collect results
        results = {
            "cost": round(value(model.objective), 4),
            "status": LpStatus[status],
            "variables": [(f"Qty_{ingredient}", round(x[ingredient].varValue, 4)) for ingredient in variables],
        }

    return render_template("index.html", results=results)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    
