import pandas as pd
import tkinter as tk
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import random

# --- logic --- #
df = pd.read_json('test.json')

# vectorise and convert to matrix. 1 = existing, 0 = non-existing
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['ingredients'])

# ingredients-vocabulary
vocab = set(mlb.classes_)

knn = NearestNeighbors(n_neighbors=5, metric='cosine')  # cosine-distance works best
knn.fit(X)

# random possible outputs for a 'no ingredient entered' error message
rnd_err_msg = {"What do you want me to do now?",
               "I suggest: a grocery trip",
               "Time to sin and order take out after all",
               "Perhaps a water soup with some ice?"}

# asks for ingredients and finds the closest fitting neighbours
def find_recipes():
    # strip removes empty spaces
    user_input =entry.get().strip()

    # if user input is empty (= no input/ingredients entered)
    if not user_input:
        # outputs a random error message from a list of possible funny responses
        output_text.set(random.choice(list(rnd_err_msg)))
        return

    # creates a set of all with ',' seperated, entered ingredients, in lower case and without spaces
    user_ingredients = [x.strip().lower() for x in user_input.split(',') if x.strip()]
    # boolean that checks, if any entered ingredient is not in our vocab set
    unknown = [x for x in user_ingredients if x not in vocab]
    # if any unknown ingredient was found
    if unknown:
        output_text.set(f"Error: Unknown ingredient(s): {', '.join(unknown)}")
        return

    user_vector = mlb.transform([user_ingredients])
    # finds best fitting recipes
    distances, indices = knn.kneighbors(user_vector)

    results = []
    for i , idx in enumerate(indices[0]):
        similarity = 1 - distances[0][i]
        results.append(f"{i+1}. {df.iloc[idx]['id']} (Score: {similarity:.2f})")

    output_text.set("\n".join(results))


# --- UI --- #
root = tk.Tk()
root.title("Recipe Finder")

tk.Label(root, text="What ingredients do you have? (e.g. milk, sugar, eggs...)").pack(pady=5)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

submit_btn = tk.Button(root, text="Find Recipes", command=find_recipes)
submit_btn.pack(pady=5)

output_text = tk.StringVar()
output_label = tk.Label(root, textvariable=output_text, justify="left", anchor="w")
output_label.pack(padx=10, pady=10, fill='both')

root.mainloop()
