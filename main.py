import pandas as pd
import tkinter as tk
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import random

# --- logic --- #
df = pd.read_json('renamed_recipes.json')

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
    user_input = entry.get().strip()

    # if user input is empty (= no input/ingredients entered)
    if not user_input:
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

    scores = []
    for i, recipe_vector in enumerate(X):
        # flatten the recipe vector if it's a sparse matrix
        rv = recipe_vector.toarray().flatten() if hasattr(recipe_vector, "toarray") else recipe_vector
        # create a mask: True where the recipe requires an ingredient
        needed = rv == 1
        # check if the user has those required ingredients
        has = user_vector[0][needed]
        # calculate score
        score = np.sum(has) / np.sum(needed)
        scores.append((i, score))

    # sort recipes by score (best match first)
    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for rank, (idx, score) in enumerate(scores[:5]):
        results.append(f"{rank+1}. {df.iloc[idx]['id']} (Score: {score:.2f})")

    output_text.set("\n".join(results))


# --- UI --- #
root = tk.Tk()
root.geometry("600x400")
root.title("Recipe Finder")

tk.Label(root, text="What ingredients do you have? (e.g. milk, sugar, eggs...)").pack(pady=5)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

submit_btn = tk.Button(root, text="Find Recipes", command=find_recipes)
submit_btn.pack(pady=5)

output_text = tk.StringVar()
output_label = tk.Label(root, textvariable=output_text, justify="left", anchor="w")
output_label.pack(padx=10, pady=10, fill='both')

# --- Auto-Complete Recipe-Suggestions --- #

suggestion_box = tk.Listbox(root, height=5)
suggestion_box.pack(pady=(0, 5))
suggestion_box.place_forget()  # hidden at first

# updates the initial suggestions
def update_suggestions(event):
    typed = entry.get().split(',')[-1].strip().lower()
    if typed == "":
        suggestion_box.place_forget()
        return
    
    matches = [item for item in vocab if item.startswith(typed)]
    if matches:
        # sorting: exact matches at first, then shortest first, then alphabetical
        matches.sort(key=lambda x: (x != typed, len(x), x))
        
        suggestion_box.delete(0, tk.END)
        for match in matches[:10]:  # 10 suggestions max
            suggestion_box.insert(tk.END, match)
        suggestion_box.place(x=entry.winfo_x(), y=entry.winfo_y() + entry.winfo_height())
    else:
        suggestion_box.place_forget()

# prints the determined suggestions
def insert_suggestion(event):
    if suggestion_box.curselection():
        selected = suggestion_box.get(suggestion_box.curselection())
        current_text = entry.get()
        parts = [x.strip() for x in current_text.split(',')]
        parts[-1] = selected
        new_text = ', '.join(parts) + ', '
        entry.delete(0, tk.END)
        entry.insert(0, new_text)
        suggestion_box.place_forget()

entry.bind('<KeyRelease>', update_suggestions)
suggestion_box.bind("<<ListboxSelect>>", insert_suggestion)

root.mainloop()
