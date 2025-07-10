import pandas as pd
import tkinter as tk
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import NearestNeighbors
import random

df = pd.read_json('test.json')

# Vektorisieren und in Matrix umwandeln. 1 = vorhanden, 0 = nicht vorhanden
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['ingredients'])

# Zutaten-Vokabular
vocab = set(mlb.classes_)

knn = NearestNeighbors(n_neighbors=5, metric='cosine')  # cosine-Distanz funktioniert gut
knn.fit(X)

rnd_err_msg = {"What do you want me to do now?",
               "I suggest: a grocery trip",
               "Time to sin and order take out after all",
               "Perhaps a water soup with some ice?"}

#nach Zutaten fragen
def find_recipes():
    user_input =entry.get().strip()

    if not user_input:
        # da evtl die funny fehlermeldungen? hahahaha
        output_text.set(random.choice(list(rnd_err_msg)))
        return

    user_ingredients = [x.strip().lower() for x in user_input.split(',') if x.strip()]
    unknown = [x for x in user_ingredients if x not in vocab]
    if unknown:
        output_text.set(f"Error: Unknown ingredient(s): {', '.join(unknown)}")
        return

    user_vector = mlb.transform([user_ingredients])
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


# vorherige version
#valid_input = False
#while not valid_input:
    #strip entfernt leerzeichen
 #   user_input = input("What ingredients do you have? (e.g. milk, sugar, eggs...)\n> ").strip()
  #  if user_input:
        #lower macht alles lowercase
        #zutaten bei jedem komma trennen/split
        #if z.strip() entfernt leere/leerzeichen-zutaten
   #     user_ingredients = [x.strip().lower() for x in user_input.split(',') if x.strip()]
        #nur wenn mindestens eine existierende zutat eingegeben wurde weitermachen
    #    unknown = [x for x in user_ingredients if x not in vocab]
     #   if unknown:
      #      print(f"Error: Unknown ingredient: {', '.join(unknown)}")
       # elif any(user_ingredients):
        #    valid_input = True
        #else:
         #   print("Error: Input can't be empty.\n")

#user_vector = mlb.transform([user_ingredients])

#from sklearn.neighbors import NearestNeighbors

#knn = NearestNeighbors(n_neighbors=5, metric='cosine')  # cosine-Distanz funktioniert gut
#knn.fit(X)

# Ähnlichste Rezepte finden
#distances, indices = knn.kneighbors(user_vector)

# Ergebnisse anzeigen
#for i, idx in enumerate(indices[0]):
 #   similarity = 1 - distances[0][i]
  #  print(f"{i+1}. {df.iloc[idx]['id']} (Score: {similarity:.2f})")
