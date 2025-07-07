import pandas as pd

df = pd.read_json('test.json')

from sklearn.preprocessing import MultiLabelBinarizer

# Vektorisieren und in Matrix umwandeln. 1 = vorhanden, 0 = nicht vorhanden
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['ingredients'])

# Zutaten-Vokabular
vocab = set(mlb.classes_)

#nach Zutaten fragen
valid_input = False
while not valid_input:
    #strip entfernt leerzeichen
    user_input = input("What ingredients do you have? (e.g. milk, sugar, eggs...)\n> ").strip()
    if user_input:
        #lower macht alles lowercase
        #zutaten bei jedem komma trennen/split
        #if z.strip() entfernt leere/leerzeichen-zutaten
        user_ingredients = [x.strip().lower() for x in user_input.split(',') if x.strip()]
        #nur wenn mindestens eine existierende zutat eingegeben wurde weitermachen
        unknown = [x for x in user_ingredients if x not in vocab]
        if unknown:
            print(f"Error: Unknown ingredient: {', '.join(unknown)}")            
        elif any(user_ingredients):
            valid_input = True
        else:
            print("Error: Input can't be empty.\n")

user_vector = mlb.transform([user_ingredients])

from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=5, metric='cosine')  # cosine-Distanz funktioniert gut
knn.fit(X)

# Ähnlichste Rezepte finden
distances, indices = knn.kneighbors(user_vector)

# Ergebnisse anzeigen
for i, idx in enumerate(indices[0]):
    similarity = 1 - distances[0][i]
    print(f"{i+1}. {df.iloc[idx]['id']} (Score: {similarity:.2f})")
