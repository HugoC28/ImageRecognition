import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Le chemin d'accès au dossier
folder_path = "dataset/test/"

IMG_HEIGHT = 224
IMG_WIDTH = 224

# Parcourir chaque fichier du dossier
for filename in os.listdir(folder_path):
    # Le chemin complet du fichier
    file_path = os.path.join(folder_path, filename)

    # Charger l'image que vous souhaitez prédire
    image = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

    # Convertir l'image en tableau numpy
    image_array = img_to_array(image)

    # Étendre les dimensions du tableau numpy pour qu'il ait la forme (1, 224, 224, 3)
    image_array = image_array.reshape((1,) + image_array.shape)

    # Prétraiter les données
    image_array = image_array / 255.0

    # Charger le modèle sauvegardé
    model = keras.models.load_model('model.h5')

    # Faire une prédiction sur l'image
    prediction = model.predict(image_array)

    # Afficher la prédiction
    print("Nom du fichier : "+filename+" et la prédiction est ")
    print(prediction)
