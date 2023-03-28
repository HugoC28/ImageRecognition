import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


# Le chemin d'accès au dossier
folder_path = "Weeds/dataset/test/" #replace by folder_path = "dataset/test/" if it doesn't work.

IMG_HEIGHT = 500
IMG_WIDTH = 500

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
    model = keras.models.load_model('Weeds/model.h5') #replace by 'model.h5' if it doesn't work.

    # Faire une prédiction sur l'image
    prediction = model.predict(image_array)
    index_max_prediction = np.argmax(prediction)
    class_prediction= "Cocklebur" if index_max_prediction==0 else ("Foxtail" if index_max_prediction==1 else ("Pigweed" if index_max_prediction==2 else ("Ragweed")))

    # Afficher la prédiction
    print("The image "+filename+" is a "+class_prediction+" with a probability of "+str(np.around(prediction[0][index_max_prediction]*100,decimals=2))+" %.")
