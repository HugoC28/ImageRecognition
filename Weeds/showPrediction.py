import tkinter as tk
import customtkinter as CTk
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random as rd

# Le chemin d'accès au dossier
folder_path = "Weeds/dataset/test/"
images_list = os.listdir(folder_path)

IMG_HEIGHT = 500
IMG_WIDTH = 500

# Créer une fenêtre centrée à l'écran
fenetre = CTk.CTk()
width = 1400
height = 650
screen_width = fenetre.winfo_screenwidth()  # Width of the screen
screen_height = fenetre.winfo_screenheight() # Height of the screen
 
# Calculate Starting X and Y coordinates for Window
x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
fenetre.geometry('%dx%d+%d+%d' % (width, height, x, y))
fenetre.resizable(width=0, height=0)

# Définir le titre de la fenêtre
fenetre.title("Image Recognition")

# Charger l'image avec la bibliothèque PIL
image = Image.open("Weeds/dataset/test/foxtail (1).jpg")
width, height = image.size
ratio = 600/height
newSize = (int(width*ratio),int(height*ratio))
image = image.resize(newSize)


# Ajouter une étiquette pour afficher l'image
label_image = tk.Label(fenetre)
label_image.pack()

# Afficher l'image
photo = ImageTk.PhotoImage(image)
label_image.configure(image=photo)

# Ajouter une étiquette pour afficher du texte
label_texte = CTk.CTkLabel(fenetre, text="Click on the button to predict the species of a weed.",font=('Arial', 20))
label_texte.pack()

# Définir une fonction à appeler lorsqu'on clique sur le bouton
def on_click():
    ##Prediction
    # Selection d'un fichier aléatoire
    index=rd.randint(0,len(images_list))
    # Le chemin complet du fichier
    file_path = os.path.join(folder_path, images_list[index])
    # Charger l'image que vous souhaitez prédire
    image = load_img(file_path, target_size=(IMG_HEIGHT, IMG_WIDTH))

    # Convertir l'image en tableau numpy
    image_array = img_to_array(image)

    # Étendre les dimensions du tableau numpy pour qu'il ait la forme (1, 224, 224, 3)
    image_array = image_array.reshape((1,) + image_array.shape)

    # Prétraiter les données
    image_array = image_array / 255.0

    # Charger le modèle sauvegardé
    model = keras.models.load_model('Weeds/model.h5')

    # Faire une prédiction sur l'image
    prediction = model.predict(image_array)
    index_max_prediction = np.argmax(prediction)
    class_prediction= "Cocklebur" if index_max_prediction==0 else ("Foxtail" if index_max_prediction==1 else ("Pigweed" if index_max_prediction==2 else ("Ragweed")))

    # Afficher la prédiction
    my_text = "The image "+images_list[index]+" is a "+class_prediction+" with a probability of "+str(np.around(prediction[0][index_max_prediction]*100,decimals=2))+" %."
    label_texte.configure(text = my_text)
    ##Affichage de l'image
    global photo
    # Charger la nouvelle image
    newImage = Image.open(file_path)
    width, height = newImage.size
    ratio = 600/height
    newSize = (int(width*ratio),int(height*ratio))
    newImage = newImage.resize(newSize)
    # Mettre à jour la variable photo avec la nouvelle image
    photo = ImageTk.PhotoImage(newImage)
    # Mettre à jour l'image affichée
    label_image.configure(image=photo)

# Ajouter un bouton cliquable
bouton = CTk.CTkButton(fenetre, text="New Image", command=on_click)
bouton.pack()

# Lancer la boucle principale de la fenêtre
fenetre.mainloop()
