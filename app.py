import streamlit as st
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from skimage.transform import resize
from PIL import Image
from tensorflow.keras.models import load_model
import os


# Dimensiones de la imagen de entrada
ancho = 64
alto = 64

#Clases
names = ['0','1','2','3','4', '5']
x=1

def model_prediction(img,model):
    #img_resize = resize(img,ancho,alto)
    x=preprocess_input(img)
    x = np.expand_dims(x,axis=0)
    preds = model.predict(x)
    return preds

def main():
    model=""
    if model=='':
        #archivo = os.path.join(os.getcwd(),'64x64_SIGNS1')
        #model = load_model(archivo)
        model = load_model('Modelo')
    st.title("Clasificador numeros")
    predicts = ""
    img_file_buffer = st.file_uploader("Carga una imagen", type=["png","jpg","jpeg"])
    # El usuario carga una imagen
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))    
        st.image(image, caption="Imagen", use_column_width=False)
    
    # El botón predicción se usa para iniciar el procesamiento
    if st.button("Predicción"):
         predicts = model_prediction(image, model)
         st.success('EL DIAGNÓSTICO ES: {}'.format(names[np.argmax(predicts)]))
    
if __name__ == '__main__':
    main()