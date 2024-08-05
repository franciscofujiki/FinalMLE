import streamlit as st
from utils import text2image, imageclassification
from PIL import Image

if 'clicked_classification' not in st.session_state:
    st.session_state.clicked_classification = False

if 'clicked_generate' not in st.session_state:
    st.session_state.clicked_generate = False

if 'clicked_generate_and_classification' not in st.session_state:
    st.session_state.clicked_generate_and_classification = False

def click_generate_button():
    st.session_state.clicked_generate = True

def click_classification_button():
    st.session_state.clicked_classification = True

def click_generate_and_classification_button():
    st.session_state.clicked_generate_and_classification = True

st.title("Text to Image and Classification")
st.text_input("Prompt para crear imagen", key="prompt")

left_column, center_column, right_column = st.columns(3)

with left_column:
    chosen = st.radio(
        'Elija modelo',
        ("Modelo 1", "Modelo 2"), key="modelo")
    st.button("Generar Imagen", on_click=click_generate_button)
    st.write("* Se tomará el prompt para generar la imagen")

with center_column:
    st.button("Clasificar Imagen", on_click=click_classification_button)
    st.write("* Se tomará la última imagen generada")

with right_column:
    st.button("Generar y Clasificar Imagen", on_click=click_generate_and_classification_button)
    st.write("* Se tomará el prompt para generar la imagen y se clasficará la imagen")

st.divider()

# Cuando haga clic en Generar
if st.session_state.clicked_generate:
    if st.session_state.prompt != "":
        if st.session_state.modelo == "Modelo 1":
            st.session_state.result = text2image(st.session_state.prompt, 1)
            st.image(st.session_state.result["path_image"], caption=st.session_state.result["prompt"])
        elif st.session_state.modelo == "Modelo 2":
            st.session_state.result = text2image(st.session_state.prompt, 2)
            st.image(st.session_state.result["path_image"], caption=st.session_state.result["prompt"])
    else:
        st.warning("Debe escribir un prompt")

    st.session_state.clicked_generate = False

# Cuando haga clic en Clasificar
if st.session_state.clicked_classification:
    if 'result' in st.session_state:
        file_path = st.session_state.result["path_image"]
        # Open the image
        image = Image.open(file_path)

        clasificacion = imageclassification(image)
        st.image(st.session_state.result["path_image"], caption=st.session_state.result["prompt"])
        st.write("Clasificación: {0}".format(clasificacion))
    else:
        st.warning("Debe generar una imagen")

    st.session_state.clicked_classification = False

# Cuando haga clic en Generar y Clasificar
if st.session_state.clicked_generate_and_classification:
    if st.session_state.prompt != "":
        if st.session_state.modelo == "Modelo 1":

            st.session_state.result = text2image(st.session_state.prompt, 1)
            file_path = st.session_state.result["path_image"]

            # Open the image
            image = Image.open(file_path)

            clasificacion = imageclassification(image)

            st.image(file_path, caption=st.session_state.result["prompt"])
            st.write("Clasificación: {0}".format(clasificacion))

        elif st.session_state.modelo == "Modelo 2":

            st.session_state.result = text2image(st.session_state.prompt, 2)
            file_path = st.session_state.result["path_image"]

            # Open the image
            image = Image.open(file_path)

            clasificacion = imageclassification(image)

            st.image(file_path, caption=st.session_state.result["prompt"])
            st.write("Clasificación: {0}".format(clasificacion))

    else:
        st.warning("Debe escribir un prompt")

    st.session_state.clicked_generate_and_classification = False

# st.write(st.session_state)