from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

# IC: https://huggingface.co/microsoft/resnet-50
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

def text2image(prompt, model=1):
    # Procesamos la imagen según el prompt enviado
    if model == 1:
        return {"prompt": prompt, "name_image": "prueba 2", "path_image": "image_generate/perro.png"}
    elif model == 2:
        return {"prompt": prompt, "name_image": "prueba 1", "path_image": "image_generate/gato.png"}
    else:
        return {"error": "Debe eligir el modelo 1 o 2"}

def imageclassification(image):
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]