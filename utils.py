from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset

# T2I: https://huggingface.co/CompVis/stable-diffusion-v1-4
t2i_1 = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
t2i_1 = t2i_1.to("cuda")

# T2I: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
t2i_2 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
t2i_2 = t2i_2.to("cuda")

# IC: https://huggingface.co/microsoft/resnet-50
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# prompt: descripción de la imagen a generar
# model: elige el modelo 1 o 2 para generar la imagen, por default elige la 1
def text2image(prompt, model=1):
    # Procesamos la imagen según el prompt enviado
    if model == 1:
        image = t2i_1(prompt).images[0]
    elif model == 2:
        image = t2i_2(prompt=prompt).images[0]
    else:
        return {"error": "Debe eligir el modelo 1 o 2"}

    # Damos formato a la imagen para que no tenga un nombre largo y reemplazamos los espacios por _
    name_img = prompt.replace(" ","_")
    name_img = name_img[:20]
    name_img = "{0}.png".format(name_img)
    path_img = "image_generate/{0}".format(name_img)
    # Guardamos la imagen
    image.save(path_img)

    return {"prompt": prompt, "name_image": name_img, "path_image": path_img}

def imageclassification(image):
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]