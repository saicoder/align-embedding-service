import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import requests
import numpy as np
from transformers import AlignProcessor, AlignModel
from os import getenv

app = FastAPI()

# Initialize devices
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and processor
# clip_model = CLIPModel.from_pretrained(
#     "openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load BLIP model and processor
# blip_model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base").to(device)
# blip_processor = BlipProcessor.from_pretrained(
#     "Salesforce/blip-image-captioning-base")

processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
model = AlignModel.from_pretrained("kakaobrain/align-base")


def get_text_embedding(text: str) -> np.ndarray:
    inputs = processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)
    return text_embedding.cpu().numpy()


def get_image_embedding(image_url: str) -> np.ndarray:
    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error opening image: {e}")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs)
    return image_embedding.cpu().numpy()


def get_image_description(image_url: str) -> str:
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error opening image: {e}")

    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate description with custom parameters
    outputs = model.generate(
        **inputs,
        max_length=50,  # Adjust the maximum length of the description
        min_length=10,  # Adjust the minimum length of the description
        length_penalty=5.0,  # Adjust the length penalty to balance between length and quality
        num_beams=4,  # Use beam search to generate more diverse and better-quality descriptions
        early_stopping=True  # Stop early when an optimal description is found
    )

    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Define request models
class TextRequest(BaseModel):
    text: str


class ImageRequest(BaseModel):
    image_url: str


@app.post("/text-embedding/")
async def text_embedding(request: TextRequest):
    embedding = get_text_embedding(request.text)
    return StreamingResponse(io.BytesIO(embedding.tobytes()), media_type="application/octet-stream")


@app.post("/image-embedding/")
async def image_embedding(request: ImageRequest):
    embedding = get_image_embedding(request.image_url)
    return StreamingResponse(io.BytesIO(embedding.tobytes()), media_type="application/octet-stream")


@app.post("/image-description/")
async def image_description(request: ImageRequest):
    description = get_image_description(request.image_url)
    return {"description": description}


if __name__ == "__main__":
    import uvicorn
    port = getenv('PORT', '3001')
    uvicorn.run(app, host="0.0.0.0", port=int(port))
