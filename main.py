from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepforest import main
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
import base64
import matplotlib.pyplot as plt
import cloudinary
import cloudinary.uploader

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configure Cloudinary
cloudinary.config(
    cloud_name='dgajjwgwv',  
    api_key='647235858843329',        
    api_secret='KmyOqu73auGvZ6AOJZWtjDdTLlQ'   
)

# Load the pre-trained model
model = main.deepforest()
model.use_release()

@app.post("/predict")
async def predict_trees(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Ensure the image array has 3 channels
        if image_array.ndim == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]


        # Predict trees in the image
        predictions = model.predict_image(image=image_array)

        # Count the number of trees
        number_of_trees = len(predictions)
        
        # Annotate the image with predictions
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)
        for _, row in predictions.iterrows():
            rect = plt.Rectangle((row['xmin'], row['ymin']), row['xmax'] - row['xmin'], row['ymax'] - row['ymin'],
                                    fill=False, color="red", linewidth=1)
            ax.add_patch(rect)
        # Save the annotated image to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)

        # Upload the original image to Cloudinary
        original_response = cloudinary.uploader.upload(io.BytesIO(image_data), resource_type="image")
        original_image_url = original_response['secure_url']

        # Upload the annotated image to Cloudinary
        annotated_response = cloudinary.uploader.upload(buf, resource_type="image")
        annotated_image_url = annotated_response['secure_url']

        return JSONResponse(content={
            "number_of_trees": number_of_trees,
            "original_image_url": original_image_url,
            "annotated_image_url": annotated_image_url
        })
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")