import os
import io
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Gemini Image Generation API")
os.environ['GEMINI_API_KEY'] = os.getenv("API_KEY")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-image/")
async def generate_image(
    file: UploadFile = File(...), 
    prompt: str = Form(...),
    history: str = Form(None)
):
    """
    Generate an image using Gemini API with an uploaded image and text prompt.
    Supports conversation history in specific JSON structure.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable must be set")

    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        input_image_bytes = await file.read()
        
        with open("temp_input.jpeg", "wb") as f:
            f.write(input_image_bytes)

        uploaded_file = client.files.upload(file="temp_input.jpeg")

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        if history:
            history_list = json.loads(history)
            for hist_item in history_list:
                if hist_item['role'] == 'user':
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=hist_item['content'])]
                        )
                    )
                elif hist_item['role'] == 'model' and 'image_base64' in hist_item:
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[
                                types.Part.from_bytes(
                                    mime_type="image/png",
                                    data=base64.b64decode(hist_item['image_base64'])
                                )
                            ]
                        )
                    )

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_modalities=["image", "text"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_CIVIC_INTEGRITY",
                    threshold="OFF",
                ),
            ],
            response_mime_type="text/plain",
        )

        generated_image_bytes = None

        for chunk in client.models.generate_content_stream(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            
            if chunk.candidates[0].content.parts[0].inline_data:
                generated_image_bytes = chunk.candidates[0].content.parts[0].inline_data.data
                break

        os.remove("temp_input.jpeg")

        if not generated_image_bytes:
            raise HTTPException(status_code=500, detail="Failed to generate image")

        image_base64 = base64.b64encode(generated_image_bytes).decode('utf-8')

        return {
            'image_base64': image_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)