import streamlit as st
import requests
import base64
import json

def generate_image(uploaded_file, prompt, history=None):
    """
    Send image and prompt to FastAPI backend with structured history
    """
    url = "http://localhost:8000/generate-image/"
    
    files = {
        'file': (uploaded_file.name, uploaded_file, uploaded_file.type)
    }
    data = {
        'prompt': prompt
    }
    
    if history:
        data['history'] = json.dumps(history)
    
    try:
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def main():

    if "history" not in st.session_state:
        st.session_state.history = []
    
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None

    with st.sidebar:
        st.header("Upload Base Image")
        uploaded_file = st.file_uploader(
            "Choose an image", 
            type=['jpeg', 'jpg', 'png'],
            help="Upload base image to modify"
        )
        
        if uploaded_file is not None:
            st.session_state.uploaded_image = uploaded_file
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    for item in st.session_state.history:
        if item['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(item['content'])
        elif item['role'] == 'model':
            with st.chat_message("assistant"):
                if 'image_base64' in item:
                    st.image(base64.b64decode(item['image_base64']))

    if prompt := st.chat_input("What would you like to do with the image?", disabled=not st.session_state.uploaded_image):
        st.session_state.history.append({
            "role": "user", 
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner('Generating image...'):
                generation_response = generate_image(
                    st.session_state.uploaded_image, 
                    prompt,
                    history=st.session_state.history
                )
                
                if generation_response:
                    generated_image_bytes = base64.b64decode(generation_response['image_base64'])
                    
                    st.image(generated_image_bytes, caption='Generated Image')
                    
                    st.session_state.history.append({
                        "role": "model", 
                        "image_base64": generation_response['image_base64']
                    })

if __name__ == "__main__":
    main()
