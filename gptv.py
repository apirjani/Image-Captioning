import base64
import requests
import os
from dotenv import load_dotenv
from load_data import load_data
from few_shot_data_gptv import encode_image, example_captions, labels, base64_images

import base64
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Key
api_key = os.getenv('OPENAI_API_KEY')

# Function to prepare the payload for the API request
def prepare_payload(image_path, caption1, caption2):
    base64_test_image = encode_image(image_path)
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a state-of-the-art image caption classifier. I am going to give you an image, along with two captions that attempt to describe the image. Your task is to choose the caption that better describes the image.\n\nHere are 8 examples of images and captions that you can use to train yourself: "
                    }
                ]
            }
        ]
    }

    # Dynamically generate example messages
    for i in range(len(example_captions)):
        payload["messages"][0]["content"].extend([
            {
            "type": "text",
            "text": f"Example {i+1}:\n\nCaption 1: \"{example_captions[i][0]}\"\nCaption 2: \"{example_captions[i][1]}\"\n\nBest caption: Caption {labels[i]}"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_images[i]}"
            }
            }
        ])
    
    # Add the test image and captions
    payload["messages"][0]["content"].extend([
        {
        "type": "text",
        "text": f"Now, I am going to show you a new image along with two captions. Please choose the caption that best describes the image. Follow the format from the examples provided, and do not include additional information in your response.\n\nCaption 1: \"{caption1}\"\nCaption 2: \"{caption2}\"\n\nBest caption: Caption "
        },
        {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_test_image}"
        }
        }
    ])
    payload["max_tokens"] = 300
    return payload
    


# Function to call the OpenAI API and get the model's response
def call_openai_api(image_path, caption1, caption2):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = prepare_payload(image_path, caption1, caption2)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


if __name__ == '__main__':
    
    # Load the data
    data = load_data(num_examples=5)    
    
    # Iterate over the data and make API calls
    for index, row in data.iterrows():
        image_path = "all_data/allimages/" + row['Image Filename']
        # print(f"Processing image: {image_path}")
        caption1 = row['Description1']
        # print(f"Caption 1: {caption1}")
        caption2 = row['Description2']
        # print(f"Caption 2: {caption2}")
        result = call_openai_api(image_path, caption1, caption2)
        print(f"Image: {row['Image Filename']}, Predicted Label: {result}")
