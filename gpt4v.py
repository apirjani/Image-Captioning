import base64
import requests
import os
from dotenv import load_dotenv
from load_data import load_data
from few_shot_data_gptv import encode_image, example_captions, labels, base64_images
from utils import save_predictions, evaluate_model

import base64
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

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
                        "text": f"You are a state-of-the-art image caption classifier. I am going to give you an image, along with two captions that attempt to describe the image. Your task is to choose the caption that better describes the image. Your response should be a 1 or 2, corresponding to the caption that you selected. Any additional text generated results in a penalty.\n\nCaption 1: \"{caption1}\"\nCaption 2: \"{caption2}\"\n\n"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_test_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    return payload
    


# Function to call the OpenAI API and get the model's response
def call_openai_zeroshot(image_path, caption1, caption2):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = prepare_payload(image_path, caption1, caption2)
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def call_openai_fewshot(image_path, caption1, caption2):
    
    messages = []

    # Adding instructional message for the model
    messages.append({
        "role": "system",
        "content": {
            "type": "text",
            "text": "You are a state-of-the-art image caption classifier. Your task is to choose the caption that better describes the image. Please respond with a 1 or 2, corresponding to the caption that you select."
        }
    })

    # Adding examples for few-shot learning
    for i in range(len(example_captions)):
        messages.append({
            "role": "user",
            "content": {
                "type": "text",
                "text": f"Example {i+1}:\n\nCaption 1: \"{example_captions[i][0]}\"\nCaption 2: \"{example_captions[i][1]}\"\n\n"
            }
        })
        messages.append({
            "role": "user",
            "content": {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[i]}"
                }
            }
        })
        messages.append({
            "role": "assistant",
            "content": {
                "type": "text",
                "text": f"{labels[i]}"
            }
        })

    # Adding the new test image and captions for the model to evaluate
    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"Now, I am going to show you a new image along with two captions. Please choose the caption that best describes the image. Follow the format from the examples provided - your response should be a 1 or 2, corresponding to the caption that you selected. Any additional text generated results in a penalty.\n\nCaption 1: \"{caption1}\"\nCaption 2: \"{caption2}\"\n\n"
        }, {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
            }
        }]
    })

    # Using the OpenAI Python client library to create the completion
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-turbo",  # Ensure the model supports your requirements
        messages=messages,
        max_tokens=300,
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    
    # Load the data
    data = load_data()    
    
    # Iterate over the data and make API calls
    predictions = []
    human_labels = []
    for index, row in data.iterrows():
        image_path = "all_data/allimages/" + row['Image Filename']
        caption1 = row['Description1']
        caption2 = row['Description2']
        result = call_openai_zeroshot(image_path, caption1, caption2)
        predictions.append((row['Image Filename'], result))
        human_labels.append(row['Label'])
        print(f"Image: {row['Image Filename']}, Predicted Label: {result}")
    
    # Evaluate the model on human evaluations
    predicted_labels = [int(prediction) for _, prediction in predictions]
    evaluate_model(predicted_labels, human_labels)
    
    # Save the predictions
    save_predictions("gpt4v_predictions.csv", predictions)