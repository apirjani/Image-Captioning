# Andre's Image Captioning Project

## Project Description

The goal of this project, inspired by Andrej Karpathy's blog on image captioning using multimodal RNNs, is three-fold:
1. To learn more about the challenges of training data curation, as well as evaluation benchmark curation when developing transformers.
2. To compare the performance of state-of-the-art multimodal language models like GPT4-V with state-of-the-art multimodal encoder (or dual-encoder) models such as VisualBERT and CLIP on image captioning tasks.
3. To experiment with in-context learning (ICL) using GPT-4V on the image captioning task.

### Task Specifics

The task that these models are used to perform in this project is the following: given an image and two captions, choose the caption that better describes the image.

### **Sample**

Image: 

![person_320](https://github.com/apirjani/Image-Captioning/assets/89765975/f35c2c4a-7163-46db-aca4-581ad64ed369)

Caption 1: "The image shows two adults, a man and a woman, walking together in an urban square; the man is wearing a light brown jacket and jeans while carrying a red shopping bag, and the woman is dressed in a beige trench coat with black pants. In the background, there is another individual wearing a dark coat, walking away from the camera."

Caption 2: "The image depicts a man and a woman walking through a European-style town square, past a stone fountain with a statue, bordered by colorful buildings and storefronts."

Sample Prediction: Caption 2

## Execution Instructions

First, please make sure to run 'pip install -r requirements.txt' to install all necessary dependencies. Also, please make sure to enter your OpenAI API Key in the .env file to use GPT-4V.

To run a specific model on a test dataset, do the following:

1. Navigate to the "{model_name}.py" file, and add a parameter to load_data() that is the number of samples in your test set.
2. Simply run the command "python {model_name}.py" to run the script. Evaluation metrics will be printed into the terminal after execution,
   and a file called "{model_name}\_predictions.csv" will be generated containing the model's predictions on your test set.

## Results

GPT-4V (few-shot)

- Accuracy: 0.63
- Precision for class 1: 0.615
- Precision for class 2: 0.628
- Recall for class 1: 0.2
- Recall for class 2: 0.92
- F1 Score for class 1: 0.30
- F1 Score for class 2: 0.74

GPT-4V (zero-shot)

- Accuracy: 0.66
- Precision for class 1: 0.63
- Precision for class 2: 0.67
- Recall for class 1: 0.38
- Recall for class 2: 0.85
- F1 Score for class 1: 0.47
- F1 Score for class 2: 0.746

CLIP

- Accuracy: 0.58
- Precision for class 1: 0.48
- Precision for class 2: 0.69
- Recall for class 1: 0.65
- Recall for class 2: 0.53
- F1 Score for class 1: 0.55
- F1 Score for class 2: 0.60

VisualBERT

- Accuracy: 0.53
- Precision for class 1: 0.42
- Precision for class 2: 0.61
- Recall for class 1: 0.45
- Recall for class 2: 0.58
- F1 Score for class 1: 0.43
- F1 Score for class 2: 0.59
