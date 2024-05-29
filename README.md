# Andre's Image Captioning Project

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
