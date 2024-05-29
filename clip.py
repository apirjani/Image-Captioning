from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from load_data import load_data
from utils import save_predictions, evaluate_model

def load_and_process(image_path, text):
    # Load the model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    max_sequence_length = 77
    
    # Load the image
    image = Image.open(image_path)
    
    # Process inputs
    text = [caption[:max_sequence_length] for caption in text]
    inputs = processor(text, images=image, return_tensors="pt", padding=True)

    # Generate embeddings
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score in logits
    probs = logits_per_image.softmax(dim=1)      # we can take softmax to get probabilities
    
    return probs

if __name__ == "__main__":
    # Load the data
    data = load_data()    
    
    # Iterate over the data and make API calls
    predictions = []
    human_labels = []
    for index, row in data.iterrows():
        image_path = "all_data/allimages/" + row['Image Filename']
        caption1 = row['Description1']
        caption2 = row['Description2']
        result = load_and_process(image_path, [caption1, caption2]).argmax().item() + 1
        predictions.append((row['Image Filename'], result))
        human_labels.append(row['Label'])
        print(f"Image: {row['Image Filename']}, Predicted Label: {result}")
    
    # Evaluate the model on human evaluations
    predicted_labels = [int(prediction) for _, prediction in predictions]
    evaluate_model(predicted_labels, human_labels)
    
    # Save the predictions
    save_predictions("clip_predictions.csv", predictions)
    



