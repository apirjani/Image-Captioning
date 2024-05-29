import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch import nn
from torch.nn.functional import cosine_similarity
from transformers import BertTokenizer, VisualBertModel
from load_data import load_data, load_annotations
from utils import save_predictions, evaluate_model

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def transform_image():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_regions(image, boxes, transform):
    regions = []
    # Image dimensions
    img_width, img_height = image.size

    for box in boxes:
        # Parse the normalized coordinates
        _, x_center, y_center, width, height, _ = box

        # Convert to pixel coordinates
        x = (x_center - width / 2) * img_width
        y = (y_center - height / 2) * img_height
        w = width * img_width
        h = height * img_height

        # Crop and transform
        region = image.crop((x, y, x + w, y + h))
        region = transform(region)
        regions.append(region)

    return torch.stack(regions)

def create_feature_extractor():
    # Load a pre-trained ResNet50 model
    model = resnet50(pretrained=True)
    # Remove the last two layers (avgpool and fc)
    feature_extractor = nn.Sequential(*list(model.children())[:-2])
    feature_extractor.eval()
    return feature_extractor

def extract_features(regions, feature_extractor):
    with torch.no_grad():
        # Extract features from the convolutional layer
        features = feature_extractor(regions)
        # Apply adaptive pooling to reduce spatial dimensions to 1x1
        features = nn.AdaptiveAvgPool2d((1, 1))(features)
        # Flatten the features for processing
        features = features.view(features.size(0), -1)

    return features

# Update the get_regional_visual_embeddings function to use this new extractor
def get_regional_visual_embeddings(image_path, annotations):
    image = load_image(image_path)
    transform = transform_image()
    regions = get_regions(image, annotations, transform)
    feature_extractor = create_feature_extractor()  # Use the new feature extractor
    features = extract_features(regions, feature_extractor)
    return features.unsqueeze(1)

def get_whole_image_visual_embeddings(image_path):
    image = load_image(image_path)
    transform = transform_image()
    image = transform(image).unsqueeze(0)
    feature_extractor = create_feature_extractor()
    features = extract_features(image, feature_extractor)
    return features.unsqueeze(1)

def load_and_process(image_path, text, annotations):
    # Load the model and processor
    model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    inputs = tokenizer(text, return_tensors="pt")
    text_token_count = inputs['input_ids'].size(1)  # Number of text tokens

    
    # Load the image
    visual_embeds = get_regional_visual_embeddings(image_path, annotations) if annotations is not None else get_whole_image_visual_embeddings(image_path)
    num_visual_tokens = visual_embeds.size(1)  # Assuming visual_embeds is [batch_size, num_regions, feature_dim]

    visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
    visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
    

    inputs.update(
        {
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        }
    )

    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    # Extract text and visual embeddings from the last hidden state
    text_embeddings = last_hidden_state[:, :text_token_count, :].mean(dim=1)  # Embeddings for text tokens
    visual_embeddings = last_hidden_state[:, -num_visual_tokens:, :].mean(dim=1) # Embeddings for visual tokens

    return text_embeddings, visual_embeddings

def compute_similarities(caption1_embeddings, caption2_embeddings):
    # Compute the cosine similarity between the first caption embedding and image embedding
    text_emb, visual_emb = caption1_embeddings
    avg_text_emb = text_emb.mean(dim=0, keepdim=True)
    avg_visual_emb = visual_emb.mean(dim=0, keepdim=True)
    similarity1 = cosine_similarity(avg_text_emb, avg_visual_emb)

    # Compute the cosine similarity between the second caption embedding and image embedding
    text_emb, visual_emb = caption2_embeddings
    avg_text_emb = text_emb.mean(dim=0, keepdim=True)
    avg_visual_emb = visual_emb.mean(dim=0, keepdim=True)
    similarity2 = cosine_similarity(avg_text_emb, avg_visual_emb)

    return [similarity1, similarity2]
    
if __name__ == "__main__":
    # Load the data
    data = load_data()

    # Load annotations data
    annotations_data = load_annotations()   

    # Iterate over the data and make API calls
    predictions = []
    human_labels = []
    for index, row in data.iterrows():
        image_path = "all_data/allimages/" + row['Image Filename']
        caption1 = row['Description1']
        caption2 = row['Description2']
        annotations = annotations_data[annotations_data['Image Filename'] == row['Image Filename']].values
        if len(annotations) == 0:
            caption1_result = load_and_process(image_path, caption1, None)
            caption2_result = load_and_process(image_path, caption2, None)
        else:
            caption1_result = load_and_process(image_path, [caption1]*len(annotations), annotations)
            caption2_result = load_and_process(image_path, [caption2]*len(annotations), annotations)
        similarities = compute_similarities(caption1_result, caption2_result)
        result = torch.stack(similarities).argmax().item() + 1
        predictions.append((row['Image Filename'], result))
        human_labels.append(row['Label'])
        print(f"Image: {row['Image Filename']}, Predicted Label: {result}")

    # Evaluate the model on human evaluations
    predicted_labels = [int(prediction) for _, prediction in predictions]
    evaluate_model(predicted_labels, human_labels)
    
    # Save the predictions
    save_predictions("visualBert_predictions.csv", predictions)
