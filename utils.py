from sklearn.metrics import precision_score, recall_score, f1_score

def save_predictions(file_path, predictions):
    with open(file_path, 'w') as f:
        f.write("Image Filename,Prediction\n")
        for image_filename, prediction in predictions:
            f.write(f"{image_filename},{prediction}\n")

def evaluate_model(predictions, ground_truth):
    correct = 0
    total = len(predictions)
    for i in range(total):
        if predictions[i] == ground_truth[i]:
            correct += 1
    accuracy = correct / float(total)  # Ensure float division
    precision = precision_score(ground_truth, predictions, average=None)
    recall = recall_score(ground_truth, predictions, average=None)
    f1 = f1_score(ground_truth, predictions, average=None)
    print(f"Accuracy: {accuracy}") 
    print(f"Precision for prediction of 1: {precision[0]}, Precision for a prediction of 2: {precision[1]}")
    print(f"Recall for prediction of 1: {recall[0]}, Recall for a prediction of 2: {recall[1]}")
    print(f"F1 Score for prediction of 1: {f1[0]}, F1 Score for a prediction of 2: {f1[1]}")
