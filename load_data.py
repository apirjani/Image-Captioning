import pandas as pd
import os

def load_data(data_path='all_data/labeled_data.csv', num_examples=99):
    data = pd.read_csv(data_path)
    #shuffle data
    data = data.sample(frac=1).reset_index(drop=True)
    return data[:num_examples]

def load_annotations(directory_path='all_data/annotations/'):
    annotations = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=['class', 'x_center', 'y_center', 'width', 'height'])
        data['Image Filename'] = filename.replace('.txt', '.jpg')
        annotations.append(data)
    return pd.concat(annotations, ignore_index=True)


if __name__ == '__main__':
    data = load_annotations()
    print(data.head())

    
