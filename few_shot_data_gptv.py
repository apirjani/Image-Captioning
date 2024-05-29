import base64

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

base_image_path = "all_data/allimages/"
example_image_paths = ["crop001702.jpg", "crop001694.jpg", "1003.jpg", "person_166.jpg", "person_320.jpg", "person_and_bike_019.jpg", "384.jpg", "crop001274.jpg"]
example_image_paths = [base_image_path + image_path for image_path in example_image_paths]
# Getting the base64 string
base64_images = [encode_image(path) for path in example_image_paths]
example_captions = [
    ("The image shows a group of people at a car show, standing near a Lancia exhibit where several cars, including a silver one in the foreground, are displayed.", 
     "One of the people is wearing a long black coat and appears to be speaking on a cellphone, while the other, further in the background to the right, is dressed in a casual jacket and jeans, observing a car at an auto show."),
    ("In the foreground, a man in a light grey sweater looks up and slightly away from the camera. In the center, a pair of individuals engage in conversation; the man wears a black cap and a dark shirt, and the woman wears a black top and is gesturing with her right hand.",
     "The image shows a bustling auto show with a crowd of people examining and interacting around new cars in an indoor exhibition space. Some visitors are closely inspecting the vehicles while others are in conversation."),
    ("The image shows a woman riding a bicycle wearing a red jacket and face mask, with other people visible in the background, likely walking on a city street. The scene appears to be captured in an urban setting with a somewhat blurred and tilted perspective.",
     "The image shows three people: a person in a bright pink jacket riding a bicycle, a person dressed in black walking behind them, and another person wearing a beige coat walking in the opposite direction. All three are wearing masks."),
    ("The image shows two pairs of people walking in opposite directions on a metal pedestrian bridge, with autumnal trees in the background.",
     "The image shows two women walking away on a steel bridge; one appears older with gray hair and a black coat, while the other is younger, wearing a black jacket and blue jeans."),
    ("The image shows two adults, a man and a woman, walking together in an urban square; the man is wearing a light brown jacket and jeans while carrying a red shopping bag, and the woman is dressed in a beige trench coat with black pants. In the background, there is another individual wearing a dark coat, walking away from the camera.",
     "The image depicts a man and a woman walking through a European-style town square, past a stone fountain with a statue, bordered by colorful buildings and storefronts."),
    ("A person is locking a bicycle to a fence in a park covered with fallen leaves. The environment appears chilly and autumnal.",
     "The person is a man wearing a blue vest over a jacket and checking or fixing the front basket of a bicycle in a park-like environment."),
    ("The image you've shared shows just one person, upside down, possibly engaging in an adventurous activity. This person is wearing casual clothing and a face mask.",
     "The image features an upside-down view of a person with a surgical mask, surrounded by a foggy or misty environment. The focus is on the individual's face, which is centered in the composition, adding a mysterious or disorienting feel to the image."),
    ("The image shows two men in a grassy field; one is jumping with his arms raised while attempting to catch or hit a ball, and the other, wearing a shirt labeled ""Google,"" stands watching him.",
     "The person leaping in the air is wearing a light grey sweatshirt, white pants, and is barefoot, actively reaching upwards with both hands, possibly engaging in a sport or game outdoors.")
]
labels = ['1', '2', '1', '2', '2', '1', '2', '1']
