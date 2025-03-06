import torch
import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import numpy as np
from torch.nn.functional import cosine_similarity


# haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=45),          # Randomly rotate the image by ±15 degrees
    transforms.RandomHorizontalFlip(p=0.5),         # Randomly flip the image horizontally with a 50% probability
    transforms.ColorJitter(brightness=0.2,          # Randomly adjust brightness
                           contrast=0.2, 
                           saturation=0.2, 
                           hue=0.1),
])



def preprocess_data_averaging(directory):
    # Store embeddings and corresponding labels
    embeddings_dict = {}

    # Load images from the file system
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        
        if os.path.isdir(person_dir):
            total_embeddings = 0  # Track number of embeddings for each person
            embedding_sum = 0  # Sum of all embeddings for averaging

            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                
                # Read and check the image
                image = cv2.imread(image_path)
                if image is not None:
                    # Include the original image's embedding
                    embedding = extract_face_embedding(image, person_name)
                    if embedding is not None:
                        embedding_sum += embedding
                        total_embeddings += 1

                    # Apply augmentations and accumulate embeddings
                    for _ in range(5):  # Applying augmentations 5 times per image
                        augmented_image = augment_image(image)
                        augmented_embedding = extract_face_embedding(augmented_image, person_name)
                        if augmented_embedding is not None:
                            embedding_sum += augmented_embedding
                            total_embeddings += 1

            # Calculate the average embedding if we have valid embeddings
            if total_embeddings > 0:
                avg_embedding_value = embedding_sum / total_embeddings
                if person_name not in embeddings_dict:
                    embeddings_dict[person_name] = []
                embeddings_dict[person_name].append(avg_embedding_value)
    
    # for key,val in embeddings_dict.items():
    #     print(key,val[0].shape)
    return embeddings_dict



def preprocess_data(directory):
    # store embeddings and corresponding labels
    embeddings_dict = {}
    person_names = ["harish","harsha", "nigesh"]
    # load the images from the file system 
    for person_name in person_names:
        # print(person_name)
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                # encode images to numerical values 
                image = cv2.imread(image_path)
                if image is not None:
                    # Apply augmentations multiple times to increase diversity
                    for _ in range(5):  # Applying augmentations 5 times per image
                        augmented_image = augment_image(image)
                        embedding = extract_face_embedding(augmented_image, person_name)
                    embedding = extract_face_embedding(image, person_name)
                    if embedding is not None:
                        if person_name not in embeddings_dict:
                            embeddings_dict[person_name] = []
                        embeddings_dict[person_name].append(embedding)
    print("Preprocessing done")
    return embeddings_dict




def augment_image(image):

    # Convert OpenCV image (BGR) to PIL image for transforms
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToPILImage()(image)
    
    # Apply the augmentation transformations
    image = augmentation_transforms(image)
    
    # Convert the image back to OpenCV format (BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

# class for face recognition
class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # embedding model
        self.embedding_net = InceptionResnetV1(pretrained='vggface2').eval() 

    def forward(self, x):
        return self.embedding_net(x)


def extract_face_embedding(image, name):
    #haar classifier code 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box.astype(int)
        
            # maintain bounding box within the image
            x, y = max(0, x), max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                # extract face region and preprocess
                face_region = image[y:y+h, x:x+w]
                if face_region.size > 0:
                    # resize to match model's expected input size
                    face_resized = cv2.resize(face_region, (160, 160)) 
                    # convert image to tensor values 
                    face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
                    
                    # generate embedding using Siamese network
                    siamese_net = SiameseNetwork()
                    embedding = siamese_net(face_tensor).squeeze()
                    
                    return embedding
    
    return None



# embedding comparison using cosine similarity

def recognize_face(embedding, embeddings_dict):
    if not embeddings_dict:
        return "Unknown"
    
    # Normalize the embedding
    embedding = embedding / embedding.norm()
    
    max_similarity = -1
    recognized_label = "Unknown"
    threshold = 0.65  # Cosine similarity threshold
    
    for label, embeddings in embeddings_dict.items():
        # Normalize each stored embedding and calculate cosine similarity
        similarities = [cosine_similarity(embedding, emb / emb.norm(), dim=0) for emb in embeddings]
        max_label_similarity = max(similarities)
        
        if max_label_similarity > max_similarity:
            max_similarity = max_label_similarity
            recognized_label = label
    
    return recognized_label if max_similarity > threshold else "Unknown"



# recognize face with similarity score
def recognize_face_for_image(embedding, embeddings_dict):
    if not embeddings_dict or embedding == None:
        return "Unknown", 0.0

    # Normalize the embedding
    embedding = embedding / embedding.norm()

    max_similarity = -1
    recognized_label = "Unknown"
    threshold = 0.6  # Cosine similarity threshold

    for label, embeddings in embeddings_dict.items():
        similarities = [cosine_similarity(embedding, emb / emb.norm(), dim=0).item() for emb in embeddings]
        max_label_similarity = max(similarities)

        if max_label_similarity > max_similarity:
            max_similarity = max_label_similarity
            recognized_label = label
   

    return (recognized_label, max_similarity) if max_similarity > threshold else ("Unknown", max_similarity)

    
    