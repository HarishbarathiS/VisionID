import cv2
import torch
from torchvision import transforms
from app.face_embedding import SiameseNetwork, preprocess_data, recognize_face_for_image, preprocess_data_averaging
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize the face cascade and siamese netowrk
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
siamese_net = SiameseNetwork().to('cuda' if torch.cuda.is_available() else 'cpu')



def real_time_face_recognition(embeddings):
    cap = cv2.VideoCapture(0)
    frame_skip = 1  # Process every 2nd frame to reduce lag
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        

        # Resize frame for faster processing
        scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        if frame_count % frame_skip == 0:
            # Detect faces in the downscaled grayscale frame
            boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for box in boxes:
                x, y, w, h = [int(coord / scale_factor) for coord in box]  # Scale back to original frame size
                
                # Extract and preprocess the face region
                face_region = frame[y:y+h, x:x+w]
                if face_region.size > 0:
                    face_resized = cv2.resize(face_region, (160, 160))
                    face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
                    face_tensor = face_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

                    # Generate embedding
                    with torch.no_grad():
                        embedding = siamese_net(face_tensor).squeeze()
                    
                    # Recognize face
                    recognized_label, similarity = recognize_face_for_image(embedding, embeddings)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame,f"{recognized_label} ({similarity:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        # Display frame
        cv2.imshow('Real-time Face Recognition', frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Directory with face images
    dir = "."
    
    # Preprocess data 
    embeddings_dict = preprocess_data_averaging(dir)
    print(f"Loaded {len(embeddings_dict)} faces.")
    print(embeddings_dict.keys())
    real_time_face_recognition(embeddings_dict)

