from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

class ImageClassifier:
    """
    Zero-shot image classifier using CLIP (Contrastive Language-Image Pre-Training)
    """
    
    def __init__(self):
        print("Loading CLIP model for image classification...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use a smaller model for speed/memory efficiency if possible, 
        # but 'openai/clip-vit-base-patch32' is standard and good quality
        self.model_id = "openai/clip-vit-base-patch32"
        
        try:
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            print(f"CLIP model loaded on {self.device}")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise e

        # Define the activity labels we want to classify
        # These natural language descriptions help CLIP understand the context
        self.activity_labels = [
            "a photo of a person walking",
            "a photo of a person walking upstairs", 
            "a photo of a person walking downstairs",
            "a photo of a person sitting",
            "a photo of a person standing",
            "a photo of a person laying down"
        ]
        
        # Map descriptions back to our standard keys
        self.label_map = {
            "a photo of a person walking": "WALKING",
            "a photo of a person walking upstairs": "WALKING_UPSTAIRS",
            "a photo of a person walking downstairs": "WALKING_DOWNSTAIRS",
            "a photo of a person sitting": "SITTING",
            "a photo of a person standing": "STANDING",
            "a photo of a person laying down": "LAYING"
        }

    def predict(self, image_file):
        """
        Predict activity from an image file stream
        """
        try:
            # Load image
            image = Image.open(image_file)
            
            # Process inputs
            inputs = self.processor(
                text=self.activity_labels, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)  # softmax to get probabilities
            
            # Get predicted class
            predicted_idx = probs.argmax().item()
            predicted_label_desc = self.activity_labels[predicted_idx]
            predicted_activity = self.label_map[predicted_label_desc]
            confidence = probs[0][predicted_idx].item()
            
            # Format all probabilities
            probabilities = {}
            for idx, label_desc in enumerate(self.activity_labels):
                activity_name = self.label_map[label_desc]
                probabilities[activity_name] = float(probs[0][idx])
                
            return predicted_activity, confidence, probabilities
            
        except Exception as e:
            print(f"Error during image prediction: {e}")
            raise e

# specific instance for usage
image_classifier = None

def get_image_classifier():
    global image_classifier
    if image_classifier is None:
        image_classifier = ImageClassifier()
    return image_classifier

if __name__ == "__main__":
    # Test block
    try:
        classifier = get_image_classifier()
        print("Classifier initialized successfully")
    except Exception as e:
        print(f"Initialization failed: {e}")
