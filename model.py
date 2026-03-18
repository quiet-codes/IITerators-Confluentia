from sentence_transformers import SentenceTransformer

# Specify the model name from HuggingFace Model Hub
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Change to any other model if needed, e.g., "paraphrase-multilingual-MiniLM-L12-v2"

# Specify the local folder name to save the model
SAVE_PATH = "C:\Users\Rishabh Mishra\Desktop\Confluentia"  # Change as needed

# Download and save the model
model = SentenceTransformer(MODEL_NAME)
model.save(SAVE_PATH)

print(f"Model '{MODEL_NAME}' has been downloaded and saved to '{SAVE_PATH}'")