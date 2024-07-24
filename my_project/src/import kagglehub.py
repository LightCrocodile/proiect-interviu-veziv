import kagglehub

def download_model():
    path = kagglehub.model_download("keras/gemma2/keras/gemma2_27b_en")
    print("Path to model files:", path)
    return path

if __name__ == "__main__":
    download_model()
