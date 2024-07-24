import sys
import os

sys.path.append(r'D:\ardu\my_project\src')

try:
    from keras_nlp.my_modules.models.gemma.gemma_tokenizer import GemmaTokenizer
    print("Import successful")
except ImportError as e:
    print(f"Import error: {e}")
