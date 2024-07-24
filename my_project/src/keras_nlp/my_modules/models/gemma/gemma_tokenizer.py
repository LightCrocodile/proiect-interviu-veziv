import sentencepiece as spm
import numpy as np
import json
import os

class GemmaTokenizer:
    def __init__(self, spm_model_path):
        # Load SentencePiece model
        self.tokenizer = spm.SentencePieceProcessor(model_file=spm_model_path)
    
    @staticmethod
    def from_pretrained(model_path):
        spm_model_path = os.path.join(model_path, 'vocabulary.spm')
        return GemmaTokenizer(spm_model_path)
    
    def encode(self, text, return_tensors=None):
        tokens = self.tokenizer.encode(text, out_type=int)
        if return_tensors == "np":
            return {"input_ids": np.array(tokens)}
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=True):
        # Decode token IDs back to text
        return self.tokenizer.decode(token_ids)
