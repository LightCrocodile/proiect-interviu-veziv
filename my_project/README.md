# Proiect Model de Învățare Automată

## Descriere

Acest proiect implementează un model de învățare automată care utilizează un model pre-antrenat pentru a genera oferte bazate pe cerințele clienților. Proiectul include scripturi pentru antrenarea modelului și generarea de oferte folosind un tokenizer personalizat `GemmaTokenizer`. Modelul este bazat pe [Gemma-2].

## Cerințe

- Python 3.8 sau mai recent
- `transformers` - Biblioteca Hugging Face Transformers
- `torch` - Biblioteca PyTorch
- `tensorflow` - Biblioteca TensorFlow (dacă este necesară pentru alte scripturi)
- `sentencepiece` - Biblioteca SentencePiece
- Alte dependențe: vezi `requirements.txt`

## Instalare

1. **Clonează repository-ul:**

   ```bash
   git clone https://github.com/username/repository.git
   cd repository
   ```

2. **Creează și activează un mediu virtual:**

   ```bash
   python -m venv venv
   # Pe Windows
   venv\Scripts\activate
   # Pe macOS/Linux
   source venv/bin/activate
   ```

3. **Instalează dependențele:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Asigură-te că ai datele necesare:**

   - Modelul pre-antrenat Gemma-2
   - Datele pentru antrenament

## Utilizare

### Antrenare Model

Pentru a antrena modelul, rulează scriptul `train_model.py`:


python src/train_model.py

Asigură-te că datele de antrenament sunt plasate în directorul `./data/Oferte test` și că modelul pre-antrenat se află în directorul specificat.

### Generare Oferte

Pentru a genera o ofertă pe baza cerințelor clientului, folosește scriptul `generate_offer.py`:

python src/generate_offer.py

Asigură-te că tokenizerul și modelul necesar se află în directorul specificat.

