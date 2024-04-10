# initialization_script.py or setup.py

import os
import nltk

def download_nltk_resources():
    # Check if the NLTK data directory exists, if not, create it
    nltk_data_dir = os.path.join(os.environ.get('NLTK_DATA', ''), 'tokenizers')
    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    # Set NLTK data directory
    nltk.data.path.append(nltk_data_dir)

    # Download the 'punkt' tokenizer if not already downloaded
    if not os.path.exists(os.path.join(nltk_data_dir, 'punkt')):
        nltk.download('punkt', download_dir=nltk_data_dir)

# Run the function to download NLTK resources
download_nltk_resources()
