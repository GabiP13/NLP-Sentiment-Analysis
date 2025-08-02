%%bash

if [[ ! -d "./data" ]]
then
  echo "Downloading files if missing"
  git clone https://github.com/kabirahuja2431/CSE447-547MAutumn2024.git
  cp -r ./CSE447-547MAutumn2024/"Project 2"/data .
  cp ./CSE447-547MAutumn2024/"Project 2"/wordvec_tests.py .
  cp ./CSE447-547MAutumn2024/"Project 2"/nn_tests.py .
  cp ./CSE447-547MAutumn2024/"Project 2"/glove.py .
  cp ./CSE447-547MAutumn2024/"Project 2"/siqa.py .
  wget https://homes.cs.washington.edu/~kahuja/cse447/project2/glove.6B.50d.txt -O data/embeddings/glove.6B/glove.6B.50d.txt
  wget https://homes.cs.washington.edu/~kahuja/cse447/project2/X_train_st.pt -O data/sst/X_train_st.pt
  wget https://homes.cs.washington.edu/~kahuja/cse447/project2/X_dev_st.pt -O data/sst/X_dev_st.pt
  wget https://homes.cs.washington.edu/~kahuja/cse447/project2/train_data_embedded.pt -O data/socialiqa-train-dev/train_data_embedded.pt
  wget https://homes.cs.washington.edu/~kahuja/cse447/project2/dev_data_embedded.pt -O data/socialiqa-train-dev/dev_data_embedded.pt
fi





%%bash
# Install required packages
pip install pandas
pip install sentence-transformers





import os
import json
import re
from typing import List, Tuple, Dict, Union
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import Adam

from nn_tests import (
    Exercise1Runner,
    Exercise2Runner,
    Exercise3Runner,
    Exercise4Runner,
    Exercise5Runner,
    Exercise6Runner,
)

nltk.download("punkt")
nltk.download('punkt_tab')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE} device")





parent_dir = os.path.dirname(os.path.abspath("__file__"))
data_dir = os.path.join(parent_dir, "data")
