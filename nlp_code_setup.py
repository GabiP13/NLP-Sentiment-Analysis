# We start by loading the SST dataset
train_df_binary = pd.read_csv(
    f"{data_dir}/sst/sst_train_binary.csv"
)
dev_df_binary = pd.read_csv(
    f"{data_dir}/sst/sst_dev_binary.csv"
)

train_df_multiclass = pd.read_csv(
    f"{data_dir}/sst/sst_train_multiclass.csv"
)
dev_df_multiclass = pd.read_csv(
    f"{data_dir}/sst/sst_dev_multiclass.csv",
)




# We provide you the code to get sentence transformer embeddings

from sentence_transformers import SentenceTransformer


def get_st_embeddings(
    sentences: List[str],
    st_model: SentenceTransformer,
    batch_size: int = 32,
    device: str = "cpu",
):
    """
    Compute the sentence embedding using the Sentence Transformer model.

    Inputs:
    - sentence: The input sentence
    - st_model: SentenceTransformer model
    - batch_size: Encode in batches to avoid memory issues in case multiple sentences are passed

    Returns:
    torch.Tensor: The sentence embedding of shape [d,] (when only 1 sentence) or [n, d] where n is the number of sentences and d is the embedding dimension
    """

    st_model.to(device)
    sentence_embeddings = None

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        batch_embeddings = st_model.encode(batch_sentences, convert_to_tensor=True)
        if sentence_embeddings is None:
            sentence_embeddings = batch_embeddings
        else:
            sentence_embeddings = torch.cat(
                [sentence_embeddings, batch_embeddings], dim=0
            )

    return sentence_embeddings.to("cpu")



class GloveEmbeddings:

    def __init__(self, path="embeddings/glove.6B/glove.6B.50d.txt"):
        self.path = path
        self.vec_size = int(re.search(r"\d+(?=d)", path).group(0))
        self.embeddings = {}
        self.load()

    def load(self):
        for line in open(self.path, "r"):
            values = line.split()

            word_len = len(values) - self.vec_size

            word = " ".join(values[:word_len])
            vector_values = list(map(float, values[word_len:]))

            word = values[0]
            vector_values = list(map(float, values[-self.vec_size :]))
            vector = torch.tensor(vector_values, dtype=torch.float)
            self.embeddings[word] = vector

    def is_word_in_embeddings(self, word):
        return word in self.embeddings

    def get_vector(self, word):
        if not self.is_word_in_embeddings(word):
            return self.embeddings["unk"]
        return self.embeddings[word]

    # Use square operator to get the vector of a word
    def __getitem__(self, word):
        return self.get_vector(word)

glove_embeddings = GloveEmbeddings(
    path=f"{data_dir}/embeddings/glove.6B/glove.6B.50d.txt"
)




def get_sentence_embedding(
    sentence: str,
    word_embeddings: GloveEmbeddings,
    use_POS: bool = False,
    pos_weights: Dict[str, float] = None,
):
    """
    Compute the sentence embedding using the word embeddings.

    Inputs:
    - sentence: The input sentence
    - word_embeddings: GloveEmbeddings object
    - use_POS: Whether to use POS tagging
    - pos_weights: Dictionary containing POS weights

    Returns:
    torch.Tensor: The sentence embedding
    """

    sentence_embedding = None

    sentence = sentence.lower()
    tokens = word_tokenize(sentence)

    tokens = [token for token in tokens if word_embeddings.is_word_in_embeddings(token)]
    # stop_words = set(nltk.corpus.stopwords.words("english"))
    # tokens = [token for token in tokens if token not in stop_words]

    if use_POS:
        pos_tags = nltk.pos_tag(tokens)
        pos_tags = [tag[1] for tag in pos_tags]
        pos_weights = [pos_weights.get(tag, 0) for tag in pos_tags]
        pos_weights = torch.tensor(pos_weights)
    try:
        embeddings = [word_embeddings[token] for token in tokens]
        embeddings = torch.stack(embeddings)
    except:
        print("No embeddings found for the given sentence. Using zero embeddings.")
        embeddings = torch.zeros(50)
    if use_POS:
        sentence_embedding = torch.sum(embeddings * pos_weights.view(-1, 1), dim=0)
    else:
        sentence_embedding = torch.sum(embeddings, dim=0)


    return sentence_embedding



# Let's embed the sentences in training and validation data using Glove and Sentence Transformers

X_train_glove = torch.stack(
    [
        get_sentence_embedding(sentence, glove_embeddings, use_POS=False)
        for sentence in train_df_binary["sentence"].values
    ]
)
X_dev_glove = torch.stack(
    [
        get_sentence_embedding(sentence, glove_embeddings, use_POS=False)
        for sentence in dev_df_binary["sentence"].values
    ]
)

st_model = SentenceTransformer("all-mpnet-base-v2")
# Check if the embeddings already exist
if os.path.exists(f"{data_dir}/sst/X_train_st.pt") and os.path.exists(f"{data_dir}/sst/X_dev_st.pt"):
    X_train_st = torch.load(f"{data_dir}/sst/X_train_st.pt")
    X_dev_st = torch.load(f"{data_dir}/sst/X_dev_st.pt")
else:
    X_train_st = get_st_embeddings(train_df_binary["sentence"].values, st_model, device=DEVICE)
    X_dev_st = get_st_embeddings(dev_df_binary["sentence"].values, st_model, device=DEVICE)
    # Save the embeddings
    torch.save(X_train_st, f"{data_dir}/sst/X_train_st.pt")
    torch.save(X_dev_st, f"{data_dir}/sst/X_dev_st.pt")

y_train_binary = torch.tensor(train_df_binary["label"].values)
y_dev_binary = torch.tensor(dev_df_binary["label"].values)

y_train_multiclass = torch.tensor(train_df_multiclass["label"].values)
y_dev_multiclass = torch.tensor(dev_df_multiclass["label"].values)




# For convenience later we will store all datasets in a dictionary

datasets = {
    "binary": {
        "glove": {
            "X_train": X_train_glove,
            "X_dev": X_dev_glove,
            "y_train": y_train_binary,
            "y_dev": y_dev_binary,
        },
        "st": {
            "X_train": X_train_st,
            "X_dev": X_dev_st,
            "y_train": y_train_binary,
            "y_dev": y_dev_binary,
        },
    },
    "multiclass": {
        "glove": {
            "X_train": X_train_glove,
            "X_dev": X_dev_glove,
            "y_train": y_train_multiclass,
            "y_dev": y_dev_multiclass,
        },
        "st": {
            "X_train": X_train_st,
            "X_dev": X_dev_st,
            "y_train": y_train_multiclass,
            "y_dev": y_dev_multiclass,
        },
    },
}




# We provide you a utility function that creates a dataloader for you given the embeddings and labels

def create_dataloader(
    X_embed: torch.Tensor, y: torch.Tensor, batch_size: int = 32, shuffle: bool = True
):
    """
    Create a DataLoader from the input embeddings and labels.

    Inputs:
    - X_embed: torch.Tensor of shape (n, d) where n is the number of samples and d is the embedding dimension
    - y: torch.Tensor of shape (n,) containing the labels
    - batch_size: Batch size
    - shuffle: Whether to shuffle the data
    """

    dataset = TensorDataset(X_embed, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)





import torch
import torch.nn as nn
