class FFNN(nn.Module): # We inherit from nn.Module, which is the base class for all PyTorch Neural Network modules

    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int
    ):

        """
        Define the architecture of a Feedforward Neural Network with architecture described above.

        Inputs:
        - input_dim: The dimension of the input (d according to the figure above)
        - hidden_dim: The dimension of the hidden layer (h according to the figure above)
        - num_classes: The number of classes in the classification task.

        """

        super(FFNN, self).__init__() # Call the base class constructor

        # Define your network architecture below

        self.fc1 = nn.Linear(input_dim, hidden_dim) # First linear layer
        self.act1 = nn.ReLU() # First activation function
        self.fc2 = nn.Linear(hidden_dim, 1 if num_classes == 2 else num_classes) # Second linear layer
        self.act2 = nn.Sigmoid() if num_classes == 2 else nn.Softmax() # Second activation function

        # YOUR CODE HERE
        # raise NotImplementedError()

        self.initialize_weights() # Initialize the weights of the linear layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Computes the forward pass through the network.

        Inputs:
        - x : Input tensor of shape (n, d) where n is the number of samples and d is the dimension of the input

        Hint: You can call a layer directly with the input to get the output tensor, e.g. self.fc1(x) will return the output tensor after applying the first linear layer.
        """

        # logits = None

        # YOUR CODE HERE
        # raise NotImplementedError()

        layer1_output = self.fc1(x)
        act1_output = self.act1(layer1_output)
        layer2_output = self.fc2(act1_output)
        # layer2_output = self.fc2(layer1_output)
        # act2_output = self.act2(layer2_output)

        # return logits
        # return act2_output
        return layer2_output


    def initialize_weights(self):
        """
        Initialize the weights of the linear layers.

        We initialize the weights using Xavier Normal initialization and the biases to zero.

        You can read more about Xavier Initialization here: https://cs230.stanford.edu/section/4/#xavier-initialization
        """

        for layer in self.children():
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)






class FFNN_layers(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, num_hidden: int
    ):

        super(FFNN_layers, self).__init__()
        self.hidden = num_hidden

        if num_hidden == 1:
          self.fc1 = nn.Linear(input_dim, hidden_dim)
          self.act1 = nn.ReLU()
          self.fc2 = nn.Linear(hidden_dim, 1 if num_classes == 2 else num_classes)
          self.act2 = nn.Sigmoid() if num_classes == 2 else nn.Softmax()
          self.initialize_weights()
        else:
          self.fc1 = nn.Linear(input_dim, hidden_dim)
          self.act1 = nn.ReLU()
          self.fc1h = nn.Linear(hidden_dim, hidden_dim)
          self.act1h = nn.ReLU()
          self.fc2 = nn.Linear(hidden_dim, 1 if num_classes == 2 else num_classes)
          self.act2 = nn.Sigmoid() if num_classes == 2 else nn.Softmax()
          self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if(self.hidden == 1):
          layer1_output = self.fc1(x)
          act1_output = self.act1(layer1_output)
          layer2_output = self.fc2(act1_output)
          return layer2_output
        else:
          layer1_output = self.fc1(x)
          act1_output = self.act1(layer1_output)
          h1out = self.fc1h(act1_output)
          h2out = self.act1h(h1out)
          layer2_output = self.fc2(h2out)
          return layer2_output


    def initialize_weights(self):

        for layer in self.children():
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)







def evaluate_layers(
    model: nn.Module,
    X_dev: torch.Tensor,
    y_dev: torch.Tensor,
    eval_batch_size: int = 128,
    device: str = "cpu",
) -> Dict[str, float]:

    dev_dataloader = create_dataloader(X_dev, y_dev, batch_size=eval_batch_size, shuffle=False)
    is_binary_cls = y_dev.max() == 1
    model.eval()
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss() if is_binary_cls else nn.CrossEntropyLoss()

    val_loss = 0.0
    preds = []

    with torch.no_grad():
        for X_batch, y_batch in dev_dataloader:

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            y_batch = y_batch.float() if is_binary_cls else y_batch
            labels = y_batch
            batch_loss = loss_fn(logits.squeeze(-1), labels)

            classfunc = nn.Sigmoid() if is_binary_cls else nn.Softmax()
            if is_binary_cls:
              batch_preds = classfunc(logits)
              batch_preds = torch.tensor([1 if i >= 0.5 else 0 for i in batch_preds])
            else:
              batch_preds = []
              for sample in logits:
                batch_preds.append(torch.argmax(sample))
              batch_preds = torch.tensor(batch_preds)

            preds.extend(batch_preds.cpu())

            val_loss += batch_loss.item()
    val_loss /= len(dev_dataloader)
    preds = torch.stack(preds)
    accuracy, precision, recall, f1 = get_accuracy(preds, y_dev), get_precision(preds, y_dev), get_recall(preds, y_dev), get_f1_score(preds, y_dev)

    return {
        "loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }




def get_accuracy(y_pred, y_true):

    num = 0
    for i in range(len(y_pred)):
      if y_pred[i].item() == y_true[i].item(): num += 1
    return float(num/len(y_pred))


def get_precision(y_pred, y_true, epsilon=1e-8):

    n = len(y_pred)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    tp = 0
    fp = 0
    for i in range(n):
      num = y_pred[i] - y_true[i]
      num2 = y_pred[i] + y_true[i]
      if num == 1:
        fp += 1
      elif num2 == 2:
        tp += 1

    return tp / (tp + fp + epsilon)


def get_recall(y_pred, y_true, epsilon=1e-8):

    n = len(y_pred)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()


    tp = 0
    fn = 0
    for i in range(n):
      num = y_true[i] - y_pred[i]
      num2 = y_pred[i] + y_true[i]
      if num == 1:
        fn += 1
      elif num2 == 2:
        tp += 1

    return tp / (tp + fn + epsilon)


def get_f1_score(y_pred, y_true, epsilon=1e-8):

    prec = get_precision(y_pred, y_true)
    rec = get_recall(y_pred, y_true)
    return 2 * prec * rec / (prec + rec + epsilon)




def train_layers(
    model: nn.Module,
    X_train_embed: torch.Tensor,
    y_train: torch.Tensor,
    X_dev_embed: torch.Tensor,
    y_dev: torch.Tensor,
    lr: float = 1e-3,
    batch_size: int = 32,
    eval_batch_size: int = 128,
    n_epochs: int = 10,
    device: str = "cpu",
    verbose: bool = True,
):

    train_dataloader = create_dataloader(X_train_embed, y_train, batch_size=batch_size, shuffle=True)

    is_binary_cls = y_train.max() == 1

    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss() if is_binary_cls else nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr)

    train_losses = []
    dev_metrics = []

    for epoch in range(n_epochs):

        model.train()
        train_epoch_loss = 0.0
        for X_batch, y_batch in train_dataloader:

            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.float() if is_binary_cls else y_batch

            logits = model(X_batch)
            labels = y_batch

            batch_loss = loss_fn(logits.squeeze(-1), labels)
            batch_loss.backward()

            optimizer.step()

            train_epoch_loss += batch_loss.item()

        train_epoch_loss /= len(train_dataloader)
        train_losses.append(train_epoch_loss)

        eval_metrics = evaluate_layers(model, X_dev_embed, y_dev, eval_batch_size=eval_batch_size, device=device)
        dev_metrics.append(eval_metrics)

        if verbose:
            print("Epoch: %.d, Train Loss: %.4f, Dev Loss: %.4f, Dev Accuracy: %.4f, Dev Precision: %.4f, Dev Recall: %.4f, Dev F1: %.4f" % (epoch + 1, train_epoch_loss, eval_metrics["loss"], eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1"]))

    return train_losses, dev_metrics







loss_fn = nn.CrossEntropyLoss()  # Define the loss function







from torch.optim import Adam

torch.manual_seed(42)









def evaluate(
    model: nn.Module,
    X_dev: torch.Tensor,
    y_dev: torch.Tensor,
    eval_batch_size: int = 128,
    device: str = "cpu",
) -> Dict[str, float]:

    """
    Evaluates the model's loss on the validation set as well as accuracy, precision, and recall scores.

    Inputs:
    - model: The FFNN model
    - X_dev: The sentence embeddings of the validation data
    - y_dev: The labels of the validation data
    - eval_batch_size: Batch size for evaluation

    Returns:
    - Dict[str, float]: A dictionary containing the loss, accuracy, precision, recall, and F1 scores
    """

    # Create a DataLoader for the validation data
    dev_dataloader = create_dataloader(X_dev, y_dev, batch_size=eval_batch_size, shuffle=False) # Note that we don't shuffle the data for evaluation.

    # A flag to check if the classification task is binary
    is_binary_cls = y_dev.max() == 1

    # Set the model to evaluation mode. Read more about why we need to do this here: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    model.eval()

    # Transfer the model to device
    model.to(device)

    # TODO: Define the loss function. Remember to use BCEWithLogitsLoss for binary classification and CrossEntropyLoss for multiclass classification
    ####### torch.nn.BCEWithLogitsLoss for the binary case and torch.nn.CrossEntropyLoss for the multiclass case.
    # YOUR CODE HERE
    # raise NotImplementedError()
    loss_fn = nn.BCEWithLogitsLoss() if is_binary_cls else nn.CrossEntropyLoss()

    val_loss = 0.0
    preds = [] # List to store the predictions. This will be used to compute the accuracy, precision, and recall scores.

    with torch.no_grad(): # This is done to prevent PyTorch from storing gradients, which we don't need during evaluation (which saves a lot of memory and computation)
        for X_batch, y_batch in dev_dataloader: # Iterate over the batches of the validation data

            # TODO: Perform a forward pass through the network and compute loss

            # labels = y_batch

            # print("here is labels in eval", type(labels[0].item()))
            # print("here is logits in eval", type(logits[0].item()))

            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Transfer the data to device
            logits = model(X_batch)
            y_batch = y_batch.float() if is_binary_cls else y_batch # Convert the labels to float if binary classification
            labels = y_batch
            batch_loss = loss_fn(logits.squeeze(-1), labels)
            # newlabels = torch.tensor(labels, dtype=torch.float64)
            # YOUR CODE HERE
            # raise NotImplementedError()

            # TODO: Compute the predictions and store them in the preds list.
            # Remember to apply a sigmoid function to the logits if binary classification and argmax if multiclass classification
            # For binary classification, you can use a threshold of 0.5.
            classfunc = nn.Sigmoid() if is_binary_cls else nn.Softmax()
            # batch_preds = classfunc(logits)
            # print(type(batch_preds))
            if is_binary_cls:
              batch_preds = classfunc(logits)
              batch_preds = torch.tensor([1 if i >= 0.5 else 0 for i in batch_preds])
            else:
              # batch_preds = nn.Softmax(logits)
              batch_preds = []
              for sample in logits:
                batch_preds.append(torch.argmax(sample))
              batch_preds = torch.tensor(batch_preds)


            # YOUR CODE HERE
            # raise NotImplementedError()

            preds.extend(batch_preds.cpu())

            val_loss += batch_loss.item() # Accumulate the loss. Note that we use .item() to extract the loss value from the tensor.
    val_loss /= len(dev_dataloader) # Compute the average loss
    preds = torch.stack(preds) # Convert the list of predictions to a tensor

    # TODO: Compute the accuracy, precision, and recall scores
    accuracy, precision, recall, f1 = get_accuracy(preds, y_dev), get_precision(preds, y_dev), get_recall(preds, y_dev), get_f1_score(preds, y_dev)

    # YOUR CODE HERE
    # raise NotImplementedError()


    return {
        "loss": val_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }




def get_accuracy(y_pred, y_true):

    num = 0
    for i in range(len(y_pred)):
      if y_pred[i].item() == y_true[i].item(): num += 1
    return float(num/len(y_pred))


def get_precision(y_pred, y_true, epsilon=1e-8):

    n = len(y_pred)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    tp = 0
    fp = 0
    for i in range(n):
      num = y_pred[i] - y_true[i]
      num2 = y_pred[i] + y_true[i]
      if num == 1:
        fp += 1
      elif num2 == 2:
        tp += 1

    return tp / (tp + fp + epsilon)


def get_recall(y_pred, y_true, epsilon=1e-8):

    n = len(y_pred)
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()


    tp = 0
    fn = 0
    for i in range(n):
      num = y_true[i] - y_pred[i]
      num2 = y_pred[i] + y_true[i]
      if num == 1:
        fn += 1
      elif num2 == 2:
        tp += 1

    return tp / (tp + fn + epsilon)



def get_f1_score(y_pred, y_true, epsilon=1e-8):

    prec = get_precision(y_pred, y_true)
    rec = get_recall(y_pred, y_true)
    return 2 * prec * rec / (prec + rec + epsilon)






def train(
    model: nn.Module,
    X_train_embed: torch.Tensor,
    y_train: torch.Tensor,
    X_dev_embed: torch.Tensor,
    y_dev: torch.Tensor,
    lr: float = 1e-3,
    batch_size: int = 32,
    eval_batch_size: int = 128,
    n_epochs: int = 10,
    device: str = "cpu",
    verbose: bool = True,
):

    """
    Runs the training loop for `n_epochs` epochs.

    Inputs:
    - model: The FFNN model to be trained
    - X_train_embed: The sentence embeddings of the training data
    - y_train: The labels of the training data
    - X_dev_embed: The sentence embeddings of the validation data
    - y_dev: The labels of the validation data
    - lr: Learning rate for the optimizer
    - n_epochs: Number of epochs to train the model

    Returns:
    - train_losses: List of training losses for each epoch
    - dev_metrics: List of validation metrics (loss, accuracy, precision, recall, f1) for each epoch
    """

    # Create a DataLoader for the training data
    train_dataloader = create_dataloader(X_train_embed, y_train, batch_size=batch_size, shuffle=True)

    # A flag to check if the classification task is binary
    is_binary_cls = y_train.max() == 1

    # Transfer the model to device
    model.to(device)

    # TODO: Define the loss function. Remember to use BCEWithLogitsLoss for binary classification and CrossEntropyLoss for multiclass classification
    ####### torch.nn.BCEWithLogitsLoss for the binary case and torch.nn.CrossEntropyLoss for the multiclass case.
    # YOUR CODE HERE
    # raise NotImplementedError()
    loss_fn = nn.BCEWithLogitsLoss() if is_binary_cls else nn.CrossEntropyLoss()

    # TODO: Define the optimizer
    optimizer = Adam(model.parameters(), lr)
    # YOUR CODE HERE
    # raise NotImplementedError()

    train_losses = [] # List to store the training losses
    dev_metrics = [] # List to store the validation metrics

    for epoch in range(n_epochs): # Iterate over the epochs

        model.train() # Set the model to training mode
        train_epoch_loss = 0.0
        for X_batch, y_batch in train_dataloader: # Iterate over the batches of the training data

            optimizer.zero_grad()  # This is done to zero-out any existing gradients stored from previous steps
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) # Transfer the data to device
            y_batch = y_batch.float() if is_binary_cls else y_batch # Convert the labels to float if binary classification
            # TODO: Perform a forward pass through the network and compute loss

            logits = model(X_batch)
            labels = y_batch

            # print("here is labels in train", type(labels[0].item()))
            # print("here is logits in train", type(logits[0].item()))
            # classfunc = nn.Sigmoid() if is_binary_cls else nn.Softmax()
            # logits = classfunc(logits)

            batch_loss = loss_fn(logits.squeeze(-1), labels)
            # YOUR CODE HERE
            # raise NotImplementedError()

            # TODO: Perform a backward pass and update the weights
            # YOUR CODE HERE
            # raise NotImplementedError()
            batch_loss.backward()

            # TODO: Perform a step of optimization
            # YOUR CODE HERE
            # raise NotImplementedError()
            optimizer.step()

            train_epoch_loss += batch_loss.item()

        train_epoch_loss /= len(train_dataloader)
        train_losses.append(train_epoch_loss)

        eval_metrics = evaluate(model, X_dev_embed, y_dev, eval_batch_size=eval_batch_size, device=device)
        dev_metrics.append(eval_metrics)

        if verbose:
            print("Epoch: %.d, Train Loss: %.4f, Dev Loss: %.4f, Dev Accuracy: %.4f, Dev Precision: %.4f, Dev Recall: %.4f, Dev F1: %.4f" % (epoch + 1, train_epoch_loss, eval_metrics["loss"], eval_metrics["accuracy"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1"]))
            # print(f"Epoch: {epoch + 1}, Train Loss: {train_epoch_loss}, Dev Loss: {eval_metrics['loss']}, Dev Accuracy: {eval_metrics['accuracy']}, Dev Precision: {eval_metrics['precision']}, Dev Recall: {eval_metrics['recall']}, Dev F1: {eval_metrics['f1']}")

    return train_losses, dev_metrics
