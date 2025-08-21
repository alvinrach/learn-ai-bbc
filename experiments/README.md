# Experiments

## 1. Post Padding

Can we use post padding? Yes. But dont forget to add ```mask_zero=True```

Post padding is making

```
(array([[9219,  673, 5020, ...,    0,    0,    0],
        [ 415, 2301, 3851, ...,    0,    0,    0],
        [ 668,  750, 1279, ...,    0,    0,    0],
        ...,
        [ 342, 2947,  267, ...,    0,    0,    0],
        [1567,   35, 1342, ...,    0,    0,    0],
        [1424,  985, 1800, ...,    0,    0,    0]], dtype=int32),
```

instead of

```
array([[    0,     0,     0, ...,  1883,  1895,   556],
       [    0,     0,     0, ...,  3654, 14699, 14700],
       [    0,     0,     0, ...,    81,   564,    81],
       ...,
       [    0,     0,     0, ...,    36,     5,  1853],
       [    0,     0,     0, ...,   554,  3493,   167],
       [    0,     0,     0, ...,   579,  2884, 10704]], dtype=int32)

```
when tokenizing

## 2. Text Vectorization

Same as before but this time using different method. Still use ```mask_zero=True```

## 3. Hugginface Tokenization

This is the first step for the PyTorch migration. We use bert-base-uncased. The vocab is not the same like before as we borrow token vocab from bert this time.

## PyTorch Experimentation

In order to migrate, there are several pytorch experiments to mimic the previous tensorflow notebooks. In the archive is the messy notebook that will be the base of experiments of 1-7

### 1. First Torch Notebook: Still Use CPU

We mimic adam epsilon to 1e-7 to make it same as TF epsilon. padding_idx=1 to mimic mask_zero=True. CrossEntropyLoss() is equivalent with categorical_crossentropy from TF. Num epochs to 60, patience to 15, LSTM hidden_size is 64, embedding_dim is 500. Lr 0.001 same as the default TF one.

### 2. Torch GPU Experiment

No parameter is changed. Only this time we use GPU. Key Change:

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Move model to GPU
model = model.to(device)

# Inside training/validation loop, move data to GPU
for X_batch, y_batch in train_loader:
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    y_pred = model(X_batch)
    loss = criterion(y_pred, y_batch)
    loss.backward()
    optimizer.step()

# Same for validation
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        # ...
```

Another thing to notice is batching when predicting. With:

```
model.eval()  # important: set to evaluation mode
paddedtesttext = paddedtesttext.to(device)

# Forward pass & get predictions
with torch.no_grad():
    outputs = model(paddedtesttext)  # [batch_size, output_dim]
    preds = torch.argmax(outputs, dim=1).cpu().numpy()  # convert to numpy

# Map integer predictions to original category names
mapping = dict(enumerate(category.columns))
pred_labels = pd.Series(preds).map(mapping)

# Combine with ArticleId
answer = pd.concat([test['ArticleId'].reset_index(drop=True), pred_labels.reset_index(drop=True)], axis=1)
answer.columns = ['ArticleId', 'Category']

answer
```

It will use all dataset as batch size.

To tackle, first, free VRAM memory with:

```
torch.cuda.empty_cache()
```

And do batch prediction

```
test_dataset = TensorDataset(paddedtesttext)
test_loader = DataLoader(test_dataset, batch_size=256)

all_preds = []
model.eval()
with torch.no_grad():
    for (X_batch,) in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)

# Map integer predictions to original category names
mapping = dict(enumerate(category.columns))
pred_labels = pd.Series(all_preds).map(mapping)

# Combine with ArticleId
answer = pd.concat([test['ArticleId'].reset_index(drop=True), pred_labels.reset_index(drop=True)], axis=1)
answer.columns = ['ArticleId', 'Category']

answer
```