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