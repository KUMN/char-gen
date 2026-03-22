# char-gen
This repository contains Autoregressive Char Level Generative Models and Small English Language Generative Models

1. small_lm_decoder_transformer_multigpu.py - Model to predict the next token given input sequence of tokens.
   - Trainable on multi-gpus with distributed data parallelism
   - Has same schema as seen in gpt2 model's state_dict (124M param) of Transformer blocks (Multihead self attention with residual connections, layernorm, feedforward), embedding, position encoding layers
   - Detailed notes are present in SmallLMImplementation.ipynb
   
3. bpe_tokenization.py - Tokenization - Byte Pair Encoding Algorithm
    - Detailed notes are present in TokenizationBytePairEncoding.ipynb
5. char_level_transformer_scaled.py - This is a character level generative model built on Transformers, for sequence modeling. It looks at the context of input sequence and predicts the next likely character in the sequence.
   -  This has 10M parameters and can be trained on single GPU or CPU.
   -  It uses 6 blocks of Multihead self-attention with 6 heads in each
   -  Detailed notes are present in char_level_transformer_scaled.ipynb

6. bigram.py - super simple bigram model that predicts next char based on input char
   - Trains neural network embeddings
   - Embedding representation for characters start with random initalization and slowly learned via backpropagation and gradient descent, during model training
   - Detail notes are present in makemore_MLP.ipynb

7. input.txt - Shakespear works dataset to train small language model
8. Unicode.txt - To train BPE tokenizer
9. names.txt - to train bigram model

10. References:
   - Andrej Karapathy's lecture - Reproduce GPT-2 (124M): https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=10
   - Andrej Karapathy's lecture - GPT: from scratch https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=8
   - Andrej Karapathy's lecture - makemore Part 2: MLP: https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=3&pp=iAQB- 
