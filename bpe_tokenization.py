# Tokenization - Byte Pair Encoding Algorithm

def get_stats(id_list):
    count_dict = {}
    for tok1, tok2 in zip(id_list, id_list[1:]):
        count_dict[(tok1, tok2)] = count_dict.get((tok1, tok2), 0) + 1
        count_dict_sorted = sorted(count_dict, key=lambda x: count_dict[x])
        #count_dict_sorted = sorted(((v, k) for k, v in count_dict.items()), reverse=True) alternate method to sort
        top_pair = count_dict_sorted[-1]
        # top_pair = max(count_dict, key=count_dict.get) # alternate method to get top pair
        top_pair_chr = (chr(top_pair[0]), chr(top_pair[1]))
    return count_dict, top_pair, top_pair_chr

def add_to_vocab(v_set, top_pair):
    mint_token_id = max(v_set) + 1
    top_pair_text = v_set[top_pair[0]] + v_set[top_pair[1]]
    v_set[mint_token_id] = top_pair_text
    return v_set, mint_token_id

def merge(id_list, top_pair, new_minted_id):
    new_tokens = []
    i = 0
    while i < len(id_list):
        if i < len(id_list)-1 and (id_list[i], id_list[i+1]) == top_pair:
            #print(tokens[:i+2])
            new_tokens.append(new_minted_id)
            #print(new_tokens)
            i += 2
        else:
            new_tokens.append(id_list[i])
            i += 1

    return new_tokens

# Decoding
# Given a sequence of integers in range [0, vocab_size], what is the text?
# note that UTF8 is multibyte. 
# When predicting or reading bytes we can come across Partial Data (like in Streaming)
# When reading from a network or serial port, you might have received only half of a multi-byte character, 
# leading to a decoding error on the next read. Use errors="replace" or "ignore": 
# We don't need to read every character perfectly, (not errors = "strict" default)
# we can bypass the error by replacing invalid characters with a marker (?).
def decode(ids):
    # given id (list of integers), return string
    tokens = (b"".join(vocab[i] for i in ids)) # raw bytes
    text = tokens.decode("utf-8", errors="replace") # throws an error as the start byte does not follow unicode rules
    return text

def encode(text):
    # given a string return list of integers (tokens)
    bytes_seq = list(text.encode("utf-8")) 
    tokens = list(bytes_seq)
    '''##### option1: method start 
    for k, v in merges.items(): # in Python3.7 and later, dict items are guaranteed to be sorted by insert order
        tokens = merge(tokens, k, v)
    ##### option1: method end'''
    ##### option2: method start 
    while len(tokens) >= 2: # otherwise decode will not work for one char strings
        stats, _, _ = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    ##### option2: method end
    return tokens

# this is tokenizatio training
with open('Unicode.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print('length of input - number of characters', len(text))
print('initial 15 characters in input: ', text[:15])

token_bytes = text.encode('utf-8')
tokens = list(map(int, token_bytes)) # map ---> maps integers to bytes
vocab = {ix: bytes([ix]) for ix in range(256)} # uses function int to convert bytes to integer representation
print("Vocabulary", vocab_set)
print(f"-----Text - length {len(text)}------")
print(f"-----Tokens - length {len(tokens)}----")
print("---ascii characters are one byte, special characters are upto 4 bytes (variable length encoding in utf-8)---")

final_vocab_size = 276
num_iters = final_vocab_size - len(vocab) # hyperparameter
ids = list(tokens) # a copy operation
merges = {}
for j in range(num_iters):
    count_dict, top_p, top_pair_chr = get_stats(ids)
    #print("count_dict", count_dict)
    print("top_pair", count_dict[top_p], top_p, top_pair_chr)
    
    vocab, new_minted_id = add_to_vocab(vocab, top_p)
    #print("----------new vocab_set----------", v_set)
    print("new_minted_id", new_minted_id, vocab[new_minted_id])
    ids = merge(ids, top_p, new_minted_id)
    merges[top_p] = new_minted_id
    print(f"---------Tokens - length {len(ids)}---------")
    print(f"merging {top_p} inot a new token {new_minted_id}")
    #print("".join(vocab_set[i] for i in new_tokens))

