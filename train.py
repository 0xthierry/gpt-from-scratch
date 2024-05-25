import torch

def main():
  with open('tinyshakespare.txt', 'r', encoding='utf-8') as f:
    text = f.read()
  
  chars = sorted(list(set(text)))

  # Tokenization, open AI uses byte pair encoding (tiktoken)
  # We're a building a character level tokenizer here for simplicity, but you usually use subword or something better than that
  stoi = { ch: i for i, ch in enumerate(chars) }
  itos = { i: ch for i, ch in enumerate(chars) }
  encode = lambda s: [stoi[ch] for ch in s]
  decode = lambda l: ''.join([itos[i] for i in l])

  # Storing the encoded text into a tensor
  data = torch.tensor(encode(text), dtype=torch.long)
  train_data_size = int(len(data) * 0.9)
  train_data = data[:train_data_size]
  val_data = data[train_data_size:]

  context_length = 8
  
  x = train_data[:context_length]
  y = train_data[1:context_length+1]

  for t in range(context_length):
    context = x[:t+1]
    target = y[t]
    print(f"Context: {context} -> Target: {target}")
  




if __name__ == '__main__':
  main()