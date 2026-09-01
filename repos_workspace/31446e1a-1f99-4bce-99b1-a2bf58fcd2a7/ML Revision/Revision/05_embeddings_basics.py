"""
EMBEDDINGS BASICS
===================
Interview must-knows:
- An embedding is a learned, DENSE, LOW-dimensional vector representation of a
  discrete object (word, user, item, category) that captures semantic
  similarity as GEOMETRIC similarity -- similar things end up close together
  in the embedding space.
- Contrast with one-hot encoding: one-hot is sparse, high-dimensional
  (vocab_size), and every pair of distinct items is EQUALLY far apart (no
  notion of similarity baked in). Embeddings are dense, low-dimensional
  (e.g. 100-1000 dims regardless of vocab size), and place related items near
  each other.
- Distributional hypothesis (for word embeddings): "a word is characterized by
  the company it keeps" -- words appearing in similar CONTEXTS get similar
  embeddings, purely from co-occurrence statistics, with no manual labeling.
- word2vec (classic example) has two training framings:
    Skip-gram: given a center word, predict surrounding context words.
    CBOW (Continuous Bag of Words): given context words, predict the center word.
  Both are trained as a (simplified) classification task, and the learned
  weight MATRIX itself (not the classifier's final predictions) is the useful
  artifact -- each row becomes a word's embedding vector.
- Similarity metric: COSINE similarity = (a.b)/(||a||*||b||) is preferred over
  raw Euclidean distance for embeddings because it ignores vector magnitude
  and only measures DIRECTION -- important since embedding magnitude often
  correlates with word frequency rather than meaning.
- Famous property (word2vec): analogies work via vector arithmetic, e.g.
  vector("king") - vector("man") + vector("woman") ~= vector("queen") --
  because the embedding space linearly encodes certain semantic/relational
  directions.
- Modern usage: embedding LAYERS are a standard first layer in NNs for any
  categorical/discrete input (an embedding layer is literally just a learnable
  lookup table, trained end-to-end with the rest of the network via
  backprop) -- used for words, user IDs, product IDs, categorical features in
  tabular deep learning, etc. Pretrained embeddings (word2vec/GloVe, or
  today, embeddings from large transformer models) can be used directly or
  fine-tuned.
- Embeddings also power retrieval/semantic search and RAG systems: encode a
  query and a corpus of documents into the same vector space, retrieve by
  nearest-neighbor (cosine) similarity instead of exact keyword match.
"""

import numpy as np

# -----------------------------------------------------------------
# 1. One-hot vs embedding: dimensionality and (lack of) similarity structure
# -----------------------------------------------------------------
vocab = ["king", "queen", "man", "woman", "apple", "banana"]
vocab_size = len(vocab)

one_hot = np.eye(vocab_size)
print(f"One-hot vectors: dim={vocab_size} (grows with vocab size)")
king_oh, queen_oh = one_hot[0], one_hot[1]
print("Cosine similarity(king, queen) in one-hot space:",
      round(king_oh @ queen_oh / (np.linalg.norm(king_oh) * np.linalg.norm(queen_oh)), 3),
      " <- always 0 for any two distinct one-hot vectors, no notion of similarity")

# -----------------------------------------------------------------
# 2. Toy skip-gram-STYLE training from scratch: learn embeddings purely from
#    co-occurrence in tiny "sentences" (illustrates the core mechanic).
# -----------------------------------------------------------------
sentences = [
    "the king rules the kingdom",
    "the queen rules the kingdom",
    "the man walked to the market",
    "the woman walked to the market",
    "i ate an apple today",
    "i ate a banana today",
]
tokens = [s.split() for s in sentences]
words = sorted(set(w for sent in tokens for w in sent))
word2idx = {w: i for i, w in enumerate(words)}
V = len(words)

def make_skipgram_pairs(tokens, window=2):
    pairs = []
    for sent in tokens:
        for i, center in enumerate(sent):
            for j in range(max(0, i - window), min(len(sent), i + window + 1)):
                if j != i:
                    pairs.append((word2idx[center], word2idx[sent[j]]))
    return pairs

pairs = make_skipgram_pairs(tokens)

EMBED_DIM = 8
rng = np.random.default_rng(0)
W_in = rng.normal(scale=0.1, size=(V, EMBED_DIM))     # "center word" embedding table
W_out = rng.normal(scale=0.1, size=(V, EMBED_DIM))    # "context word" embedding table

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

lr = 0.05
for epoch in range(300):
    total_loss = 0
    for center_idx, context_idx in pairs:
        h = W_in[center_idx]                            # look up center embedding
        scores = W_out @ h                               # score against every word
        probs = softmax(scores)
        total_loss += -np.log(probs[context_idx] + 1e-9)

        # gradient of cross-entropy w.r.t. scores is (probs - one_hot(context))
        grad_scores = probs.copy()
        grad_scores[context_idx] -= 1
        grad_W_out = np.outer(grad_scores, h)
        grad_h = W_out.T @ grad_scores

        W_out -= lr * grad_W_out
        W_in[center_idx] -= lr * grad_h
    if epoch % 100 == 0:
        print(f"epoch {epoch}: avg loss={total_loss/len(pairs):.3f}")

def cosine_sim(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

print("\nLearned embedding cosine similarities (should reflect similar CONTEXTS):")
pairs_to_check = [("king", "queen"), ("man", "woman"), ("apple", "banana"), ("king", "banana")]
for w1, w2 in pairs_to_check:
    sim = cosine_sim(W_in[word2idx[w1]], W_in[word2idx[w2]])
    print(f"  cosine_sim({w1:6s}, {w2:6s}) = {sim:+.3f}")
print("(king/queen and man/woman appear in similar sentence patterns here, so "
      "they should end up with higher similarity than unrelated pairs like "
      "king/banana -- with this tiny toy corpus the signal is noisy, but the "
      "mechanism is exactly how word2vec works at scale.)")

# -----------------------------------------------------------------
# 3. Embedding layer in a network = a learnable lookup table
# -----------------------------------------------------------------
print("\nAn nn.Embedding-style layer is just: output = EmbeddingTable[input_id]")
print("It's trained end-to-end via ordinary backprop, exactly like any other "
      "weight matrix -- gradients only update the ROWS that were looked up.")

# -----------------------------------------------------------------
# 4. Nearest-neighbor retrieval via cosine similarity (semantic search idea)
# -----------------------------------------------------------------
query_word = "king"
sims = {w: cosine_sim(W_in[word2idx[query_word]], W_in[word2idx[w]])
        for w in words if w != query_word}
top3 = sorted(sims.items(), key=lambda kv: -kv[1])[:3]
print(f"\nTop-3 nearest neighbors to '{query_word}' by cosine similarity: {top3}")
print("(This nearest-neighbor-in-embedding-space lookup is the core mechanic "
      "behind semantic search / RAG document retrieval.)")

print("\nKey talking points: dense vs one-hot, distributional hypothesis, "
      "skip-gram/CBOW, cosine similarity over Euclidean (magnitude vs "
      "direction), vector arithmetic analogies, embedding layer = learnable "
      "lookup table trained via backprop, use in retrieval/RAG.")
