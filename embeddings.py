import os
import heapq
import faiss
import cupy as cp
import pandas as pd
from gdown import download


def load_embeddings(id: str, filename: str = "embeddings.csv.gz"):
    if not os.path.exists(filename):
        download(id=id, output=filename, quiet=False)

    df = pd.read_csv(filename, header=None, engine="pyarrow")
    words = df.iloc[:, 0].tolist()
    embeddings = cp.array(df.iloc[:, 1:].to_numpy(dtype=float))
    embeddings /= cp.linalg.norm(embeddings, axis=1, keepdims=True)
    return words, cp.array(embeddings)


words, embeddings = load_embeddings("1biezzvCn3TkxRLy-7t6LA6M_bNFV8xl_")
word_to_index = {word: i for i, word in enumerate(words)}


def similarity(i: int, j: int) -> cp.ndarray:
    return cp.dot(embeddings[i], embeddings[j])


def distance(i: int, j: int):
    sim = similarity(i, j)
    if sim >= 0.5:
        return 1  # σκούρα γραμμή
    if sim >= 0.4:
        return 2  # αχνή γραμμή
    if sim >= 0.3:
        return 3  # διακεκομμένη γραμμή
    return cp.Infinity


numpy_embeddings = cp.asnumpy(embeddings)
d = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(d)
faiss_index.add(numpy_embeddings)


def get_neighbors(i: int, k: int = 100):
    """
    Returns a list of tuples (neighbor_index, cost) for the word at index i.
    Instead of iterating over every word, we query the FAISS index to get the top k nearest neighbors,
    then filter out those that don't meet our cosine similarity threshold.

    Επιστρέφει μια λίστα από tuples (index γείτονα, κόστος) για τη λέξη στο δείκτη i.
    Αντί να επαναλαμβάνει κάθε λέξη, χρησιμοποιούμε το FAISS index για να πάρουμε τους top k πλησιέστερους γείτονες,
    και μετά φιλτράρουμε αυτούς που δεν πληρούν το κατώτατο όριο ομοιότητας.
    """
    query = cp.asnumpy(cp.expand_dims(embeddings[i], axis=0))
    sims, indices = faiss_index.search(query, k)
    similarities = sims[0]
    neighbor_indices = indices[0]
    neighbors = []
    for sim, j in zip(similarities, neighbor_indices):
        if j == i:
            continue  # skip self
        if sim >= 0.5:
            cost = 1
        elif sim >= 0.4:
            cost = 2
        elif sim >= 0.3:
            cost = 3
        else:
            continue
        neighbors.append((j, cost))
    return neighbors


def find_path(from_word: str, to_word: str):
    """
    Uses Dijkstra's algorithm to find the least-cost path between two words.

    Χρησιμοποιεί τον αλγόριθμο του Dijkstra για να βρει το μονοπάτι με το μικρότερο κόστος μεταξύ δύο λέξεων.
    """
    if from_word not in word_to_index or to_word not in word_to_index:
        print("One or both of the specified words are not in the vocabulary.")
        return None, float("inf")

    source = word_to_index[from_word]
    target = word_to_index[to_word]

    # Priority queue holds (current_cost, word_index).
    heap = [(0, source)]
    dist = {source: 0}
    prev = {}

    while heap:
        cur_cost, u = heapq.heappop(heap)
        if u == target:
            break
        if cur_cost > dist[u]:
            continue
        for v, cost in get_neighbors(u):
            alt = cur_cost + cost
            if v not in dist or alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    if target not in dist:
        return None, float("inf")

    # Επανασυναρμολόγηση του path μέσω backtracking
    path = []
    u = target
    while u != source:
        path.append(u)
        u = prev[u]
    path.append(source)
    path.reverse()
    return path, dist[target]


if __name__ == "__main__":
    start_word = input("Δώσε την αρχική λέξη: ")
    end_word = input("Δώσε την τελική λέξη: ")

    path, total_cost = find_path(start_word, end_word)
    if path is None:
        print(
            "\nΔεν βρέθηκε μονοπάτι μεταξύ των λέξεων '{}' και '{}'.".format(
                start_word, end_word
            )
        )
    else:
        path_words = [words[i] for i in path]
        print("\nΜονοπάτι από '{}' σε '{}':".format(start_word, end_word))
        print(" -> ".join(path_words))
        print("Συνολικό κόστος: ", total_cost)
