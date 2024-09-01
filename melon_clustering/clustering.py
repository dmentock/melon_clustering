import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from transformers import AutoTokenizer, AutoModel
import torch
from IPython.display import display, HTML
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import stanza

from melon_db import logging_

from clustering import CACHE_DIR

class Clustering:

    def __init__(self, lang, model_name = None, tokenizer = None, model = None):
        print("DBManager initializing...")  # Print to console for debugging
        self.log = logging_.setup_logger('Cluster', level = logging_.DEBUG)
        if model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
            self.model = model
        self.nlp = stanza.Pipeline(lang)
        self.log.debug("Cluster class initialized with model: %s", model_name)

    def generate_embedding(self, target_word, text):
        self.log.debug("Generating embeddings for target_word: %s, text: %s", target_word, text)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        # self.log.debug("Tokenized inputs: %s", inputs)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # self.log.debug("Model outputs: %s", outputs)

        last_hidden_state = outputs.last_hidden_state[0]
        sentence_embedding = last_hidden_state.mean(dim=0)
        # self.log.debug("Sentence embedding: %s", sentence_embedding)

        target_ids = self.tokenizer.encode(target_word, add_special_tokens=False)
        self.log.debug("Target ids: %s", target_ids)
        target_embeddings = []

        for target_id in set(target_ids):
            positions = (inputs.input_ids[0] == target_id).nonzero(as_tuple=True)[0]
            self.log.debug(f'target_id: {target_id}')
            self.log.debug(f'positions: {positions}')
            for position in positions:
                stride = 5
                start = max(0, position - stride)
                end = min(last_hidden_state.size(0), position + stride + 1)
                context_embedding = last_hidden_state[start:end].mean(dim=0)
                self.log.debug("Context embedding: %s", context_embedding)
                target_embeddings.append(context_embedding)

        if target_embeddings:
            target_word_embedding = torch.stack(target_embeddings).mean(dim=0)
            combined_embedding = torch.cat((sentence_embedding, target_word_embedding)).numpy()
            self.log.debug("Combined embedding: %s", combined_embedding)
            return combined_embedding
        else:
            self.log.warning(f"Warning: Target word '{target_word}' not found in sentence: {text}")

    # def generate_embedding(self, word, sentence, weight=2.0):
    #     inputs = self.tokenizer(sentence, return_tensors='pt')
    #     tokens = self.tokenizer.tokenize(sentence)
    #     word_idx = tokens.index(word) if word in tokens else None

    #     with torch.no_grad():
    #         outputs = self.model(**inputs)
    #     embeddings = outputs.last_hidden_state.squeeze(0)

    #     # Emphasize the specific word embedding
    #     if word_idx is not None:
    #         embeddings[word_idx] *= weight

    #     # Combine embeddings (mean pooling)
    #     sentence_embedding = embeddings.mean(dim=0)
    #     return sentence_embedding

    def generate_embeddings_for_sentences(self, sentences):
        embeddings = []
        for target_word, text in sentences:
            embedding = self.generate_embedding(target_word, text)
            if embedding is not None:
                embeddings.append(embedding)  # Convert tensor to numpy array

        return np.array(embeddings)

    def find_optimal_clusters(self, embeddings, max_clusters=10):
        self.log.debug("Finding optimal clusters for embeddings")
        silhouette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = gmm.fit_predict(embeddings)
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            self.log.debug("Silhouette score for %d clusters: %f", n_clusters, silhouette_avg)
            silhouette_scores.append(silhouette_avg)

        optimal_clusters = np.argmax(silhouette_scores) + 2
        self.log.debug("Optimal number of clusters: %d", optimal_clusters)
        return optimal_clusters

    def plot_clusters(self, tsne_embeddings, cluster_labels, sentences):
        self.log.debug("Plotting clusters")
        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.get_cmap("tab10", len(unique_labels))

        for i, label in enumerate(unique_labels):
            indices = cluster_labels == label
            plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1], color=colors(i), label=f'Cluster {label}')

        for i, (x, y) in enumerate(tsne_embeddings):
            plt.text(x, y, str(i))

        plt.legend()
        plt.title("t-SNE Clusters")
        plt.show()

    def cluster_sentences(self, sentences, max_clusters=10):
        self.log.debug("Clustering sentences")
        embeddings = self.generate_embeddings_for_sentences(sentences)
        print("embeddings",embeddings)
        # if np.any(np.isnan(embeddings)):
        #     self.log.warning("NaN values detected in embeddings. Please check the input sentences and target words.")
        #     print("NaN values detected in embeddings. Please check the input sentences and target words.")

        if embeddings.size > 0 and not np.any(np.isnan(embeddings)):
            n_clusters = self.find_optimal_clusters(embeddings, max_clusters)
            self.log.debug("Optimal number of clusters: %d", n_clusters)
            print(f"Optimal number of clusters: {n_clusters}")

            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = gmm.fit_predict(embeddings)

            pca = PCA(n_components=2)
            pca_embeddings = pca.fit_transform(embeddings)

            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sentences)-1))
            tsne_embeddings = tsne.fit_transform(pca_embeddings)

            normalized_embeddings = (255 * (tsne_embeddings - np.min(tsne_embeddings)) / np.ptp(tsne_embeddings)).astype(int)

            sentences_with_colors_and_clusters = []
            for i, (term, sentence) in enumerate(sentences):
                color = normalized_embeddings[i]
                hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], (color[0] + color[1]) // 2)
                cluster = cluster_labels[i]
                sentences_with_colors_and_clusters.append((hex_color, cluster, f"{term}: {sentence}"))

            sentences_with_colors_and_clusters.sort(key=lambda x: (x[1], x[0]))

            display(HTML('<br>'.join([f'<div style="display: flex; align-items: center;"><div style="width: 20px; height: 20px; background-color: {color}; margin-right: 10px;"></div>Cluster {cluster}: {sentence}</div>' for color, cluster, sentence in sentences_with_colors_and_clusters])))

            self.plot_clusters(tsne_embeddings, cluster_labels, sentences)

        else:
            self.log.error("No valid embeddings to plot.")
            print("No valid embeddings to plot.")

    def is_reflexive(self, verb, sent):
        subject = None
        potential_reflexive_pronouns = []
        for word in sent.words:
            if word.head == verb.id:
                if word.deprel == 'nsubj' and word.upos == 'PRON':
                    subject = word
                elif word.deprel in ['obj', 'obl', 'iobj'] and word.upos == 'PRON':
                    potential_reflexive_pronouns.append(word)
        for pronoun in potential_reflexive_pronouns:
            if subject and pronoun.lemma == subject.lemma:
                return True
            if subject and pronoun.feats and subject.feats:
                subject_feats = dict(item.split('=') for item in subject.feats.split('|'))
                pronoun_feats = dict(item.split('=') for item in pronoun.feats.split('|'))

                if (pronoun_feats.get('Person') == subject_feats.get('Person') and
                    pronoun_feats.get('Number') == subject_feats.get('Number')):
                    return True
        return False

    def categorize_sentences_by_reflexive(self, sentences_by_verb):
        regular = []
        reflexive = []
        for verb, sentences in sentences_by_verb.items():
            for sentence in sentences:
                print("processing sentence:", sentence)
                doc = self.nlp(sentence)
                reflexive_found = False
                for sent in doc.sentences:
                    for word in sent.words:
                        if word.text == verb and word.upos == 'VERB':
                            if self.is_reflexive(word, sent):
                                print(f"Reflexive use found in sentence: '{sentence}' with verb '{word.text}'")
                                reflexive.append(sentence)
                                reflexive_found = True
                                break
                    if reflexive_found:
                        break
                if not reflexive_found:
                    print(f"No reflexive use found in sentence: '{sentence}' with verb '{verb}'")
                    regular.append(sentence)
        return regular, reflexive