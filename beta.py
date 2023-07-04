import os
import shutil
import time

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
import matplotlib.pyplot as plt

def hunter_spider(pdf_file):
    # A single 'hunter spider' opens and reads a PDF file
    document = extract_text(pdf_file)
    return document

def worker_spider(document):
    # The 'worker spider' processes the text: lemmatization and stopwords removal
    lemmatizer = WordNetLemmatizer()
    document = " ".join(
        lemmatizer.lemmatize(word)
        for word in document.lower().split()
        if word not in stopwords.words("english")
    )
    return document

if __name__ == "__main__":
    start = time.time()

    pdf_directory = "pdf"
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    num_hunter_spiders = 4
    num_worker_spiders = 4

    # Create hunter spiders pool
    hunter_pool = Pool(num_hunter_spiders)
    documents = hunter_pool.map(hunter_spider, pdf_files)

    # Create worker spiders pool
    worker_pool = Pool(num_worker_spiders)
    processed_documents = worker_pool.map(worker_spider, documents)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_documents)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    for i, cluster in enumerate(clusters):
        cluster_folder = os.path.join(pdf_directory, f"cluster_{cluster}")
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.move(pdf_files[i], os.path.join(cluster_folder, os.path.basename(pdf_files[i])))

    end = time.time()
    print(end - start)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X.toarray())
    reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)

    plt.figure(figsize=(10, 5))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters)
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='b')
    plt.title("Cluster Visualization")
    plt.show()
