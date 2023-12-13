import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt
import json
from transformers import BertTokenizer, BertModel
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
import pickle
import joblib
import random

# create a TF-IDF vectorizer to convert the post text to a numerical format
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

"""with open('dataset/20news.json', 'r') as f2:
        data2 = json.load(f2)

    for post in data2:
        selftext_list.append(post[1])
        subreddit_list.append("others")"""


stop_words = stopwords.words('english')
stop_words.extend(['come', 'order', 'try', 'go', 'get', 'make', 'drink', 'plate', 'dish', 'restaurant', 'place',
                   'would', 'really', 'like', 'great', 'service', 'came', 'got', 'feel', 'just', 'like', 'want', 'know', 'even',
                   'work'])


def balance_posts(input_file, output_file):
    # Load posts from JSON file
    with open(input_file, 'r') as f:
        posts = json.load(f)

    # Count number of posts for each subreddit
    subreddit_counts = {}
    for post in posts:
        subreddit = post['subreddit']
        subreddit_counts[subreddit] = subreddit_counts.get(subreddit, 0) + 1

    # Determine smallest number of posts for any subreddit
    min_subreddit_count = min(subreddit_counts.values())

    # Create balanced subset of posts for each subreddit
    balanced_posts = {}
    for post in posts:
        subreddit = post['subreddit']
        if subreddit not in balanced_posts:
            balanced_posts[subreddit] = []
        if (len(balanced_posts[subreddit]) < min_subreddit_count) and (post['diagnostic'] == 'true'):
            balanced_posts[subreddit].append(post)

    returnlist = []

    with open("dataset/diagnostic_balanced.json", 'r') as f:
        data = json.load(f)

    for text in data["BPD"]:
        returnlist.append({'title': text['title'], 'selftext': text['selftext'], 'subreddit': text['subreddit']})

    for text in data["Anxiety"]:
        returnlist.append({'title': text['title'], 'selftext': text['selftext'], 'subreddit': text['subreddit']})

    for text in data["bipolar"]:
        returnlist.append({'title': text['title'], 'selftext': text['selftext'], 'subreddit': text['subreddit']})

    for text in data["depression"]:
        returnlist.append({'title': text['title'], 'selftext': text['selftext'], 'subreddit': text['subreddit']})

    for text in data["schizophrenia"]:
        returnlist.append({'title': text['title'], 'selftext': text['selftext'], 'subreddit': text['subreddit']})

    for text in data["mentalillness"]:
        returnlist.append({'title': text['title'], 'selftext': text['selftext'], 'subreddit': text['subreddit']})
    f.close()
    with open(output_file, 'w') as f:
        json.dump(returnlist, f)
    f.close()


def spot_negation(text, counter):
    # find the negation (supposed to work)
    list_conj = ["further", "however", "moreover", "nevertheless", "contrary", "still",
                 "yet", "bar", "barring", "except", "excepting", "excluding", "notwithstanding", "but"]
    list_negation = ["no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "without", "dont", "don't"]
    negation = False
    for i in range(counter):
        word = text[i].lower()
        if word in list_negation:
            negation = True
        if word in list_conj:
            negation = False
    return negation


def label_diagnostic_data():
    with open('dataset/mental_full.json', 'r') as f:
        data = json.load(f)
    return_list = []
    # extract selftext and subreddit values from data
    selftext_list = []
    for post in data:
        selftext_list.append({'title': post['title'], 'selftext': post['selftext'], 'subreddit': post['subreddit']})
    diagnostic_words = ['diagnosed', 'diagnose', 'diagnosis', 'diagnostic']
    emotions_spotted = False
    counter = 0
    annex = 0
    for text in selftext_list:
        for sentence in (sent_tokenize(text['title'] + ' ' + text['selftext'])):
            tokenized = (word_tokenize(sentence))
            for word in tokenized:
                if word in diagnostic_words:
                    boolean = spot_negation(tokenized, counter)
                    emotion_found = True
                    if boolean == False:
                        if {'title': text['title'], 'selftext': text['selftext'], 'subreddit': text['subreddit'], 'diagnostic': 'true'} not in return_list:
                            return_list.append({'title': text['title'], 'selftext': text['selftext'], 'subreddit': text['subreddit']})
                            annex = annex + 1
                    break
                counter += 1
            counter = 0
        emotion_found = False
    with open("dataset/diagnostic.json", "w") as f:
        try:
            json.dump(return_list, f)
        except Exception as e:
            f.write("ERROR" + "\n")
    return 1


def generate_graph_bert():
    # read in the JSON file
    with open('dataset/data_mental_disorders.json') as f:
        data = json.load(f)

    # Extract text data and subreddit labels
    text_data = [post['selftext'] for post in data]
    subreddit_labels = [post['subreddit'] for post in data]

    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Function to encode text using BERT
    def encode_text(text):
        input_ids = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
        outputs = model(torch.tensor([input_ids]))
        last_hidden_states = outputs[0].detach().numpy()
        print(last_hidden_states)
        return last_hidden_states[0]

    # Encode text data using BERT
    X = [encode_text(text) for text in text_data]

    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    # Convert embedding to Pandas dataframe
    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['subreddit'] = subreddit_labels

    # create a dictionary to map subreddit names to colors
    subreddit_color_dict = {
        'BPD': 'red',
        'Anxiety': 'blue',
        'bipolar': 'green',
        'depression': 'pink',
        'schizophrenia': 'yellow',
        'mentalillness': 'orange',
        # add more subreddits and corresponding colors as needed
    }

    # create a list of colors based on the subreddit column
    colors = [subreddit_color_dict[subreddit] for subreddit in umap_df['subreddit']]

    umap_df = umap_df.sort_values('cluster')

    # Generate UMAP plot
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.scatter(umap_df['UMAP1'], umap_df['UMAP2'], c=colors, s=0.5)
    ax.axis('off')
    plt.title('UMAP Projection of Mental Health Subreddits (BERT Vectorization)')
    plt.show()

    # save the result to a new CSV file
    umap_df.to_csv('umap_embeddings_bert.csv', index=False)


def generate_graphtfidf(input_file, add_posts = False):
    # read in the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # create a list to store the selftext and subreddit values
    selftext_list = []
    subreddit_list = []

    # iterate over the posts and extract the selftext and subreddit values
    for post in data:
        selftext_list.append(post['title'] + ' ' + post['selftext'])
        subreddit_list.append(post['subreddit'])


    if add_posts:
        with open("3000news.json", 'r') as f:
            data_news = json.load(f)
        for news in data_news:
            selftext_list.append(news)
            subreddit_list.append("others")


    # create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'selftext': selftext_list, 'subreddit': subreddit_list})

    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # fit the vectorizer to the post text and transform the data
    text_data = df['selftext']
    X = vectorizer.fit_transform(text_data)

    # create UMAP embeddings for the posts
    umap_embeddings = umap.UMAP(n_neighbors=45,
                                n_components=2,  # reduce to 2 dimensions for visualization
                                min_dist=0.1,
                                metric='cosine').fit_transform(X)


    # create a new dataframe with the UMAP embeddings and subreddit column
    umap_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2'])

    umap_df['subreddit'] = df['subreddit']
    umap_df['selftext'] = df['selftext']

    # create a dictionary to map subreddit names to colors
    subreddit_color_dict = {
        'BPD': 'red',
        'Anxiety': 'blue',
        'bipolar': 'green',
        'depression': 'pink',
        'schizophrenia': 'yellow',
        'mentalillness': 'orange',
        'others': 'purple'
        # add more subreddits and corresponding colors as needed
    }

    # create a list of colors based on the subreddit column
    colors = [subreddit_color_dict[subreddit] for subreddit in umap_df['subreddit']]


    # plot the UMAP embeddings with colors based on subreddit
    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s=0.05)
    plt.title('UMAP embeddings with subreddit colors')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def generate_graph_tfidfvector_clusters_kmean(input_file, amount_clusters = 6):
    # read in the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # create a list to store the selftext and subreddit values
    selftext_list = []
    subreddit_list = []

    # iterate over the posts and extract the selftext and subreddit values
    for post in data:
        selftext_list.append(post['title'] + ' ' + post['selftext'])
        subreddit_list.append(post['subreddit'])

    # create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'selftext': selftext_list, 'subreddit': subreddit_list})

    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # fit the vectorizer to the post text and transform the data
    text_data = df['selftext']
    X = vectorizer.fit_transform(text_data)

    # create UMAP embeddings for the posts
    reducer = umap.UMAP(n_neighbors=45,
                                n_components=2, # reduce to 2 dimensions for visualization
                                min_dist=0.1,
                                metric='cosine')
    umap_embeddings = reducer.fit_transform(X)

    kmeans = KMeans(n_clusters=amount_clusters, random_state=42)
    kmeans.fit(umap_embeddings)

    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    f.close()
    with open('models/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    f.close()

    filename = 'models/umap_model.sav'
    joblib.dump(reducer, filename)

    # create a new dataframe with the UMAP embeddings and subreddit column
    umap_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2'])
    umap_df['cluster'] = kmeans.labels_

    umap_df['subreddit'] = df['subreddit']
    umap_df['selftext'] = df['selftext']

    # create a dictionary to map subreddit names to colors
    subreddit_color_dict = {
        'BPD': 'red',
        'Anxiety': 'blue',
        'bipolar': 'green',
        'depression': 'pink',
        'schizophrenia': 'yellow',
        'mentalillness': 'orange',
        'others': 'purple'
        # add more subreddits and corresponding colors as needed
    }

    # create a list of colors based on the subreddit column
    colors = [subreddit_color_dict[subreddit] for subreddit in umap_df['subreddit']]

    umap_df = umap_df.sort_values('cluster')

    # plot the UMAP embeddings with colors based on subreddit
    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s = 0.2)
    plt.title('UMAP embeddings with subreddit colors')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

    # Create a dictionary to map cluster numbers to colors
    cluster_color_dict = {
        -1: 'gray',  # Noise points
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'yellow',
        4: 'orange',
        5: 'purple',
        6: 'brown',
        7: 'pink',
        8: 'cyan',
        9: 'magenta',
        10: 'teal',
        11: 'lime',
        12: 'gold',
        13: 'navy',
        14: 'olive',
        15: 'maroon',
        16: 'aqua',
        17: 'silver',
        18: 'fuchsia',
        19: 'indigo',
        # add more cluster numbers and corresponding colors as needed
    }

    colors = [cluster_color_dict[cluster] for cluster in umap_df['cluster']]

    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s=0.2)
    plt.title('UMAP embeddings with subreddit colors')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

    # save the result to a new CSV file
    umap_df.to_csv('umap_embeddings_clu_kmean.csv', index=False)


def count_unique_words(input):
    unique_words = set()
    with open(input, 'r') as f:
        data = json.load(f)
    for post in data:
        words = word_tokenize(post['title'] + ' ' + post['selftext'])  # Tokenize the post into words
        unique_words.update(words)  # Add the words to the set of unique words
    print(len(unique_words))
    return len(unique_words)  # Return the count of unique words


def generate_graph_tfidfvector_clusters_hdbscan(input_file):
    # read in the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # create a list to store the selftext and subreddit values
    selftext_list = []
    subreddit_list = []

    # iterate over the posts and extract the selftext and subreddit values
    for post in data:
        selftext_list.append(post['title'] + ' ' + post['selftext'])
        subreddit_list.append(post['subreddit'])

    # create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'selftext': selftext_list, 'subreddit': subreddit_list})

    vectorizer = TfidfVectorizer(stop_words=stop_words)

    # fit the vectorizer to the post text and transform the data
    text_data = df['selftext']
    X = vectorizer.fit_transform(text_data)

    # create UMAP embeddings for the posts
    reducer = umap.UMAP(
        n_neighbors=45,
        n_components=2,  # reduce to 2 dimensions for visualization
        min_dist=0.1,
        metric='cosine'
    )
    umap_embeddings = reducer.fit_transform(X)

    # perform HDBSCAN clustering
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    cluster_labels = hdbscan_clusterer.fit_predict(umap_embeddings)

    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('models/hdbscan_model.pkl', 'wb') as f:
        pickle.dump(hdbscan_clusterer, f)
    with open('models/umap_model.sav', 'wb') as f:
        joblib.dump(reducer, f)

    # create a new dataframe with the UMAP embeddings, cluster labels, and other columns
    umap_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2'])
    umap_df['cluster'] = cluster_labels
    umap_df['subreddit'] = df['subreddit']
    umap_df['selftext'] = df['selftext']

    # create a dictionary to map subreddit names to colors
    subreddit_color_dict = {
        'BPD': 'red',
        'Anxiety': 'blue',
        'bipolar': 'green',
        'depression': 'pink',
        'schizophrenia': 'yellow',
        'mentalillness': 'orange',
        'others': 'purple'
        # add more subreddits and corresponding colors as needed
    }

    # create a list of colors based on the subreddit column
    colors = [subreddit_color_dict[subreddit] for subreddit in umap_df['subreddit']]

    # plot the UMAP embeddings with colors based on subreddit
    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s=0.1)
    plt.title('UMAP embeddings with subreddit colors')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

    # Create a dictionary to map cluster numbers to colors
    cluster_color_dict = {
        -1: 'gray',  # Noise points
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'yellow',
        4: 'orange',
        5: 'purple',
        6: 'brown',
        7: 'pink',
        8: 'cyan',
        9: 'magenta',
        10: 'teal',
        11: 'lime',
        12: 'gold',
        13: 'navy',
        14: 'olive',
        15: 'maroon',
        16: 'aqua',
        17: 'silver',
        18: 'fuchsia',
        19: 'indigo',
        # add more cluster numbers and corresponding colors as needed
    }

    # Create a list of colors based on the cluster column
    colors = [cluster_color_dict.get(cluster, 'gray') for cluster in umap_df['cluster']]

    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s=0.1)
    plt.title('UMAP embeddings with cluster colors')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

    # save the result to a new CSV file
    umap_df.to_csv('umap_embeddings_clu_hdbscan.csv', index=False)


def assign_post_to_cluster(post, vectorizer_path='models/vectorizer.pkl',
                           kmeans_path='models/kmeans_model.pkl',
                           umap_path='models/umap_model.sav',
                           umap_df_path='umap_embeddings_clu_6.csv'):
    # Load vectorizer, kmeans model, umap model and umap_df.
    # preprocess the post text
    stop_words = set(stopwords.words('english'))
    preprocessed_post = ' '.join([word.lower() for word in post.split() if word.lower() not in stop_words])

    # transform the preprocessed post into a vector using the vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer_obj = pickle.load(f)
    X = vectorizer_obj.transform([preprocessed_post])

    # get the UMAP embeddings for the post vector using the pre-fit UMAP model
    umap_model_obj = joblib.load(umap_path)
    umap_embeddings = umap_model_obj.transform(X)

    # get the cluster assignment for the post vector using the pre-fit KMeans model
    kmeans_model_obj = joblib.load(kmeans_path)
    cluster_assignment = kmeans_model_obj.predict(umap_embeddings)

    # assign the post to the corresponding cluster in the UMAP dataframe
    umap_df = pd.read_csv(umap_df_path)
    cluster_df = umap_df[umap_df['cluster'] == cluster_assignment[0]]

    # Return the cluster assignment.
    return cluster_assignment[0]


def get_top_words_per_cluster(umap_df_path, vectorizer_file, kmeans_file, n_top_words):
  """
  Get the top words per cluster.

  Args:
    umap_df_path: The path to the UMAP DataFrame.
    vectorizer_file: The path to the vectorizer file.
    kmeans_file: The path to the KMeans model file.
    n_top_words: The number of top words to return.

  Returns:
    A list of lists of top words, one list per cluster.
  """

  # Load the UMAP DataFrame.
  umap_df = pd.read_csv(umap_df_path)

  # Load the vectorizer.
  vectorizer = pickle.load(open(vectorizer_file, 'rb'))

  # Load the KMeans model.
  kmeans = pickle.load(open(kmeans_file, 'rb'))

  # Get the cluster labels for each row in the UMAP DataFrame.
  cluster_labels = kmeans.predict(vectorizer.transform(umap_df['selftext']))

  # Get the top words for each cluster.
  top_words_per_cluster = []
  for cluster_label in np.unique(cluster_labels):
    top_words = vectorizer.inverse_transform(kmeans.cluster_centers_[cluster_label])
    top_words_per_cluster.append(top_words)

  # Return the list of lists of top words.
  return top_words_per_cluster


#assign_post_to_cluster("Does anyone else struggle to occupy their time in a meaningful way? Im going to look for a job soon, once my medication starts to kick. Hopefully I'll be able to keep one this time around. However I find myself becoming depressed when I have nothing to do. I have no money and only a couple of friends that work jobs so they are always busy. Does anyone have any activities or hobbies that help them get through their days? Anhedonia is a pain in the ass.")


def get_cluster_subreddit_counts(umap_df):
    cluster_subreddit_counts = umap_df.groupby(['cluster', 'subreddit']).size().reset_index(name='count')
    return cluster_subreddit_counts


def get_cluster_subreddit_percentages(umap_df):
    cluster_subreddit_percentages = umap_df.groupby(['cluster', 'subreddit']).size().reset_index(name='count')
    cluster_total_counts = cluster_subreddit_percentages.groupby('cluster')['count'].transform('sum')
    cluster_subreddit_percentages['percentage'] = cluster_subreddit_percentages['count'] / cluster_total_counts * 100
    return cluster_subreddit_percentages


def get_cluster_subreddit_list(umap_df):
    cluster_subreddit_percentages = get_cluster_subreddit_percentages(umap_df)

    # Find the subreddit with the maximum percentage in each cluster
    most_represented_subreddits = cluster_subreddit_percentages.groupby('cluster')['percentage'].idxmax()
    most_represented_subreddits = cluster_subreddit_percentages.loc[most_represented_subreddits]

    # Create a list of duplets containing the cluster number and its most represented subreddit
    cluster_subreddit_list = list(zip(most_represented_subreddits['cluster'], most_represented_subreddits['subreddit']))

    return cluster_subreddit_list


def test_assign(input, n = 100):
    with open("dataset/mental_full.json", 'r') as f:
        data = json.load(f)

    lsit_clu = get_cluster_subreddit_list(pd.read_csv('umap_embeddings_clu_kmean.csv'))


    categories_matrice = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

    random_sample = random.sample(data, n)

    for post in random_sample:
        y = assign_post_to_cluster(post['title'] + ' ' + post['selftext'], umap_df_path=input)
        if post['subreddit'] == 'BPD':
            # put the cluster number for x
            x = 5
            # y is the found cluster
            print((str(y) + ' ' + post['subreddit']))
            categories_matrice[x][y] += 1
        elif post['subreddit'] =='Anxiety':
            # put the cluster number for x
            x = 3
            # y is the found cluster
            print((str(y) + ' ' + post['subreddit']))
            categories_matrice[x][y] += 1
        elif post['subreddit'] =='bipolar':
            # put the cluster number for x
            x = 2
            # y is the found cluster
            print((str(y) + ' ' + post['subreddit']))
            categories_matrice[x][y] += 1
        elif post['subreddit'] =='depression':
            # put the cluster number for x
            x = 4
            # y is the found cluster
            print((str(y) + ' ' + post['subreddit']))
            categories_matrice[x][y] += 1
        elif post['subreddit'] =='schizophrenia':
            # put the cluster number for x
            x = 0
            # y is the found cluster
            print((str(y) + ' ' + post['subreddit']))
            categories_matrice[x][y] += 1
        elif post['subreddit'] =='mentalillness':
            # put the cluster number for x
            x = 1
            # y is the found cluster
            print((str(y) + ' ' + post['subreddit']))
            categories_matrice[x][y] += 1
    f.close()
    print(categories_matrice)


def test_assignv2(input, n=100):
    with open("dataset/data_mental_disorders.json", 'r') as f:
        data = json.load(f)

    umap_df = pd.read_csv(input)
    cluster_subreddit = get_cluster_subreddit_percentages(umap_df)

    # Get the most represented subreddit for each cluster
    most_represented_subreddits = cluster_subreddit.groupby('cluster')['count'].idxmax()
    most_represented_clusters = cluster_subreddit.loc[most_represented_subreddits, 'cluster']

    categories_matrice = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]

    random_sample = random.sample(data, n)
    for post in random_sample:
        subreddit = post['subreddit']
        x = most_represented_clusters[subreddit]

        # y is the found cluster
        y = assign_post_to_cluster(post['title'] + ' ' + post['selftext'], umap_df_path=input)
        print((str(y) + ' ' + subreddit))
        categories_matrice[x][y] += 1

    f.close()
    print(categories_matrice)


# generate_graph_tfidfvector_clusters("dataset/diagnostic_balanced.json", 6)


# print(get_cluster_subreddit_percentages(pd.read_csv('umap_embeddings_clu.csv')))


# test_assign('umap_embeddings_clu.csv', 1000)


def plot_umap_embeddings(csv_file,  kmeans_path='models/kmeans_model.pkl'):
    # Load the CSV file
    umap_df = pd.read_csv(csv_file)
    kmeans_model_obj = joblib.load(kmeans_path)

    kmeans = KMeans(n_clusters=amount_clusters, random_state=42)

    umap_df['cluster'] = kmeans_model_obj.labels_

    umap_df['subreddit'] = df['subreddit']
    umap_df['selftext'] = df['selftext']
    # Define the subreddit color dictionary
    subreddit_color_dict = {
        'BPD': 'red',
        'Anxiety': 'blue',
        'bipolar': 'green',
        'depression': 'pink',
        'schizophrenia': 'yellow',
        'mentalillness': 'orange',
        'others': 'purple'
        # Add more subreddits and corresponding colors as needed
    }

    # Assign colors based on the subreddit column
    colors = [subreddit_color_dict[subreddit] for subreddit in umap_df['subreddit']]

    # Plot the UMAP embeddings with colors based on subreddit
    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s=0.1)
    plt.title('UMAP embeddings with subreddit colors')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


def plot_kmeans_graph(umap_df = 'umap_embeddings_clu.csv', kmeans_labels = 'models/kmeans_model.pkl'):
  """
  Plots the graph of the UMAP embeddings after the k-means algorithm has been applied.

  Args:
    umap_df: The UMAP dataframe.
    kmeans_labels: The k-means labels.
    subreddit_color_dict: A dictionary that maps subreddit names to colors.

  Returns:
    None.
  """
  # create a dictionary to map subreddit names to colors
  subreddit_color_dict = {
      'BPD': 'red',
      'Anxiety': 'blue',
      'bipolar': 'green',
      'depression': 'pink',
      'schizophrenia': 'yellow',
      'mentalillness': 'orange',
      'others': 'purple'
      # add more subreddits and corresponding colors as needed
  }

  # Create a list of colors based on the subreddit column.
  colors = [subreddit_color_dict[subreddit] for subreddit in umap_df['subreddit']]

  # Plot the UMAP embeddings with colors based on subreddit.
  plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s=0.01)
  plt.title('UMAP embeddings with subreddit colors')
  plt.xlabel('UMAP 1')
  plt.ylabel('UMAP 2')
  plt.show()


def generate_graphdoc2vec(input_file):
    """
    This function generates a Doc2Vec model and saves it to a file. It also plots the UMAP embeddings with colors
    based on the subreddit.

    Args:
        input_file: The path to the JSON file containing the posts.

    Returns:
        The Doc2Vec model.
    """

    # Read in the JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create a list to store the selftext and subreddit values
    selftext_list = []
    subreddit_list = []

    # Iterate over the posts and extract the selftext and subreddit values
    for post in data:
        selftext_list.append(post['title'] + ' ' + post['selftext'])
        subreddit_list.append(post['subreddit'])

    # Create a dataframe from the selftext and subreddit lists
    df = pd.DataFrame({'selftext': selftext_list, 'subreddit': subreddit_list})

    # Create tagged documents for training the Doc2Vec model
    tagged_documents = [gensim.models.doc2vec.TaggedDocument(words=text.split(), tags=[index])
                        for index, text in enumerate(df['selftext'])]

    # Create a Doc2Vec model
    doc2vec_model = gensim.models.Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)

    # Build the vocabulary from the tagged documents
    doc2vec_model.build_vocab(tagged_documents)

    # Train the Doc2Vec model
    doc2vec_model.train(tagged_documents, total_examples=len(tagged_documents), epochs=500)

    # Save the model to a file
    doc2vec_model.save('doc2vec_model.model')

    umap_object = umap.UMAP(n_neighbors=45,
                            n_components=2,  # reduce to 2 dimensions for visualization
                            min_dist=0.1,
                            metric='cosine')

    # Get the document vectors from the trained Doc2Vec model
    doc_vectors = [doc2vec_model.infer_vector(tagged_document.words) for tagged_document in tagged_documents]

    # Fit the UMAP object to the document vectors
    umap_embeddings = umap_object.fit_transform(doc_vectors)

    # Create a new dataframe with the UMAP embeddings and subreddit column
    umap_df = pd.DataFrame(umap_embeddings, columns=['umap_1', 'umap_2'])
    umap_df['subreddit'] = df['subreddit']
    umap_df['selftext'] = df['selftext']

    # Create a dictionary to map subreddit names to colors
    subreddit_color_dict = {
        'BPD': 'red',
        'Anxiety': 'blue',
        'bipolar': 'green',
        'depression': 'pink',
        'schizophrenia': 'yellow',
        'mentalillness': 'orange',
        'others': 'purple'
        # add more subreddits and corresponding colors as needed
    }

    # Create a list of colors based on the subreddit column
    colors = [subreddit_color_dict[subreddit] for subreddit in umap_df['subreddit']]

    # Plot the UMAP embeddings with colors based on subreddit
    plt.scatter(umap_df['umap_1'], umap_df['umap_2'], c=colors, s=0.05)
    plt.title('UMAP embeddings with subreddit colors')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


    return doc2vec_model


generate_graphtfidf("dataset/diagnostic_balanced.json", True)


# generate_graph_tfidfvector_clusters_kmean("dataset/mental_full.json", 19)


# print(get_cluster_subreddit_percentages(pd.read_csv('umap_embeddings_clu_kmean.csv')))


# test_assignv2('umap_embeddings_clu_kmean.csv')


