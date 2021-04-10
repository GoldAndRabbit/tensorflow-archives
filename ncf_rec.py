import pandas     as pd
import numpy      as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow         import keras
from tensorflow.keras   import layers
from pathlib            import Path
from zipfile            import ZipFile

movielens_data_file_url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
movielens_zipped_file   = keras.utils.get_file("ml-latest-small.zip",movielens_data_file_url,extract=False)
keras_datasets_path     = Path(movielens_zipped_file).parents[0]
movielens_dir           = keras_datasets_path / "ml-latest-small"

def load_shuffle_imdb_df():
    if not movielens_dir.exists():
        with ZipFile(movielens_zipped_file, "r") as zip:
            print("Extracting all the files now...")
            zip.extractall(path=keras_datasets_path)
            print("Done!")

    ratings_file = movielens_dir / "ratings.csv"
    df = pd.read_csv(ratings_file)
    user_ids            = df["userId"].unique().tolist()
    movie_ids           = df["movieId"].unique().tolist()
    user2user_encoded   = {x: i for i, x in enumerate(user_ids)}
    userencoded2user    = {i: x for i, x in enumerate(user_ids)}
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    df["user"]   = df["userId"].map(user2user_encoded)
    df["movie"]  = df["movieId"].map(movie2movie_encoded)
    df["rating"] = df["rating"].values.astype(np.float32)
    users_num    = len(user2user_encoded)
    movies_num   = len(movie_encoded2movie)
    min_rating   = min(df["rating"])
    max_rating   = max(df["rating"])
    print("Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}"
          .format(users_num, movies_num, min_rating, max_rating))
    df = df.sample(frac=1, random_state=2021)
    x = df[["user", "movie"]].values
    # Normalize the targets between 0 and 1. Makes it easy to train.
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    train_indices = int(0.9 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )
    params = {
        'users_num'          : users_num,
        'movies_num'         : movies_num,
        'user2user_encoded'  : user2user_encoded,
        'userencoded2user'   : userencoded2user,
        'movie2movie_encoded': movie2movie_encoded,
        'movie_encoded2movie': movie_encoded2movie,
    }
    return df, x_train, x_val, y_train, y_val, params


def plot_loss_curve(history):
    plt.grid()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def predict_rec_test(df, params):
    movie_df = pd.read_csv(movielens_dir / "movies.csv")
    user2user_encoded   = params['user2user_encoded']
    userencoded2user    = params['userencoded2user']
    movie2movie_encoded = params['movie2movie_encoded']
    movie_encoded2movie = params['movie_encoded2movie']

    user_id = df.userId.sample(1).iloc[0]
    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    print("Showing recommendations for user: {}".format(user_id))
    print("====" * 9)
    print("Movies with high ratings from user")
    print("----" * 8)
    top_movies_user = (
        movies_watched_by_user.sort_values(by="rating", ascending=False)
        .head(5)
        .movieId.values
    )
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
    for row in movie_df_rows.itertuples():
        print(row.title, ":", row.genres)

    print("----" * 8)
    print("Top 10 movie recommendations")
    print("----" * 8)
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
    for row in recommended_movies.itertuples():
        print(row.title, ":", row.genres)


class NerualCFNet(keras.Model):
    def __init__(self, users_num, movies_num, emb_size, **kwargs):
        super(NerualCFNet,self).__init__(**kwargs)
        self.num_users       = users_num
        self.num_movies      = movies_num
        self.embedding_size  = emb_size
        self.user_embedding  = layers.Embedding(
            users_num,
            emb_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias       = layers.Embedding(users_num, 1)
        self.movie_embedding = layers.Embedding(
            movies_num,
            emb_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(movies_num,1)

    def call(self, inputs):
        user_vector    = self.user_embedding(inputs[:, 0])
        user_bias      = self.user_bias(inputs[:, 0])
        movie_vector   = self.movie_embedding(inputs[:, 1])
        movie_bias     = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = dot_user_movie + user_bias + movie_bias
        return tf.nn.sigmoid(x)


df, x_train, x_val, y_train, y_val, params = load_shuffle_imdb_df()

EMB_SIZE = 32
model = NerualCFNet(params['users_num'], params['movies_num'], EMB_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(lr=0.001)
)

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=5,
    verbose=1,
    validation_data=(x_val, y_val),
)

predict_rec_test(df, params)
plot_loss_curve(history)

