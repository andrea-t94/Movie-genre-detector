model_config = dict(
    model="baseline-genre-detector",
    dense=16,
    dropout=0.2,
    batch_size=256,
    epochs=5,
    learning_rate=1e-4,
)

movielens_col_list = ['genres','original_title','overview']

LOCAL_DIR = '/Users/andreatamburri/Documents/GitHub/Movie-genre-detector/'