#model structure configuration
model_config = dict(
    model="baseline-genre-detector",
    dense=16,
    dropout=0.2,
    batch_size=256,
    epochs=5,
    learning_rate=1e-4,
)

#column of interest of MovieLens Dataset
movielens_col_list = ['genres','original_title','overview']
