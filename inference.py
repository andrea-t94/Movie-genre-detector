from genre_detector import baselineModel

model = baselineModel(name = "genre-detector")
model.load_best_model()
print(model.best_acc)