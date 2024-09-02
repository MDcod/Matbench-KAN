# Все предсказания кроме phonons сделаны на следующих гиперпараметрах
```
self.kan = KAN(width=[X_train.shape[1], 9, 1], grid=5, k=3, seed=0)
self.kan.fit(dataset, steps=20, lamb=0.001, lamb_entropy=2.7876383801674827)
```

# Предсказания phonons на этих
```
self.kan = KAN(width=[X_train.shape[1], 20, 1], grid=5, k=3, seed=0)
self.kan.fit(dataset, steps=20, lamb=0.01, lamb_entropy=2.7876383801674827)
```