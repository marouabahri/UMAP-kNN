
import seaborn as sns
import matplotlib.pyplot as plt
import umap.umap_ as umap
from skmultiflow.lazy import KNN
from skmultiflow.data.file_stream import FileStream



#input file
stream = FileStream('/PathTo/cnae9.csv', -1, 1)
stream.prepare_for_use()

# Set parameters
dim = 3 # The output dimensionality
k = 5 # The number of neighbors for kNN
windSize=1000 # The sliding window for the kNN
batch = 400 # The batch size

# Configuration of kNN and UMAP
#s='spectral'
um = umap.UMAP(random_state=42, n_components=dim, n_neighbors=15)  #, init=s
knn = KNN(n_neighbors=k, max_window_size=windSize)


size = stream.n_samples/2
n_samples = 0
corrects = 0
while n_samples < int(size):
    X, Y = stream.next_sample(batch_size=batch)

    tr_data = um.fit_transform(X)
    # s = um.embedding_
    # Learn from the first half of a batch
    b = len(X)/2 -1
    knn.partial_fit(tr_data[:int(b)], Y[:int(b)])
    # Test the second half of a batch
    # predict for the second half batch
    my_pred = knn.predict(tr_data[int(b)+1:len(X)])
    for i in range(int(b)+1, len(X)):
        if Y[i] == my_pred[i-int(b)-1]:
            corrects+=1
        n_samples+=1
    knn = knn.partial_fit(tr_data[int(b)+1:len(X)], Y[int(b)+1:len(X)])
    plt.scatter(tr_data[:, 0], tr_data[:, 1], c=[sns.color_palette()[x] for x in Y])

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
#plt.figure(figsize=(8, 8))
plt.title('Decomposition using UMAP')
plt.show()

# Displaying results
print("KNN's performance: " + str((corrects/n_samples)*100))