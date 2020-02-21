from imageio import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def kmeans_clustering(X):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    return kmeans.labels_


def estimate_initial_parameters(Y, n_states):
    print("KMeans clustering")
    data = image_to_dataset(Y)
    prediction = data.assign(value=kmeans_clustering(data[['value']].values))
    X = dataset_to_image(prediction, Y.shape)

    means = np.zeros(n_states)
    stds = np.zeros(n_states)

    for s in range(n_states):
        data_s = data[prediction.value == s]
        s_mean = data_s.value.mean()
        s_std = data_s.value.std()
        means[s] = s_mean
        stds[s] = s_std
    return X, means, stds


def image_to_dataset(X):
    x = np.arange(X.shape[0], dtype=int)
    y = np.arange(X.shape[1], dtype=int)
    yy, xx = np.meshgrid(y, x)
    data = pd.DataFrame({
        'c1': xx.reshape(-1),
        'c2': yy.reshape(-1),
        'value': X.reshape(-1)
    })
    return data


def dataset_to_image(data, shape):
    pred_img = np.zeros(shape)
    pred_img[data.c1.values, data.c2.values] = data.value.values
    return pred_img


img_path = "./sol.jpg"
image = imread(img_path, as_gray=True)
Y = image/255#[20:-20, 20:300]/255

Y += np.random.normal(loc=0, scale=0.1, size=Y.shape)

plt.imshow(Y)
plt.title("Input image (Y) with noise")
plt.show()

n_clusters = 3
n_em_iterations = 10
n_map_iterations = 10

print("Initializing parameters")
X, means, stds = estimate_initial_parameters(Y, n_clusters)

plt.imshow(X)
plt.title("KMeans estimated states (X)")
plt.show()


print("Learning parameters")
X_current = X.copy()
means_current = means.copy()
stds_current = stds.copy()
sum_U = np.zeros(n_em_iterations)
for em_it in range(n_em_iterations):
    print("EM iteration {}".format(em_it))

    print("\tMAP estimation")
    sum_U_MAP = np.zeros(n_map_iterations)
    for map_it in range(n_map_iterations):
        print("\t\tMAP iteration {}".format(map_it))
        U_prior = np.zeros((n_clusters, *X.shape))
        U_conditional = np.zeros((n_clusters, *X.shape))

        for s in range(n_clusters):
            # updating the conditional energy function
            U_conditional[s, :, :] = ((Y - means[s])**2)/(2*stds[s]**2) + np.log2(stds[s])

            # updating the prior energy function
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    u_prior = 0
                    # get neighbor potentials
                    if i - 1 >= 0:
                        u_prior += int(X_current[i - 1, j] != s)/2
                    if i + 1 < X.shape[0]:
                        u_prior += int(X_current[i + 1, j] != s)/2
                    if j - 1 >= 0:
                        u_prior += int(X_current[i, j - 1] != s)/2
                    if j + 1 < X.shape[1]:
                        u_prior += int(X_current[i, j + 1] != s)/2
                    U_prior[s, i, j] = u_prior
        # aggregate energies
        U = U_prior + U_conditional
        # compute the new states
        X_current = np.argmin(U, axis=0)
        # compute the minimum energy of the new state
        min_U = np.min(U, axis=0)
        sum_U_MAP[map_it] = min_U.sum()

        if (map_it >= 3) and (sum_U_MAP[map_it - 2:map_it].std())/sum_U_MAP[map_it] < 0.0001:
            break
    # X_MAP = X_current.copy()
    sum_U[em_it] = min_U.sum()

    print("\tParameter estimation")
    P_s_y = np.zeros((n_clusters, *X_current.shape))
    for s in range(n_clusters):
        y_prob = 1/np.sqrt(2*np.pi*(stds_current[s]**2))*np.exp(-((Y - means_current[s])**2)/(2*stds_current[s]**2))
        U_s = np.zeros_like(X_current, dtype=float)
        for i in range(X_current.shape[0]):
            for j in range(X_current.shape[1]):
                u = 0
                if i - 1 >= 0:
                    u += int(X_current[i - 1, j] != s)/2
                if i + 1 < X.shape[0]:
                    u += int(X_current[i + 1, j] != s)/2
                if j - 1 >= 0:
                    u += int(X_current[i, j - 1] != s)/2
                if j + 1 < X.shape[1]:
                    u += int(X_current[i, j + 1] != s)/2
                U_s[i, j] = u
        P_s_y[s, :, :] = np.multiply(y_prob, np.exp(-U_s))
    P_y = P_s_y.sum(0)
    P_l_y = P_s_y/P_y

    for s in range(n_clusters):
        means_current[s] = np.multiply(P_l_y[s, :, :], Y).sum()/P_l_y[s, :, :].sum()
        stds_current[s] = np.sqrt(np.multiply(P_l_y[s, :, :], (Y - means_current[s])**2).sum()/P_l_y[s, :, :].sum())

    if (em_it >= 3) and (sum_U[em_it - 2:em_it].std()/sum_U[em_it] < 0.0001):
        break

plt.imshow(X_current)
plt.title("HMRF estimated states (X)")
plt.show()