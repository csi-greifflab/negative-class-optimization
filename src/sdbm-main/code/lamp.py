# Code adapted from: https://github.com/fcmr/clf-boundary-map/blob/master/lamp.py

import tqdm
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KDTree

#np.random.seed(0)
# Forced projection method as decribed in the paper "On improved projection 
# techniques to support visual exploration of multi-dimensional data sets"
def force_method(X, init='random', delta_frac=10.0, max_iter=50):
    # TODO: tSNE from scikit learn can initialize projections based on PCA
    if init == 'random':
        X_proj = np.random.rand(X.shape[0], 2)
    # TODO: something like:
    # else if init == ' PCA':
    #   X_proj = PCA(X)

    vec_dist = distance.pdist(X, 'euclidean')
    dmin = np.min(vec_dist)
    dmax = np.max(vec_dist)
    dist_matrix = distance.squareform(vec_dist)

    dist_diff = dmax - dmin

    index = np.random.permutation(X.shape[0])
    # TODO: better stopping criteria?
    # TODO: this is _slow_: consider using squared distances when possible
    #       - using sqeuclidean it is faster but results are worse
    for k in range(max_iter):
        for i in range(X_proj.shape[0]):
            #x_prime = X_proj[i]
            instance1 = index[i]
            x_prime = X_proj[instance1]
            for j in range(X_proj.shape[0]):
                instance2 = index[j]
                if instance1 == instance2:
                    # FIXME: the paper compares x\prime to q\prime, here I'm
                    # comparing only the indices
                    continue
                q_prime = X_proj[instance2]

                #if np.allclose(x_prime, q_prime):
                #    continue

                v = q_prime - x_prime
                dist_xq = distance.euclidean(x_prime, q_prime)
                delta = dist_matrix[instance1, instance2]/dist_diff - dist_xq
                # FIXME the algorithm desbribed in the paper states:
                # "move q_prime in the direction of v by a fraction of delta"
                # what is a good value for delta_frac?
                delta /= delta_frac

                X_proj[instance2] = X_proj[instance2] + v*delta

    # TODO: is normalization really necessary?
    X_proj = (X_proj - X_proj.min(axis=0)) / (X_proj.max(axis=0) - X_proj.min(axis=0))

    return X_proj

# Heavily based on lamp implementation from: 
# https://github.com/thiagohenriquef/mppy

# In my tests, this method worked reasonably well when data was normalized
# in range [0,1]. 
def lamp2d(X, num_ctrl_pts=None, delta=10.0, ctrl_pts_idx=None):
    # k: the number of control points
    # LAMP paper argues that few control points are needed. sqrt(|X|) is used
    # here as it the necessary number for other methods
    if ctrl_pts_idx is None:
        if num_ctrl_pts is None:
            k = int(np.sqrt(X.shape[0]))
        else:
            k = num_ctrl_pts
        ctrl_pts_idx = np.random.randint(0, X.shape[0], k)

    X_s = X[ctrl_pts_idx]
    Y_s = force_method(X_s, delta_frac=delta)

    X_proj = np.zeros((X.shape[0], 2))
    # LAMP algorithm
    for idx in range(X.shape[0]):
        skip = False

        # 1. compute weighs alpha_i
        alpha = np.zeros(X_s.shape[0])
        for i in range(X_s.shape[0]):
            diff = X_s[i] - X[idx] 
            diff2 = np.dot(diff, diff)
            if diff2 < 1e-4:
                # X_s[i] and X[idx] are almost the same point, so
                # project to the same point (Y_s[i]
                X_proj[idx] = Y_s[i]
                skip = True
                break
            alpha[i] = 1.0/diff2

        if skip == True:
            continue

        # 2. compute x_tilde, y_tilde
        sum_alpha = np.sum(alpha)
        x_tilde = np.sum(alpha[:, np.newaxis]*X_s, axis=0)/sum_alpha
        y_tilde = np.sum(alpha[:, np.newaxis]*Y_s, axis=0)/sum_alpha

        # 3. build matrices A and B
        x_hat = X_s - x_tilde
        y_hat = Y_s - y_tilde
        
        alpha_sqrt = np.sqrt(alpha)
        A = alpha_sqrt[:, np.newaxis]*x_hat
        B = alpha_sqrt[:, np.newaxis]*y_hat

        # 4. compute the SVD decomposition UDV from (A^T)B
        u, s, vh = np.linalg.svd(np.dot(A.T, B))
        # 5. Make M = UV

        aux = np.zeros((X.shape[1], 2))
        aux[0] = vh[0]
        aux[1] = vh[1]
        M = np.dot(u, aux)
        # 6. Compute the mapping (x - x_tilde)M + y_tilde
        X_proj[idx] = np.dot(X[idx] - x_tilde, M) + y_tilde


    X_proj = (X_proj - X_proj.min(axis=0)) / (X_proj.max(axis=0) - X_proj.min(axis=0))
    return X_proj

# FIXME: precompute kdtree?
def ilamp(data, data_proj, p, k=6, pre_kdtree=None):
    # 0. compute X_s and Y_s
    if pre_kdtree is not None:
        tree = pre_kdtree
    else:
        tree = KDTree(data_proj)

    dist, ind = tree.query(p, k=k)
    # ind is a (1xdim) array
    #ind = ind[0]
    X_proj = data_proj[ind]
    X = data[ind]

    ret = [None,]*p.shape[0]

    # 1. compute weights alpha_i
    # Haverão p alfas e cada alfa terá k pesos (a_i para 1 <= i <= k)
    alpha = np.zeros((p.shape[0], k))
    idx_removal = []
    for j in range(p.shape[0]):
        for i in range(k):
            diff = X_proj[j][i] - p[j]
            diff2 = np.linalg.norm(diff)**2

            # FIXME: in the original paper, if the point is too close to a "real"
            # data point, the real one is returned. Keep it this way?
            if diff2 < 1e-6:
                # difference is too small, the counter part to p
                # precisely X[i]
                ret[j] = X[j][0]
                alpha[j, :] = -1
                idx_removal.append(j)
                break
            else:
                alpha[j, i] = 1.0/diff2
        # Os pontos removidos são aqueles cuja projeção inversa já é um ponto do dataset
        # e portanto eles não precisam de uma projeção inversa.
    p = np.delete(p, [idx_removal], axis=0)
    X_proj = np.delete(X_proj, [idx_removal], axis=0)
    X = np.delete(X, [idx_removal], axis=0)
    alpha = np.delete(alpha, [idx_removal], axis=0)
    

    #sum_alpha = np.sum(alpha, axis=0)
    sum_alpha = np.sum(alpha, axis=1)
    # 2. compute x_tilde, y_tilde
    #x_tilde = np.sum(alpha[:, np.newaxis]*X, axis=0)/sum_alpha
    x_tilde = np.sum(alpha[:, :, np.newaxis]*X, axis=1)/sum_alpha[:, np.newaxis]
    
    #y_tilde = np.sum(alpha[:, np.newaxis]*X_proj, axis=0)/sum_alpha
    y_tilde = np.sum(alpha[:, :, np.newaxis]*X_proj, axis=1)/sum_alpha[:, np.newaxis]

    # 3. build matrices A and B
    x_hat = X - x_tilde[:, np.newaxis, :]
    y_hat = X_proj - y_tilde[:, np.newaxis, :]

    alpha_sqrt = np.sqrt(alpha)
    #A = alpha_sqrt[:, np.newaxis]*y_hat
    A = alpha_sqrt[:, :, np.newaxis]*y_hat
    #B = alpha_sqrt[:, np.newaxis]*x_hat
    B = alpha_sqrt[:, :, np.newaxis]*x_hat

    A = A.reshape(p.shape[0],2,k) # Para tentar simular um A.T para várias matrizes empilhadas (stacked)
    ATB = np.matmul(A,B)
    u, s, vh = np.linalg.svd(ATB)

    aux = np.zeros((p.shape[0], 2, vh.shape[-1]))
    aux[:, 0] = vh[:, 0]
    aux[:, 1] = vh[:, 1]

    u = u.reshape(p.shape[0],u.shape[-1],u.shape[-2])

    M = np.matmul(u, aux)
    
    jump_idx = 0
    for j in tqdm.tqdm(range(p.shape[0])):
        while ret[j+jump_idx] is not None:
            jump_idx += 1
        res = np.dot(p[j] - y_tilde[j], M[j]) + x_tilde[j]
        if len(res.shape) > 1:
            print("Here - j: {j} - jump_idx: {jump_idx}")
            raise Exception()
        ret[j+jump_idx] = res

    # for j in tqdm(range(p.shape[0])):
    #     if ret[j] is not None:
    #         continue
    #     Aj = A[j]
    #     Bj = B[j]
    #     u, s, vh = np.linalg.svd(np.dot(Aj.T, Bj))

    #     # 5. let M = UV
    #     #aux = np.zeros((2, X.shape[1]))
    #     aux = np.zeros((2, vh.shape[-1]))
    #     aux[0] = vh[0]
    #     aux[1] = vh[1]
    #     M = np.dot(u, aux)
    #     ret[j] = np.dot(p[j] - y_tilde[j], M) + x_tilde[j]

    return np.array(ret)


def refine_lamp(X, X_proj, max_iter=50, k=8):
    tree = KDTree(X)
    for i in range(max_iter):
        for i in range(X.shape[0]):
            p = X[i]
            proj_p = X_proj[i]
            # 1. take N(B(x))
            dist, ind = tree.query([p], k=k)
            ind = ind[0]
            
            # neighbors of the point in nD
            neigh_bp = X[ind]

            # where they project
            neigh_p = X_proj[ind]

            center = np.sum(neigh_p, axis=0)/neigh_p.shape[0]

            # move the projected point a little in direction of the center
            X_proj[i] += (center - proj_p)/10.0

