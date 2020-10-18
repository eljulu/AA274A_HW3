# !/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########
    # raise NotImplementedError("Implement me!")
    # record dim
    k, ell, c = np.shape(F)
    m, n, c = np.shape(I)
    # padding
    leftright = np.zeros((m, ell//2, c))
    updown = np.zeros((k // 2, n + 2 * (ell//2), c))
    I = np.hstack((leftright, I, leftright))
    I = np.vstack((updown, I, updown))
    I = np.hstack((I, np.zeros((m + 2 * (k//2),1,c))))
    I = np.vstack((I, np.zeros((1, n + 2 * (ell//2) + 1,c))))
    # correlation
    G = np.zeros((m,n))
    f = F.flatten()
    # print(f)
    for i in range(m):
        for j in range(n):
            t_ij = I[i:i+k, j:j+ell, :].flatten()
            G[i][j] = np.dot(f, t_ij)
    return G
    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########
    # raise NotImplementedError("Implement me!")
    # record dim
    k, ell, c = np.shape(F)
    m, n, c = np.shape(I)
    # padding
    leftright = np.zeros((m, ell//2, c))
    updown = np.zeros((k // 2, n + 2 * (ell//2), c))
    I = np.hstack((leftright, I, leftright))
    I = np.vstack((updown, I, updown))
    I = np.hstack((I, np.zeros((m + 2 * (k//2),1,c))))
    I = np.vstack((I, np.zeros((1, n + 2 * (ell//2) + 1,c))))
    # correlation
    G = np.zeros((m,n))
    f = F.flatten()
    fnorm = np.linalg.norm(f)
    # print(f)
    for i in range(m):
        for j in range(n):
            t_ij = I[i:i+k, j:j+ell, :].flatten()
            G[i][j] = np.dot(f, t_ij)/fnorm/np.linalg.norm(t_ij)
    return G
    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 200, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        stop = time.time()
        print 'Correlation function runtime:', stop - start, 's'
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
    # a = np.array([[1, 2, 3],
    #        [2, 4, 5],
    #        [3, 5, 6]])
    # a = np.dstack((a,a,a))
    # print(populatevector(a))