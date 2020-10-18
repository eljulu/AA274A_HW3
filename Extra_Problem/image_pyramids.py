#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt


def half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    return image[::2,::2,:]
    # raise NotImplementedError("Implement me!")
    ########## Code ends here ##########


def blur_half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    blurred = cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0.7)
    return blurred[::2,::2,:]
    # raise NotImplementedError("Implement me!")
    ########## Code ends here ##########


def two_upscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        upscaled_image: A 2x-upscaled version of image.
    """
    ########## Code starts here ##########
    # raise NotImplementedError("Implement me!")
    image = np.repeat(image, 2, axis = 0)
    image = np.repeat(image,2,  axis = 1)
    return image
    ########## Code ends here ##########


def bilinterp_upscale(image, scale):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
        scale: How much larger to make the image

    Returns
        upscaled_image: A scale-times upscaled version of image.
    """
    m, n, c = image.shape

    f = (1./scale) * np.convolve(np.ones((scale, )), np.ones((scale, )))
    f = np.expand_dims(f, axis=0) # Making it (1, (2*scale)-1)-shaped
    filt = f.T * f

    ########## Code starts here ##########
    scaled = np.zeros((m * scale, n * scale, c))
    scaled[::scale, ::scale, :] = image
    return cv2.filter2D(scaled, -1, filt)
    # raise NotImplementedError("Implement me!")
    ########## Code ends here ##########


def main():
    # OpenCV actually uses a BGR color channel layout,
    # Matplotlib uses an RGB color channel layout, so we're flipping the 
    # channels here so that plotting matches what we expect for colors.
    test_card = cv2.imread('test_card.png')[..., ::1].astype(float)
    favicon = cv2.imread('favicon-16x16.png')[..., ::1].astype(float)
    test_card /= test_card.max()
    favicon /= favicon.max()

    # Note that if you call matplotlib's imshow function to visualize images,
    # be sure to pass in interpolation='none' so that the image you see
    # matches exactly what's in the data array you pass in.
    
    ########## Code starts here ##########

    # 1/8 size down
    test_card_size_down = half_downscale(half_downscale((half_downscale(test_card))))
    # plt.imshow(test_card_size_down, interpolation = 'none')
    cv2.imwrite('test_card_1_8.png',255*test_card_size_down)
    # raise NotImplementedError("Implement me!")

    # blurred size down
    test_card_size_down = blur_half_downscale(blur_half_downscale((blur_half_downscale(test_card))))
    cv2.imwrite('blur_test_card_1_8.png', 255 * test_card_size_down)

    # upscale
    favicon_upscale = two_upscale(two_upscale(two_upscale(favicon)))
    cv2.imwrite('favicon_upscale.png', 255 * favicon_upscale)

    # bilinterp upsclae
    favicon_upscale_bilinterp = bilinterp_upscale(favicon, 8)
    cv2.imwrite('favicon_upscale_bilinterp.png', 255 * favicon_upscale_bilinterp)
    ########## Code ends here ##########


if __name__ == '__main__':
    main()
