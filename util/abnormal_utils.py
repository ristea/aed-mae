import numpy as np


def gaussian_filter(support, sigma):
    mu = support[len(support) // 2 - 1]
    # mu = np.mean(support)
    filter = 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter


def filt(input, dim=9, range=302, mu=21):
    filter_3d = np.ones((dim, dim, dim)) / (dim ** 3)
    filter_2d = gaussian_filter(np.arange(1, range), mu)

    frame_scores = input  # convolve(input, filter_3d)
    # frame_scores = frame_scores.max((1, 2))

    padding_size = len(filter_2d) // 2
    in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
    frame_scores = np.correlate(in_, filter_2d, 'valid')
    return frame_scores

