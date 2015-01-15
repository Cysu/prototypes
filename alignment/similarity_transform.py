import numpy as np
import skimage.transform as tf
import matplotlib.pyplot as plt


def set_conf():
    return {
        'num_points': 10,
        'coor_range': (0, 100),
        'scale_range': (0.5, 2),
        'translation_range': (-10, 10),
        'noise_std': 10.0
    }


def generate_points(conf):
    n = conf['num_points']
    coor_min, coor_max = conf['coor_range']
    return coor_min + np.random.rand(n, 2) * (coor_max - coor_min)


def generate_transform(conf):
    # scale
    scale_min, scale_max = conf['scale_range']
    scale = scale_min + np.random.rand() * (scale_max - scale_min)
    # rotation
    rotation = np.random.rand() * 2 * np.pi
    # translation
    trans_min, trans_max = conf['translation_range']
    translation = trans_min + np.random.rand(2) * (trans_max - trans_min)
    # create transform
    return tf.SimilarityTransform(
        scale=scale, rotation=rotation, translation=translation)


def similarity_transform(src_points, tform, conf):
    # conduct transformation
    dst_points = tform(src_points)
    # add gaussian noise
    dst_points += np.random.randn(*dst_points.shape) * conf['noise_std']
    return dst_points


def estimate_transform_own(src_points, dst_points):
    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src = src_points - src_mean
    dst = dst_points - dst_mean
    C = src.T.dot(dst)
    U, s, V = np.linalg.svd(C)
    rotation = V.dot(np.diag([1, np.linalg.det(V.dot(U.T))])).dot(U.T)
    scale = np.trace(dst.dot(rotation).dot(src.T)) / \
        np.trace(src.dot(src.T))
    translation = (dst_mean.T - scale * rotation.dot(src_mean.T)).T
    theta = np.arccos(rotation[0, 0])
    if -np.sin(theta) * rotation[0, 1] < 0:
        theta = 2 * np.pi - theta
    return tf.SimilarityTransform(
        scale=scale, rotation=theta, translation=translation)


def evaluate(src_points, dst_points, 
             tform, tform_estimated_skimage, tform_estimated_own):
    def _compute_mse(t):
        diff = t(src_points) - dst_points
        return np.linalg.norm(diff, axis=1).mean()

    print "Mean L2-norm of difference:"
    print "True transformation", _compute_mse(tform)
    print "Estimated by skimage", _compute_mse(tform_estimated_skimage)
    print "Estimated by own implementation", _compute_mse(tform_estimated_own)


def visualize(src_points, dst_points,
              tform, tform_estimated_skimage, tform_estimated_own):
    fig, ax = plt.subplots()
    scatters = []
    scatters += [ax.scatter(dst_points[:, 0], dst_points[:, 1], 20, color='g')]
    est = tform(src_points)
    scatters += [ax.scatter(est[:, 0], est[:, 1], 20, color='b')]
    est = tform_estimated_skimage(src_points)
    scatters += [ax.scatter(est[:, 0], est[:, 1], 20, color='r')]
    est = tform_estimated_own(src_points)
    scatters += [ax.scatter(est[:, 0], est[:, 1], 20, color='c')]
    plt.legend(scatters,
        ['dst with noise', 'true transform',
        'estimated by skimage', 'estimated by own'],
        scatterpoints=1, loc='lower left', ncol=4, fontsize=10)
    plt.show()


def main():
    # setup constants
    conf = set_conf()
    # generate random key points
    src_points = generate_points(conf)
    # generate random transformation
    tform = generate_transform(conf)
    # compute transformed key points with gaussian noise
    dst_points = similarity_transform(src_points, tform, conf)
    # estimate the transformation by skimage
    tform_estimated_skimage = tf.estimate_transform(
        'similarity', src_points, dst_points)
    # estimate the transformation by own implementation
    tform_estimated_own = estimate_transform_own(src_points, dst_points)
    # quantitatively compare two estimation methods
    evaluate(src_points, dst_points,
        tform, tform_estimated_skimage, tform_estimated_own)
    # qualitatively compare two estimation methods
    visualize(src_points, dst_points,
        tform, tform_estimated_skimage, tform_estimated_own)


if __name__ == '__main__':
    main()
