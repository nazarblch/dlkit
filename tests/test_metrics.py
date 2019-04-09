import unittest
import numpy as np
import dlkit.metrics.binary


class TestBinary(unittest.TestCase):

    def test_nsd(self):
        spacing = (2, 2, 2)
        tolerance = 4

        mask_gt = np.zeros((3, 2, 100, 100, 100), np.uint8)
        mask_pred = np.zeros((3, 2, 100, 100, 100), np.uint8)
        mask_gt[:, 0, 0:50, :, :] = 1
        mask_pred[:, 0, 0:51, :, :] = 1
        mask_gt[:, 1, 50:, :, :] = 1
        mask_pred[:, 1, 51:, :, :] = 1
        print(dlkit.metrics.binary.compute_metrics(mask_gt, mask_pred, metrics=('nsd',), spacing=spacing, tolerance=tolerance))

        mask_gt = np.zeros((100, 100, 100), np.uint8)
        mask_pred = np.zeros((100, 100, 100), np.uint8)
        mask_gt[0:50, :, :] = 1
        mask_pred[0:51, :, :] = 1
        print(dlkit.metrics.binary.compute_metrics(mask_gt, mask_pred, metrics=('nsd',), spacing=spacing, tolerance=tolerance))

        mask_gt = np.zeros((3, 100, 100, 100), np.uint8)
        mask_pred = np.zeros((3, 2, 100, 100, 100), np.uint8)
        mask_gt[:, 50:, :, :] = 1
        mask_pred[:, 0, :51, :, :] = 1
        mask_pred[:, 1, 51:, :, :] = 1
        print(dlkit.metrics.binary.compute_metrics(mask_gt, mask_pred, metrics=('nsd',), spacing=spacing, tolerance=tolerance))


if __name__ == '__main__':
    unittest.main()
