import unittest
import torch
import dlkit.nn


NO_CUDA = not torch.cuda.is_available()


class TestModules(unittest.TestCase):

    def test_append_grid(self):
        # 3d data_loader
        module = dlkit.nn.AppendGrid()
        features = torch.randn(4, 5, 9, 10, 11)
        output = module(features)
        self.assertEqual(output.shape, (4, 8, 9, 10, 11))
        # 2d data_loader
        module = dlkit.nn.AppendGrid()
        features = torch.randn(4, 5, 9, 10)
        output = module(features)
        self.assertEqual(output.shape, (4, 7, 9, 10))

    @unittest.skipIf(NO_CUDA, 'cuda is not available')
    def test_append_grid_cuda(self):
        # 3d data_loader
        module = dlkit.nn.AppendGrid()
        features = torch.randn(4, 5, 9, 10, 11).cuda()
        output = module(features)
        self.assertEqual(output.shape, (4, 8, 9, 10, 11))
        # 2d data_loader
        module = dlkit.nn.AppendGrid()
        features = torch.randn(4, 5, 9, 10).cuda()
        output = module(features)
        self.assertEqual(output.shape, (4, 7, 9, 10))


if __name__ == '__main__':
    unittest.main()
