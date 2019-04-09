import unittest
import os
import dlkit.data3d as data


DATA_ROOT_MSD = os.environ.get('DATA_ROOT_MSD')


class TestFunctional(unittest.TestCase):

    def test_zoom(self):
        pass

    def test_elastic_transform(self):
        pass


class TestTransforms(unittest.TestCase):

    def test_resize(self):
        pass

    def test_squeeze_to_interval(self):
        pass

    def test_standardize(self):
        pass

    def test_to_tensor(self):
        pass

    def test_encode_one_hot(self):
        pass

    def test_litter(self):
        pass

    def test_choice(self):
        pass

    def test_label_elastic_transform(self):
        pass


@unittest.skipIf(DATA_ROOT_MSD is None, 'MSD data root is not provided')
class TestMSD(unittest.TestCase):

    pass


if __name__ == '__main__':
    unittest.main()
