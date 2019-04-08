import unittest
import dlkit.data3d as data


class TestFunctional(unittest.TestCase):

    def test_zoom(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_elastic_transform(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())


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


class TestDatasets(unittest.TestCase):

    pass


if __name__ == '__main__':
    unittest.main()
