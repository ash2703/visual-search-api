import unittest

class TestApp(unittest.TestCase):
    """Initial unit test for learning"""

    def test_return_backwards_string(self):
        """Test return backwards simple string"""
        random_string = "This is my test string"
        self.assertEqual(random_string, random_string)

if __name__ == "__main__":
    unittest.main()
