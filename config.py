import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG = True
DATA_DIR = os.path.join(BASE_DIR, 'data')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'data', 'test')
TEST_DATA_METRICS = os.path.join(BASE_DIR, 'data', 'test', 'list_of_metrics.json')
TEST_DATA_STATISTICS = os.path.join(BASE_DIR, 'data', 'test', 'average_statistics.json')
