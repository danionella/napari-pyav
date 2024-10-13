import numpy as np
from napari_pyav import napari_get_reader

# # tmp_path is a pytest fixture
def test_reader(tmp_path):

    import urllib.request
    video_path = str(tmp_path / "test.mp4")
    urllib.request.urlretrieve(r"http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4", video_path)
    reader = napari_get_reader([video_path])([video_path])[0][0]
    print(reader, type(reader))
    for frame in reader:
        assert isinstance(frame, np.ndarray)

def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
