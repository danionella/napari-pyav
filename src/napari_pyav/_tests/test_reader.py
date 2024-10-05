import numpy as np
from napari_pyav import napari_get_reader


# # tmp_path is a pytest fixture
def test_reader(tmp_path):

    import urllib.request
    video_path = str(tmp_path / "test.mp4")
    urllib.request.urlretrieve("https://ia803405.us.archive.org/27/items/archive-video-files/test.mp4", video_path)
    reader = napari_get_reader([video_path])([video_path])[0]
    print(reader, type(reader))
    for frame in reader:
        assert isinstance(frame, np.ndarray), str(type(reader)) + str(reader)
        # """An example of how you might test your plugin."""

    # # write some fake data using your supported file format
    # my_test_file = str(tmp_path / "myfile.npy")
    # original_data = np.random.rand(20, 20)
    # np.save(my_test_file, original_data)

    # # try to read it back in
    # reader = napari_get_reader(my_test_file)
    # assert callable(reader)

    # # make sure we're delivering the right format
    # layer_data_list = reader(my_test_file)
    # assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    # layer_data_tuple = layer_data_list[0]
    # assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

    # # make sure it's the same as it started
    # np.testing.assert_allclose(original_data, layer_data_tuple[0])


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
