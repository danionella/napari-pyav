import av
import numpy as np
from napari_pyav import napari_get_reader


def _make_video(path, n=20, shape=(64, 96)):
    """Write a short H.264 test video; frame i has mean grey level about 10 * i."""
    with av.open(path, 'w') as c:
        s = c.add_stream('libx264', rate=25)
        s.height, s.width, s.pix_fmt = shape[0], shape[1], 'yuv420p'
        for i in range(n):
            im = np.full((*shape, 3), 10 * i, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(im, format='rgb24')):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


# tmp_path is a pytest fixture
def test_reader(tmp_path):
    video_path = str(tmp_path / "test.mp4")
    _make_video(video_path)
    reader = napari_get_reader([video_path])([video_path])[0][0]
    assert reader.shape == (20, 64, 96, 3)
    assert reader.size == 20 * 64 * 96 * 3
    frames = [frame for frame in reader]
    assert len(frames) == 20 and all(isinstance(f, np.ndarray) for f in frames)
    assert abs(reader[5].mean() - 50) < 5   # random access returns the right frame


def test_get_reader_pass():
    reader = napari_get_reader("fake.file")
    assert reader is None
