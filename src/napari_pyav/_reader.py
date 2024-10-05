"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
import av

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # if isinstance(path, list):
    #     raise ValueError("This reader only supports single file paths")

    # # if we know we cannot read the file, we immediately return None.
    # if not path.endswith(".mp4"):
    #     return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    paths = [path] if isinstance(path, str) else path
    out = [(FastVideoReader(path), {}, 'image') for path in paths]
    return out


class FastVideoReader:
    ''' A class to read video files quickly, using the pyav package. It allows for fast seeking and reading of frames. 
    
    Args:
        filename (str): path to the video file
        threading (bool): enable threading in the decoder. Default is True.
        read_format (str): format to read the video in. Default is 'gray'. Other options include 'rgb24', 'bgr24', 'yuv420p', etc.
        pts_lookup (np.ndarray): lookup seek table for pts values. Default is None (generate when needed).
        thread_count (int): number of threads to use for decoding. Default is 0 (auto).
    '''
    def __init__(self, filename, threading=True, read_format='gray', pts_lookup=None, thread_count=0):
        self.container = av.open(filename)
        self.stream = self.container.streams.video[0]
        self.stream.codec_context.thread_count = thread_count
        self.stream.codec_context.thread_type = 'AUTO' if threading else 'SLICE' # FRAME/AUTO/SLICE
        self.framegenerator = self.container.decode(video=0)
        self.read_format = read_format
        self._pts_lookup = pts_lookup
        self._pts_per_frame = 1 / (self.stream.guessed_rate * self.stream.time_base)
        self._init_pts = int(next(self.framegenerator).pts)
        self._frame_to_pts = lambda n: round(n * self._pts_per_frame) + self._init_pts
        self.rewind()

    def read(self):
        ''' Read the next frame in the specified format. '''
        frame_obj = next(self.framegenerator)
        self.last_pts = frame_obj.pts
        im = frame_obj.to_ndarray(format=self.read_format)
        #print(frame_obj.pts, frame_obj.dts, frame_obj.time)
        del frame_obj
        return im

    def rewind(self):
        ''' Rewind the video to the beginning. '''
        self.container.seek(0)
        self.framegenerator = self.container.decode(video=0)
        self.last_pts = None

    def read_frame(self, frame_idx):
        ''' Read the specified frame index. 
        
        Args:
            frame_idx (int): index of the frame to read
        '''
        if frame_idx == 0:
            self.rewind()
            return self.read()
        if self.last_pts is not None and self.last_pts == self._frame_to_pts(frame_idx-1):
            return self.read()
        target_pts = self._frame_to_pts(frame_idx)
        self.container.seek(target_pts-self._init_pts, backward=True, stream=self.container.streams.video[0])
        self.framegenerator = self.container.decode(video=0)
        frame_obj = next(self.framegenerator)
        while frame_obj.pts != target_pts:
            assert frame_obj.pts <= target_pts, f'pts glitch: {frame_obj.pts} > {target_pts}'
            frame_obj = next(self.framegenerator)
        frame = frame_obj.to_ndarray(format=self.read_format)
        self.last_pts = frame_obj.pts
        return frame
    
    def close(self):
        self.container.close()
        
    def __del__(self):
        self.close()
        
    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):  # single frame
            return self.read_frame(index)
        elif isinstance(index, tuple) and isinstance(index[0], int):
            return self.read_frame(index[0])
        elif isinstance(index, slice):
            frames = [self.read_frame(i) for i in np.r_[index]]
            return np.array(frames)
        else:
            raise NotImplementedError, "slicing of {type(index)} : {index} not implemented yet"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
    
    def __iter__(self):
        self.rewind()
        return self
    
    def __next__(self):
        try:
            return self.read()
        except:
            self.rewind()
            raise StopIteration
    
    @property
    def frame_shape(self):
        ''' Return the shape of the video frames. '''
        return self.container.streams.video[0].codec_context.height, self.container.streams.video[0].codec_context.width

    @property
    def nframes(self):
        ''' Return the number of frames in the video. '''
        return self.container.streams.video[0].frames

    @property
    def dtype(self):
        return np.uint8

    @property
    def shape(self):
        return (self.nframes, *self.frame_shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.product(self.shape)
    
    @staticmethod
    def static_shape(filename):
        ''' Get the shape of a video (static method). 
        
        Args:
            filename (str): path to the video file'''
        with av.open(filename) as container:
            stream = container.streams.video[0]
            shape = np.array([stream.frames, stream.codec_context.height, stream.codec_context.width])
        return shape