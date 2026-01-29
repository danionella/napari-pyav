import napari
import numpy as np
from napari.layers import Image
from magicgui import magicgui
import sounddevice as sd
import av
import tqdm

from ._reader import FastVideoReader


GLOBAL_STATE = dict(audio_stream=None, playing=False)
_maxjitter = 10  # maximum allowed seek jitter

class AudioReader:
    def __init__(self, filename):
        self.container = av.open(filename)
        self.audio_stream = self.container.streams.audio[0]
        self.audio_stream.codec_context.thread_count = 0
        self.audio_stream.codec_context.thread_type = "AUTO"
        self.atb = self.audio_stream.time_base
        self.chunkgenerator = self.container.decode(self.audio_stream)

    def seek(self, time_seconds):
        self.container.seek(int(time_seconds / self.atb), backward=True, any_frame=True, stream=self.audio_stream)
        self.chunkgenerator = self.container.decode(self.audio_stream)

    def rewind(self):
        self.seek(0)
        self.chunkgenerator = self.container.decode(self.audio_stream)

    def read(self):
        frame = next(self.chunkgenerator)
        data = frame.to_ndarray()
        meta = dict(pts=frame.pts, time=frame.time, rate=frame.sample_rate, layout=frame.layout.name)
        del frame
        return data, meta
    
    def close(self):
        self.container.close()

    def __del__(self):
        try: 
            self.close()
        except:
            pass


from qtpy import QtCore
import threading


class CoalescedStepSetter(QtCore.QObject):
    def __init__(self, viewer, axis: int):
        super().__init__()
        self._viewer = viewer
        self._axis = axis
        self._lock = threading.Lock()
        self._latest = None
        self._scheduled = False

    def request(self, step: int) -> None:
        with self._lock:
            self._latest = int(step)
            if self._scheduled:
                return
            self._scheduled = True
        # run once on the Qt main loop; if many requests come in, they collapse to one
        QtCore.QTimer.singleShot(0, self._apply_latest)

    @QtCore.Slot()
    def _apply_latest(self) -> None:
        with self._lock:
            step = self._latest
            self._scheduled = False
        if step is None:
            return
        # Prefer setting a single axis (less churn than writing the whole tuple)
        self._viewer.dims.set_current_step(self._axis, step)


@magicgui(
    image={"label": "video layer"}, call_button=" Play with audio", volume_dB={"widget_type": "FloatSlider", "min": -20, "max": 60})
def _av_widget_function(image: Image, viewer: napari.Viewer, playback_speed: float = 1.0, volume_dB: float = 20.0, rewind: bool = True):

    if GLOBAL_STATE["playing"]:
        GLOBAL_STATE["audio_stream"].stop()
        GLOBAL_STATE["playing"] = False
        _av_widget_function.call_button.set_icon("play")
        _av_widget_function.call_button.text = " Play with audio"
        return

    video_reader_obj = image.data
    if not isinstance(video_reader_obj, FastVideoReader):
        napari.utils.notifications.show_error("The selected layer is not a video layer opened with this plugin.")
        return
    if not image.visible:
        napari.utils.notifications.show_error("The selected video layer is not visible.")
        return
    filename = str(video_reader_obj.container.name)
    ar = AudioReader(filename)
    blocksize = ar.audio_stream.codec_context.frame_size
    if blocksize is None or blocksize <= 0:
        napari.utils.notifications.show_error("Could not determine audio frame size.")
        return
    time_to_vframe = float(video_reader_obj._pts_per_frame * video_reader_obj.stream.time_base)
    t2v = lambda t: int(t / time_to_vframe)
    v2t = lambda v: v * time_to_vframe
    last_frame = -1
    setter = CoalescedStepSetter(viewer, axis=0)

    def callback(outdata, frames, time, status):
        nonlocal last_frame
        if status:
            print(status)
        try:
            read_audio_chunk, meta = ar.read()
            i_video_frame = t2v(meta["time"])
            if abs(viewer.dims.current_step[0] - i_video_frame) > _maxjitter:
                ar.seek(v2t(viewer.dims.current_step[0]))
                read_audio_chunk, meta = ar.read()
                i_video_frame = t2v(meta["time"])
            if i_video_frame != last_frame:
                # viewer.dims.current_step = (i_video_frame, 0, 0)
                setter.request(i_video_frame)
                last_frame = i_video_frame
            read_audio_chunk *= (10 ** (volume_dB / 20))
            outdata[:] = read_audio_chunk.T

        except StopIteration:
            GLOBAL_STATE["playing"] = False
            _av_widget_function.call_button.set_icon("play")
            _av_widget_function.call_button.text = " Play with audio"
            if rewind:
                setter.request(0)
                ar.seek(0)
            ar.close()
            raise sd.CallbackStop

    audio_stream = sd.OutputStream(channels=2, callback=callback, blocksize=blocksize, samplerate=int(ar.audio_stream.codec_context.sample_rate*playback_speed))
    audio_stream.start()
    GLOBAL_STATE["playing"] = True
    _av_widget_function.call_button.set_icon("pause")
    _av_widget_function.call_button.text = " Pause"
    GLOBAL_STATE["audio_stream"] = audio_stream


def get_widget(*args, **kwargs):
    print(*args, **kwargs)
    _av_widget_function.call_button.set_icon("play")
    return _av_widget_function
