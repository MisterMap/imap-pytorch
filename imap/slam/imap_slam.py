import threading


class IMAPSLAM(object):
    def __init__(self, map_builder, tracker):
        self._tracker = tracker
        self._tracker_mutex = threading.Lock()
        self._map_builder = map_builder
        self._map_builder_mutex = threading.Lock()
        self._map_builder_loop = threading.Thread(target=self.map_builder_loop)
        self._map_builder_initialized = False

    def map_builder_loop(self):
        while True:
            self.map_builder_step()

    def map_builder_step(self):
        with self._map_builder_mutex:
            model = self._map_builder.step()
        with self._tracker_mutex:
            self._tracker.update_model(model)

    def update(self, current_frame):
        if not self._map_builder_initialized:
            self.init_map_builder(current_frame)
            return
        with self._tracker_mutex:
            tracked_position = self._tracker.track_current_frame(current_frame)
        with self._map_builder_mutex:
            self._map_builder.set_current_frame(current_frame, tracked_position)

    def init_map_builder(self, first_frame):
        self._map_builder.init(first_frame)
        self.map_builder_step()
        self._map_builder_loop.run()
        self._map_builder_initialized = True
