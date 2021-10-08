class Frame(object):
    def __init__(self, image, depth, index, position=None):
        self.image = image
        self.depth = depth
        self.index = index
        self.position = position
        self.model_index = 0
        self.region_sample_weights = None
