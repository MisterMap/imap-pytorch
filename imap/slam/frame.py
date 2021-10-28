class Frame(object):
    def __init__(self, image, depth, ground_truth_position, index):
        self.image = image
        self.depth = depth
        self.id = index
        self.ground_truth_position = ground_truth_position
