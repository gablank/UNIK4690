import utilities


class ManualPlaygroundDetector:
    def __init__(self, petanque_detection):
        pass

    def detect(self, image):
        metadata = image.get_metadata()

        if "playground_poly" in metadata:
            return [tuple(coord) for coord in metadata["playground_poly"]]
        else:
            # user will be prompted to adjust polygon and can save it for later with 'd'
            return [(100,100), (100,200), (200,200), (200,100)]
            # playground_poly = utilities.select_polygon(image.get_bgr())
            # metadata = utilities.update_metadata(image.path, {"playground_poly": playground_poly})
