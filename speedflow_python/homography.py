import numpy as np, yaml, cv2, os

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(source, target)
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)
        
def load_points(yml_path: str):
    with open(yml_path, "r") as f:
        d = yaml.safe_load(f)
    source = np.array(d["SOURCE"], dtype=np.float32)
    target = np.array(d["TARGET"], dtype=np.float32)
    return source, target