from plate_vision_pipeline.state import PipelineState, Segmentation
import numpy as np

def latency(node_name):
    def decorator(function):
        def wrapper(self, state: PipelineState):
            import time
            start = time.perf_counter()
            ret = function(self, state)
            end = time.perf_counter()
            print(f"{node_name} took {end - start} seconds")
            ret['latency_ms'] = {node_name: (end - start)*1000}
            return ret
        return wrapper
    return decorator


class DetectNode:
    def __init__(self, model) -> None:
        self.model = model

    @latency('detect')
    def __call__(self, state: PipelineState):
        try:
            detections = self.model.predict(state['image'])
            return {'detections': detections}
        except Exception as e:
            return {'detections': [], 'errors': [f'detect: {e}']}

class SegmentNode:
    def __init__(self, model) -> None:
        self.model = model

    @latency('segment')
    def __call__(self, state: PipelineState):
        try:
            detections = state['detections']
            image = state['image']
            if not detections:
                return {'segmentations': []}
            masks: list[np.ndarray] = self.model.predict(image, [d['bbox'] for d in detections])

            segments: list[Segmentation] =[]
            for i in range(len(detections)):
                segment: Segmentation = Segmentation(bbox=detections[i]['bbox'],
                   cls=detections[i]['cls'],
                   conf=detections[i]['conf'],
                   mask=masks[i])
                segments.append(segment)
            return {'segmentations': segments}
        except Exception as e:
            return {'segmentations': [], 'errors': [f'segment: {e}']}


class DescribeNode:
    def __init__(self, model) -> None:
        self.model = model

    @latency('describe')
    def __call__(self, state: PipelineState):
        try:
            describe = self.model.predict(state['image'], state['detections'])
            return {'description' : describe}
        except Exception as e:
           return {'description': "", 'errors': [f'describe: {e}']}

class MeasureNode:
    def __init__(self,model) -> None:
        self.model = model

    @latency('measure')
    def __call__(self, state: PipelineState):
        attemps = state.get('measure_attempts',0) + 1
        try:
            if 'description' not in state:
                return {'measure_attempts' : attemps}
            return  {'structure' : self.model.predict(state['description']), 'measure_attempts' : attemps}
        except Exception as e:
            return {'measure_attempts' : attemps, 'errors': [f'measure: {e}']}
