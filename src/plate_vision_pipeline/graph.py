from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from plate_vision_pipeline.state import PipelineState
def route_after_detect(state:dict) -> list[str]:
    '''Metodo que dirige a los nodos segment y describe en paralelo. O si falla la deteccion envia al final'''
    if 'detections' not in state:
        raise Exception('detections no existe')
    if state['detections'] == []:
        return [END]
    return ['segment', 'describe']

class RouteAfterMeasure:
    def __init__(self, max_attempts: int = 3) -> None:
        self.max_attempts = max_attempts
    def __call__(self, state:dict) -> str:
        if 'structure' not in state and state['measure_attempts'] < self.max_attempts:
            return 'describe'
        return END


def build_graph(detect_node,segment_node,describe_node,measure_node, route_after_measure:RouteAfterMeasure = RouteAfterMeasure()) -> CompiledStateGraph:
    graph = StateGraph(PipelineState)

    # Declaracion de los nodos
    graph.add_node('detect', detect_node)
    graph.add_node('segment' ,segment_node)
    graph.add_node('describe',describe_node)
    graph.add_node('measure', measure_node)

    graph.set_entry_point('detect')

    #Declaracion de las aristas
    graph.add_conditional_edges('detect', route_after_detect)
    graph.add_edge('describe', 'measure')
    graph.add_conditional_edges('measure',route_after_measure)
    graph.add_edge('segment',END)

    return graph.compile()
