import signal
import multiprocessing


class DrugDetectorProcess(multiprocessing.Process):
    def __init__(self, inference_event: multiprocessing.Event) -> None:
        super().__init__()

        self.inference_event = inference_event

        signal.signal(signal.SIGUSR1, self.execute)

    def run(self):
        while True:
            signal.pause()

    def execute(self, *args) -> None:
        
        self.inference_event.set()
    