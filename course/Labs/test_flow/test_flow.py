from metaflow import FlowSpec, step, kubernetes

class TestFlow(FlowSpec):
    """
    A simple flow to test Metaflow setup with GCP and Kubernetes
    """

    @step
    def start(self):
        """
        Start the flow and store a message
        """
        self.message = "Hello from the start step!"
        print(self.message)
        self.next(self.process)

    @kubernetes
    @step
    def process(self):
        """
        Process step that runs on Kubernetes
        """
        import socket
        print(f"Previous message: {self.message}")
        self.kube_message = f"Hello from Kubernetes! Running on host: {socket.gethostname()}"
        print(self.kube_message)
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow and print both messages
        """
        print("Flow messages:")
        print(f"Start message: {self.message}")
        print(f"Kubernetes message: {self.kube_message}")

if __name__ == '__main__':
    TestFlow()
