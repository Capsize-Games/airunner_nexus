import queue
from fast_stable_diffusion_request_queue_worker import FastStableDiffusionRequestQueueWorker


if __name__ == '__main__':
    # app = Server(__name__)
    # app.run(debug=True)
    port = 5000
    host = "http://localhost"
    app = FastStableDiffusionRequestQueueWorker(
        port=port,
        host=host
    )
