import queue
from stable_diffusion_request_queue_worker import StableDiffusionRequestQueueWorker

if __name__ == '__main__':
    # app = Server(__name__)
    # app.run(debug=True)
    port = 5000
    host = "http://localhost"
    app = StableDiffusionRequestQueueWorker(
        port=port,
        host=host
    )
