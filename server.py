import argparse

from stable_diffusion_request_queue_worker import StableDiffusionRequestQueueWorker

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='http://localhost')
    parser.add_argument('--timeout', action='store_true', default=False)
    parser.add_argument('--chunk-size', type=int, default=1024)
    parser.add_argument('--max-client-connections', type=int, default=1)
    parser.add_argument('--model-base-path', type=str, default='.')

    return parser.parse_args()

if __name__ == '__main__':
    # get command line arguments
    args = parse_args()
    app = StableDiffusionRequestQueueWorker(
        port=args.port,
        host=args.host,
        do_timeout=args.timeout,
        chunk_size=args.chunk_size,

        # future:
        max_client_connections=args.max_client_connections,
        model_base_path=args.model_base_path
    )
