import base64
import json
from logger import logger
from exceptions import FailedToSendError
from stable_diffusion_request_queue_worker import StableDiffusionRequestQueueWorker


class FastStableDiffusionRequestQueueWorker(StableDiffusionRequestQueueWorker):
    def response_queue_worker(self):
        """
        Wait for responses from the stable diffusion runner and send
        them to the client
        """

    def handle_image(self, image, options):
        print("HANDLE IMAGE RESPONSE")
        # image is bytes and therefore not json serializable,
        # convert it to base64 first
        image = base64.b64encode(image).decode()
        response = {
            "image": image,
            "reqtype": options["reqtype"],
            "pos_x": options["options"]["pos_x"],
            "pos_y": options["options"]["pos_y"],
        }
        # encode response as a byte string
        response = json.dumps(response).encode()
        if response is not None and response != b'':
          # logger.info("prepping image")
            img = response #self.prep_image(response, options)
            try:
                bytes_sent = 0
                #expected_byte_size = settings.BYTE_SIZES[self.image_size]
                # actual_size = len(img)
                #
                # if actual_size < expected_byte_size:
                #     print("sending image of size {}".format(actual_size))
                #     # pad the image img_bytes with zeros
                #     img = img + b'\x00' * (expected_byte_size - actual_size)

                # send message in chunks
                chunk_size = 1024
                for i in range(0, len(img), chunk_size):
                    chunk = img[i:i + chunk_size]
                    self.send_image_chunk(chunk)

                #time.sleep(0.001)
                self.send_end_message()

                if self.soc_connection:
                    #self.soc_connection.settimeout(1)
                    pass
                else:
                    raise FailedToSendError()
            except FailedToSendError as ect:
                logger.error("failed to send message")
                # cancel the current run
                self.sdrunner.cancel()
                logger.error(ect)
                self.reset_connection()

    def __init__(self, *args, **kwargs):
        """
        Initialize the worker
        """
        super().__init__(*args, **kwargs)