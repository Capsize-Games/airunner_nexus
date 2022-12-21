import os
import torch
import base64
import io
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipelineSafe
)
from pytorch_lightning import seed_everything
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from PIL import Image
from logger import logger


class SDRunner:
    _current_model = ""
    scheduler_name = "ddpm"
    do_nsfw_filter = True
    do_watermark = True
    initialized = False
    schedulers = {
        "ddpm": DDPMScheduler,
        "ddim": DDIMScheduler,
        "plms": PNDMScheduler,
        "lms": LMSDiscreteScheduler,
        "euler_a": EulerAncestralDiscreteScheduler,
        "euler": EulerDiscreteScheduler,
        "dpm": DPMSolverMultistepScheduler,
    }
    registered_schedulers = {}

    @property
    def current_model(self):
        return self._current_model

    @current_model.setter
    def current_model(self, model):
        if self._current_model != model:
            self._current_model = model
            if self.initialized:
                self.load_model()

    @property
    def model_path(self):
        model_path = self.current_model
        return model_path

    @property
    def scheduler(self):
        if not self.model_path or self.model_path == "":
            raise Exception("Chicken / egg problem, model path not set")
        if self.scheduler_name in self.schedulers:
            if self.scheduler_name not in self.registered_schedulers:
                self.registered_schedulers[self.scheduler_name] = self.schedulers[self.scheduler_name].from_pretrained(
                    self.model_path,
                    subfolder="scheduler"
                )
            return self.registered_schedulers[self.scheduler_name]
        else:
            raise ValueError("Invalid scheduler name")

    def load_model(self):
        torch.cuda.empty_cache()
        # load StableDiffusionSafetyChecker with CLIPConfig
        self.safety_checker = StableDiffusionSafetyChecker(
            StableDiffusionSafetyChecker.config_class()
        )

        if self.do_nsfw_filter:
            self.txt2img = StableDiffusionPipelineSafe.from_pretrained(
                self.model_path,
                torch_dtype=torch.half,
                scheduler=self.scheduler,
                low_cpu_mem_usage=True,
                # safety_checker=self.safety_checker,
                # feature_extractor=self.feature_extractor,
                revision="fp16"
            )
        else:
            self.txt2img = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.half,
                scheduler=self.scheduler,
                low_cpu_mem_usage=True,
                safety_checker=None,
                revision="fp16"
            )
        self.txt2img.enable_xformers_memory_efficient_attention()
        self.txt2img.to("cuda")
        self.img2img = StableDiffusionImg2ImgPipeline(**self.txt2img.components)
        self.inpaint = StableDiffusionInpaintPipeline(**self.txt2img.components)

    def initialize(self):
        if not self.initialized:
            self.load_model()
            self.initialized = True

    def do_reload_model(self):
        if self.reload_model:
            self.load_model()

    def prepare_model(self):
        # get model and switch to it
        model = self.options.get(f"{self.action}_model", self.current_model)

        if self.action in ["inpaint", "outpaint"]:
            if model in [
                "stabilityai/stable-diffusion-2-1-base",
                "stabilityai/stable-diffusion-2-base"
            ]:
                model = "stabilityai/stable-diffusion-2-inpainting"
            else:
                model = "stabilityai/stable-diffusion-inpainting"

        if model != self.current_model:
            self.current_model = model

    def change_scheduler(self):
        if self.model_path and self.model_path != "":
            self.txt2img.scheduler = self.scheduler
            self.img2img.scheduler = self.scheduler
            self.inpaint.scheduler = self.scheduler

    def prepare_scheduler(self):
        scheduler_name = self.options.get(f"{self.action}_scheduler", "ddpm")
        if self.scheduler_name != scheduler_name:
            self.scheduler_name = scheduler_name
            self.change_scheduler()

    def prepare_options(self, data):
        action = data.get("action", "txt2img")
        options = data["options"]
        self.reload_model = False
        self.seed = int(options.get(f"{action}_seed", 42))
        self.guidance_scale = float(options.get(f"{action}_scale", 7.5))
        self.num_inference_steps = int(options.get(f"{action}_steps", 50))
        self.strength = float(options.get(f"{action}_strength", 0.8))
        self.height = int(options.get(f"{action}_height", 512))
        self.width = int(options.get(f"{action}_width", 512))
        self.enable_community_models = bool(options.get(f"enable_community_models", False))
        self.C = int(options.get(f"{action}_C", 4))
        self.f = int(options.get(f"{action}_f", 8))
        self.prompt = options.get(f"{action}_prompt", "")
        self.negative_prompt = options.get(f"{action}_negative_prompt", "")
        self.batch_size = int(data.get(f"{action}_n_samples", 1))
        do_nsfw_filter = bool(options.get(f"do_nsfw_filter", False))
        do_watermark = bool(options.get(f"do_watermark", False))
        if do_nsfw_filter != self.do_nsfw_filter:
            self.do_nsfw_filter = do_nsfw_filter
            self.reload_model = True
        if do_watermark != self.do_watermark:
            self.do_watermark = do_watermark
            self.reload_model = True
        self.action = action
        self.options = options

    def generator_sample(self, data, image_handler):
        self.image_handler = image_handler
        return self.generate(data)

    def generate(self, data):
        self.prepare_options(data)
        self.prepare_scheduler()
        self.prepare_model()
        self.initialize()
        self.do_reload_model()

        # sample the model
        with torch.no_grad() as _torch_nograd, \
            torch.cuda.amp.autocast() as _torch_autocast:
            try:
                for n in range(0, self.batch_size):
                    self.seed = self.seed + n
                    seed_everything(self.seed)
                    image = None
                    if self.action == "txt2img":
                        image = self.txt2img(
                            self.prompt,
                            negative_prompt=self.negative_prompt,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                            callback=self.callback
                        ).images[0]
                    elif self.action == "img2img":
                        bytes = base64.b64decode(data["options"]["pixels"])
                        image = Image.open(io.BytesIO(bytes))
                        image = self.img2img(
                            prompt=self.prompt,
                            negative_prompt=self.negative_prompt,
                            image=image.convert("RGB"),
                            strength=self.strength,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                            callback=self.callback
                        ).images[0]
                        pass
                    elif self.action in ["inpaint", "outpaint"]:
                        bytes = base64.b64decode(data["options"]["pixels"])
                        mask_bytes = base64.b64decode(data["options"]["mask"])

                        image = Image.open(io.BytesIO(bytes))
                        mask = Image.open(io.BytesIO(mask_bytes))

                        # convert mask to 1 channel
                        # print mask shape
                        image = self.inpaint(
                            prompt=self.prompt,
                            negative_prompt=self.negative_prompt,
                            image=image,
                            mask_image=mask,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                            callback=self.callback
                        ).images[0]
                        pass

                    # use pillow to convert the image to a byte array
                    img_byte_arr = self.image_to_byte_array(image)
                    if img_byte_arr:
                        self.image_handler(img_byte_arr, data)
            except TypeError as e:
                if self.action in ["inpaint", "outpaint"]:
                    print(f"ERROR IN {self.action}")
                print(e)
            except Exception as e:
                print("Error during generation 1")
                print(e)

    def image_to_byte_array(self, image):
        img_byte_arr = None
        if image:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            # return flask.Response(img_byte_arr, mimetype='image/png')
        return img_byte_arr

    def callback(self, step, time_step, latents):
        self.tqdm_callback(step, int(self.num_inference_steps * self.strength))

    def __init__(self, *args, **kwargs):
        self.tqdm_callback = kwargs.get("tqdm_callback", None)
        super().__init__(*args)