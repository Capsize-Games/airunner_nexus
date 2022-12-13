import os
import argparse
from itertools import islice
import cv2
import torch
import numpy as np
from einops import repeat
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from einops import rearrange
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from runai.logger import logger
from stablediffusion.classes import utils
from stablediffusion.classes.exceptions import DDIMException
from stablediffusion.classes.settings import Txt2ImgArgs, Img2ImgArgs
from stablediffusion.classes.supersampler import SuperSampler
from stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
from stablediffusion.ldm.models.diffusion.plms import PLMSSampler
from stablediffusion.classes.nsfwfilter import NSFWFilter
from stablediffusion.classes.settings_manager import settingsManager
from stablediffusion.ldm.util import instantiate_from_config
import logging
from tqdm.auto import trange, tqdm
from omegaconf import OmegaConf
HERE = os.path.dirname(os.path.abspath(__file__))


class StableDiffusionEngine:
    _verbose = False
    _here = None
    _bg_upsampler = None
    _plms_sampler = None
    _ddim_sampler = None
    _gfp_sampler = None
    _safety_feature_extractor_path = None
    _safety_model_path = None
    args = []
    current_sampler = None
    image_handler = None
    last_sample = None
    parser = None
    config = None
    data = None
    n_rows = None
    outpath = None
    wm_encoder = None
    base_count = None
    grid_count = None
    sample_path = None
    start_code = None
    initialized = False
    current_model = None
    _device = ""
    _model = None
    _resize = False
    _supersample = True

    @property
    def device(self):
        if self._device:
            return self._device
        return settingsManager.device_name

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def here(self):
        if self._here is None:
            self._here = os.path.dirname(os.path.abspath(__file__))
        return self._here

    @property
    def bg_upsampler(self):
        if self._bg_upsampler is None:
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2
            )
            self._bg_upsampler = RealESRGANer(
                scale=2,
                model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=True)  # need to set False in CPU mode
        return self._bg_upsampler

    @property
    def gfp(self):
        if self._gfp_sampler is None:
            self._gfp_sampler = GFPGANer(
                os.path.join(self.here, "../../gfpgan/bin/GFPGANv1.3.pth"),
                upscale=2,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self.bg_upsampler,
                device="cpu"
            )
        return self._gfp_sampler

    @property
    def plms_sampler(self):
        if self._plms_sampler is None:
            self._plms_sampler = PLMSSampler(self.model)
        return self._plms_sampler

    @property
    def ddim_sampler(self):
        if self._ddim_sampler is None:
            self._ddim_sampler = DDIMSampler(self.model)
        return self._ddim_sampler

    @gfp.setter
    def gfp(self, value):
        self._gfp_sampler = value

    @property
    def init_latent(self):
        # convert opt.init_img to tensor from base64
        init_image = self.load_image(
            settingsManager.init_img,
            settingsManager.batch_size
        )
        # move to latent space
        return self.model.get_first_stage_encoding(
            self.model.encode_first_stage(init_image)
        )

    @property
    def prompts(self):
        prompts = self.data[0]
        if isinstance(prompts, tuple):
            prompts = list(prompts)
        return prompts

    @property
    def prompt_input(self):
        return self.model.get_learned_conditioning(self.prompts)

    @property
    def unconditional_conditioning(self):
        if settingsManager.scale != 1.0:
            return self.model.get_learned_conditioning([
                settingsManager.negative_prompt
            ])
        return None

    @property
    def safety_feature_extractor_path(self):
        return self._safety_feature_extractor_path

    @safety_feature_extractor_path.setter
    def safety_feature_extractor_path(self, value):
        self._safety_feature_extractor_path = value

    @property
    def safety_model_path(self):
        return self._safety_model_path

    @safety_model_path.setter
    def safety_model_path(self, value):
        self._safety_model_path = value

    @property
    def CURRENT_MODEL_PATH(self):
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "GPU",
            settingsManager.model_version
        )

    @property
    def MODEL_PT_NAME(self):
        return "model.pt"

    @property
    def CURRENT_MODEL(self):
        return os.path.join(
            self.CURRENT_MODEL_PATH,
            self.MODEL_PT_NAME
        )

    @property
    def SAFETY_CHECKER_MODEL_IN_PATH(self):
        return os.path.join(
            self.CURRENT_MODEL_PATH,
            "stablediffusion",
            "safety_checker"
        )

    @property
    def SAFETY_CHECKER_MODEL_OUT(self):
        return os.path.join(
            self.CURRENT_MODEL_PATH,
            "safety_model.pt"
        )

    @property
    def SAFETY_CHECKER_MODEL_NAME(self):
        return "pytorch_model.bin"

    @property
    def SAFETY_CHECKER(self):
        return os.path.join(
            self.SAFETY_CHECKER_MODEL_IN_PATH,
            self.SAFETY_CHECKER_MODEL_NAME
        )

    @property
    def STABLE_DIFFUSION_PATH(self):
        return os.path.join(
            self.CURRENT_MODEL_PATH,
            "stablediffusion"
        )

    @property
    def SAFETY_FEATURE_EXTRACTOR_PICKLE_PATH_IN(self):
        return os.path.join(
            self.STABLE_DIFFUSION_PATH,
            "feature_extractor"
        )

    @property
    def SAFETY_FEATURE_EXTRACTOR_PICKLE_OUT(self):
        return os.path.join(
            self.CURRENT_MODEL_PATH,
            "safety_feature_extractor.pkl"
        )

    @property
    def MODEL_OUT(self):
        return os.path.join(
            self.CURRENT_MODEL_PATH,
            "model.pt"
        )

    @property
    def STABLE_DIFFUSION_CONFIG(self):
        return os.path.join(
            self.CURRENT_MODEL_PATH,
            "stablediffusion",
            "config.yaml"
        )

    @property
    def CHECK_POINT_PATH(self):
        return os.path.join(
            self.STABLE_DIFFUSION_PATH,
            "%s-pruned-emaonly.ckpt" % settingsManager.model_version
        )

    @property
    def INPAINT_FINETUNE_CHECKPOINT_PATH(self):
        return os.path.join(
            self.STABLE_DIFFUSION_PATH,
            "%s-pruned-emaonly.ckpt" % settingsManager.model_version
        )

    @property
    def INPAINT_FINETUNE_CHECKPOINT_PATH_OUT(self):
        return os.path.join(
            self.CURRENT_MODEL_PATH,
            "model.pt"
        )

    @property
    def SAFETY_FEATURE_EXTRACTOR_PICKLE_PATH_OUT(self):
        return self.CURRENT_MODEL_PATH

    @property
    def resize(self):
        return True # self._resize

    @property
    def supersample(self):
        return True # self._supersample

    def __init__(self, *args, **kwargs):
        settingsManager.model_version = kwargs.get("model_version")
        settingsManager.model_name = kwargs.get("model_name")
        self.current_model = self.CURRENT_MODEL
        self.kwargs = kwargs
        self.device = kwargs.get("device", self.device)
        settingsManager.current_model = self.kwargs.get("model_name")[0]
        self.args = kwargs.get("args", self.args)
        self.opt = {}
        self.safety_feature_extractor_path = kwargs.get("safety_feature_extractor_path")
        self.safety_model_path = kwargs.get("safety_model_path")[0]
        self.load_nsfw_filter()
        self.init_model(kwargs.get("options", {}))

    def load_nsfw_filter(self):
        self.nsfw_filter = NSFWFilter(
            safety_feature_extractor_path=self.safety_feature_extractor_path,
            safety_model_path=self.safety_model_path,
            model_version=settingsManager.model_version
        )

    def initialize(self):
        logger.info("Attempting to initialize model")
        if self.initialized:
            logger.info("Model already initialized")
            return
        self.initialized = True
        if not self.model or not self.device:
            #self.load_model()

            # load from config
            MODEL_PATH = os.path.join(HERE, "GPU", "v1-5", "stablediffusion")
            path_conf = os.path.join(MODEL_PATH, "config.yaml")
            path_ckpt = os.path.join(MODEL_PATH, "last.ckpt")
            config = OmegaConf.load(path_conf)
            self.model = self.load_model_from_config(config, path_ckpt)
            logger.info("Model loaded")
        else:
            logger.info("Model already loaded") if self.model else logger.info("Model not loaded")
            logger.info("Device already loaded") if self.device else logger.info("Device not loaded")
        self.initialize_start_code()

    def _load_model_by_name(self, model_name, device):
        logger.info(
            "Loading model from " + "" if model_name is None else model_name
        )
        utils.clear_cache()
        return torch.load(
            model_name,
            map_location=torch.device(device)
        )

    def _load_current_model(self):
        return self._load_model_by_name(self.current_model, self.device)

    def load_model(self):
        try:
            self.model = self._load_current_model()
            self.model = self.model.to(self.device)
            logger.info("Model loaded")
        except Exception as e:
            logger.error("Failed to load model")
            logger.error(e)

    def unload_current_model(self):
        # logger.info("Unloading model %s from memory" % self.current_model)
        # del self.model
        # self.model = None
        # #self.load_safety_feature_extractor_pickle()
        # self.reset_samplers()
        # utils.clear_cache()

        # move model to cpu
        self.model.to("cpu")

    def change_device(self, device):
        if self.device != device:
            self.device = device
            self.model = self.model.to(self.device)
            self.plms_sampler.model.to(self.device)
        utils.clear_cache()

    def numpy_to_pil(self, images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def load_replacement(self, x):
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize(
                (hwc[1], hwc[0])
            )
            y = (np.array(y) / 255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def initialize_start_code(self):
        if settingsManager.fixed_code:
            logger.info("Using fixed code")
            settingsManager.start_code = None
            # torch.randn([
            #     settingsManager.n_samples,
            #     settingsManager.C,
            #     settingsManager.H // settingsManager.f,
            #     settingsManager.W // settingsManager.f
            # ], device=self.device)
        # else:
        #     logger.info("Using random code")
        #     settingsManager.start_code = self.model.get_latent(self.init_latent)

    def get_first_stage_sample(self, model, samples):
        samples_ddim = model.decode_first_stage(samples)
        return torch.clamp((samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

    def parse_arguments(self):
        logger.info("Parsing arguments")
        parser = argparse.ArgumentParser()
        for arg in self.args:
            # check if parser has argument
            try:
                parser.add_argument(
                    f'--{arg["arg"]}',
                    **{k: v for k, v in arg.items() if k != "arg"}
                )
            except Exception as e:
                logger.error(f"Failed to parse argument {arg['arg']}")
                logger.error(e)
        self.parser = parser
        self.opt = self.parser.parse_args()

    def prepare_data(self):
        logger.info("Preparing data")
        batch_size = settingsManager.n_samples if "n_samples" in self.opt else 1
        n_rows = settingsManager.n_rows if (
                "n_rows" in self.opt and settingsManager.n_rows > 0) else 0
        prompt = settingsManager.prompt if "prompt" in self.opt else None
        data = [batch_size * [prompt]]
        self.n_rows = n_rows
        self.data = data
        logger.info("Data prepared")

    def init_model(self, options):
        self.parse_arguments()
        if options:
            settingsManager.parse_options(options)
        self.prepare_data()
        self.initialize()

    def skip_handler(self, x):
        pass

    def get_ancestral_step(self, sigma_from, sigma_to, eta=1.):
        """Calculates the noise level (sigma_down) to step down to and the amount
        of noise to add (sigma_up) when doing an ancestral sampling step."""
        if not eta:
            return sigma_to, 0.
        sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
        sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
        return sigma_down, sigma_up

    def to_d(self, x, sigma, denoised):
        """Converts a denoiser output to a Karras ODE derivative."""
        return (x - denoised) / utils.append_dims(sigma, x.ndim)

    def sample_euler_ancestral(self, x, sigmas, eta=1., options=None, image_handler=None):
        s_in = x.new_ones([x.shape[0]])
        extra_args = {}
        disable = False
        for i in trange(len(sigmas) - 1, disable=disable):
            denoised = self.model(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = self.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            d = self.to_d(x, sigmas[i], denoised)
            # Euler method
            dt = sigma_down - sigmas[i]
            x = x + d * dt
            x = x + torch.randn_like(x) * sigma_up
        self.handle_sample(x)

    def plms_sample(self, options=None, image_handler=None):
        self.current_sampler = self.plms_sampler
        self.current_sampler.parent = self
        self.image_handler = image_handler
        with torch.no_grad() as _torch_nograd, \
                settingsManager.precision_scope(settingsManager.device_name) as _precision_scope, \
                self.model.ema_scope() as _ema_scope:
            self.current_model = self.model
            # try:
            self.handle_sample(
                self.do_plms_sample(options)
            )
            # except DDIMException as e:
            #     logger.error("DDIMException: {}".format(e))
            #     return False
            # except RuntimeError as e:
            #     logger.error("RuntimeError: {}".format(e))
            #     return False

    def sample(self, options=None, image_handler=None):
        self.sample_euler_ancestral(options, image_handler)

    def ddim_sample(self, options=None, image_handler=None):
        self.current_sampler = self.ddim_sampler
        self.current_sampler.parent = self
        self.image_handler = image_handler
        self.current_sampler.make_schedule(
            ddim_num_steps=settingsManager.steps,
            ddim_eta=settingsManager.ddim_eta,
            verbose=settingsManager.verbose
        )

        # convert opt.init_img to tensor from base64
        init_image = self.load_img2img_image(
            settingsManager.init_img,
            settingsManager.batch_size
        )
        init_latent = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(init_image))  # move to latent space

        device = settingsManager.device_name

        with torch.no_grad() as _torch_nograd, \
             settingsManager.precision_scope(device) as _precision_scope, \
             self.model.ema_scope() as _ema_scope:
            self.current_model = self.model
            try:
                self.handle_sample(
                    self.do_ddim_sample(init_latent)
                )
            except DDIMException as e:
                logger.error("DDIMException: {}".format(e))
                return False
            except RuntimeError as e:
                logger.error("RuntimeError: {}".format(e))
                return False

    def do_plms_sample(self, options=None):
        logger.info("Txt2Img do_sample called")
        sample, _ = self.current_sampler.sample(
            S=settingsManager.steps,
            conditioning=self.prompt_input,
            batch_size=settingsManager.batch_size,
            shape=settingsManager.shape,
            verbose=settingsManager.verbose,
            unconditional_guidance_scale=settingsManager.scale,
            unconditional_conditioning=self.unconditional_conditioning,
            eta=settingsManager.ddim_eta,
            x_T=settingsManager.start_code,
            image_handler=self.current_sample_handler,
            et_prime_a=settingsManager.et_prime_a,
            et_prime_b=settingsManager.et_prime_b,
            et_prime_c=settingsManager.et_prime_c,
            et_prime_d=settingsManager.et_prime_d,
        )
        return sample

    def do_ddim_sample(self, init_latent):
        c = self.prompt_input
        t_enc = int(settingsManager.strength * settingsManager.steps)
        z_enc = self.current_sampler.stochastic_encode(
            init_latent,
            torch.tensor([t_enc], device=settingsManager.device)
        )
        samples = self.current_sampler.decode(
            z_enc, c, t_enc,
            unconditional_guidance_scale=settingsManager.scale,
            unconditional_conditioning=self.unconditional_conditioning,
            image_handler=self.current_sample_handler,
        )
        return samples

    def sample(self, options=None, image_handler=None, reqtype=None):
        self.init_model(options)
        self.base_count = len(os.listdir(self.sample_path))
        self.grid_count = len(os.listdir(self.outpath)) - 1
        #utils.clear_cache()
        img = None

        if reqtype == "Text2Image":
            self.args = Txt2ImgArgs
            return self.plms_sample(options, image_handler)
            # return self.do_sample(options, image_handler)
        elif reqtype == "Image2Image":
            self.args = Img2ImgArgs
            return self.ddim_sample(options, image_handler)
        else:
            logger.error("Failed to call sample")

    def current_sample_handler(self, img):
        if settingsManager.fast_sample:
            data = self.prepare_image_fast(img)
        else:
            data = self.prepare_image(img)
        self.image_handler(data, self.opt)

    def prepare_image(self, sample, finalize=False):
        clamped_sample = torch.clamp(
            (self.model.decode_first_stage(sample) + 1.0) / 2.0, min=0.0, max=1.0
        )
        if finalize:
            return self.post_process(clamped_sample)
        logger.info("Skipping post process")
        return self.prepare_image_fast(clamped_sample)

    def prepare_image_fast(self, sample):
        return 255. * rearrange(sample[0].cpu().numpy(), 'c h w -> h w c')

    def load_img2img_image(self, image, batch_size):
        """
        :param image: a one dimensional array of a 512x512x3 rgb image (0-255)
        :return:
        """
        # convert image [255, 255, 255, ...] to base64
        try:
            image = np.array(image).reshape(512, 512, 3)
        except ValueError:
            logger.warning("Image is not the correct shape")
            return None

        # convert the image to torch tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        if settingsManager.is_cuda:
            image = image.cuda()
        image = image.half()
        # convert the image to base64
        image = 2. * image - 1.
        if image is None:
            return False
        image = image.to(settingsManager.device)
        image = repeat(image, '1 ... -> b ...', b=batch_size)

        return image

    def load_image(self, image, batch_size=1, half=True):
        """
        :param image: a one dimensional array of a 512x512x3 rgb image (0-255)
        :return:
        """
        # convert image [255, 255, 255, ...] to base64
        try:
            image = np.array(image).reshape(*settingsManager.load_image_shape)
        except ValueError as e:
            logger.error(e)
            return None

        # convert the image to torch tensor
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        if settingsManager.is_cuda and half:
            image = image.cuda()
        if half:
            image = image.half()
        # convert the image to base64
        image = 2.*image - 1.
        if image is None:
            return False
        image = image.to(settingsManager.device)
        image = repeat(image, '1 ... -> b ...', b=batch_size)
        return image

    def preprocess_mask(self, mask):
        mask = self.load_image(mask, 1)
        return mask

    def handle_exception(self, e):
        logger.error(e)
        return False

    def handle_sample(self, sample):
        if settingsManager.fast_sample:
            self.last_sample = sample
            data = self.prepare_image(sample, True)
            self.image_handler(data, self.opt)
        # TODO: send message to client that sample is complete

    # def upscale(self, options, image_handler):
    #     settingsManager.parse_options(options)
    #     img = np.array(settingsManager.init_img).reshape(
    #         (512, 512, 3)
    #     ).astype(np.uint8)
    #     # swap color channels to BGR
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     enhanced = self.upscale_image(img)
    #     enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    #     image_handler(enhanced, self.opt)

    def move_gfpgan_to_device(self, device):
        self.gfp.gfpgan.to(device)
        self.gfp.face_helper.face_det.to(device)
        self.gfp.face_helper.face_parse.to(device)

    def upscale_image(self, sample):
        logger.info("Upscaling image")
        self.move_upscale_model_to_cuda()

        # Upscale the image
        logger.info("Getting sample from do_supersample")
        sample = self.do_supersample(sample)

        # Save image
        # cv2.imwrite(
        #     os.path.join(HERE, "test.png"),
        #     cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        # )

        # Move the upscale model to the CPU
        self.move_upscale_model_to_cpu()
        utils.clear_cache()

        return sample

    def process_face_enhancement(self, sample):
        if self.do_face_enhancement:
            logger.info("Enhancing faces")
            # sample = cv2.resize(
            #     sample,
            #     (512, 512),
            #     interpolation=cv2.INTER_AREA
            # )
            self.move_gfpgan_to_device(settingsManager.device)
            sample = self.do_gfpgan(sample)
            # self.move_gfpgan_to_device("cpu")
            utils.clear_cache()
        return sample

    def process_super_sample(self, sample):
        logger.info("Supersampling image...")
        if settingsManager.upscale:
            sample = self.upscale_image(sample)
        else:
            logger.info("Super sampler is disabled")
        return sample

    def process_resize_image(self, sample):
        """
        Set image to final size
        :param sample:
        :return:
        """
        sample = cv2.resize(
            sample,
            (settingsManager.final_width, settingsManager.final_height),
            interpolation=cv2.INTER_AREA
        )
        return sample

    def post_process(self, sample):
        logger.info("Running post process")
        self.move_generator_to_cpu()
        sample, has_nsfw = self.do_nsfw_filter(sample)
        sample = self.prepare_image_fast(sample)
        # sample = sample.astype(np.float32)
        if not has_nsfw:
            # sample = self.process_face_enhancement(sample)
            sample = self.process_super_sample(sample)
        else:
            logger.info("Image has NSFW content")
        # print sample type
        # move numpy.ndarray to cpu
        # change sample.dtype to numpy.dtype[float32]
        sample = sample.astype(np.float32)
        sample = self.process_resize_image(sample)

        # Save image to png
        cv2.imwrite(
            os.path.join(HERE, "supersamples", "tmp.png"),
            cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        )

        self.move_generator_to_cuda()
        return sample

    def do_nsfw_filter(self, sample):
        has_nsfw = False
        if settingsManager.do_nsfw_filter:
            logger.info("Running NSFW filter")
            sample, has_nsfw = self.nsfw_filter.filter_nsfw_content(sample)
            has_nsfw = has_nsfw[0]
        return sample, has_nsfw

    _super_sampler = None

    @property
    def do_face_enhancement(self):
        return settingsManager.gfpgan

    @property
    def super_sampler(self):
        if self._super_sampler is None:
            self._super_sampler = SuperSampler()
        return self._super_sampler

    @property
    def super_sample_model(self):
        return self.super_sampler.model

    def do_supersample(self, sample):
        return SuperSampler().run(sample)

    def move_upscale_model_to_cpu(self):
        self.super_sample_model.to(torch.device("cpu"))

    def move_upscale_model_to_cuda(self):
        self.super_sample_model.to(settingsManager.device)
        self.super_sample_model.eval()

    def move_generator_to_cpu(self):
        self.model.to(torch.device("cpu"))
        self.current_sampler = self.current_sampler.model.to(torch.device("cpu"))
        if hasattr(self.current_sampler, "first_stage_model"):
            self.current_sampler.first_stage_model = self.current_sampler.first_stage_model.to(torch.device("cpu"))
        self.nsfw_filter.switch_device("cpu")
        utils.clear_cache()

    def move_generator_to_cuda(self):
        self.model.to(settingsManager.device)
        self.current_sampler = self.current_sampler.model.to(settingsManager.device)
        if hasattr(self.current_sampler, "first_stage_model"):
            self.current_sampler.first_stage_model = self.current_sampler.first_stage_model.to(settingsManager.device)
        self.nsfw_filter.switch_device(settingsManager.device_name)

    def do_gfpgan(self, image):
        logger.info("Running gfpgan")
        cropped, restored_faces, restored_img = self.gfp.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5
        )
        return restored_img

    def load_config(self):
        self.config = OmegaConf.load(self.STABLE_DIFFUSION_CONFIG)

    def load_model_from_config(self, config, ckpt=None):
        path = self.CHECK_POINT_PATH
        logger.info(f"Loading model from {path}")
        return instantiate_from_config(config.model)

    def load_safety_feature_extractor_pickle(self):
        import pickle
        with open(
                self.SAFETY_FEATURE_EXTRACTOR_PICKLE_OUT,
                "rb"
        ) as f:
            obj = pickle.load(f)
        return obj

    def load_safety_feature_extractor(self):
        try:
            return self.load_safety_feature_extractor_pickle()
        except Exception as exc:
            logger.error("Failed to load safety feature extractor")

    def initialize_logging(self):
        # create path and file if not exist
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)