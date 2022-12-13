from stablediffusion.classes.settings_manager import settingsManager
from stablediffusion.classes.stablediffusionengine import StableDiffusionEngine
from stablediffusion.classes.settings import Txt2ImgArgs
from runai.logger import logger


class StableDiffusionRunner:
    """
    Run Stable Diffusion.
    """
    stablediffusion = None
    model = None
    device = None
    models = {}

    @property
    def generator(self):
        return self.models["generator"]

    @property
    def inpaint(self):
        return self.models["inpaint"]

    @property
    def gfpgan(self):
        return self.models["gfpgan"]

    @property
    def microsoft(self):
        return self.models["microsoft"]

    def process_data_value(self, key, value):
        """
        Process the data value. Ensure that we use the correct types.
        :param key: key
        :param value: value
        :return: processed value
        """
        if value == "true":
            return True
        if value == "false":
            return False
        if key in [
            "steps", "n_iter", "H", "W", "C", "f",
            "n_samples", "n_rows", "seed"
        ]:
            return int(value)
        if key in ["ddim_eta", "scale", "strength"]:
            return float(value)
        return value

    def process_options(self, options, data):
        """
        Process the data, compare aginast options.
        :param options: options
        :param data: data
        return: processed options
        """
        # get all keys from data
        keys = data.keys()
        logger.info("process_options keys ")
        for index, opt in enumerate(options):
            if opt[0] in keys:
                options[index] = (
                    opt[0],
                    self.process_data_value(
                        opt[0],
                        data.get(opt[0], opt[1])
                    )
                )
        return options

    def generator_sample(self, opts_in, image_handler):
        return self.do_sample("generator", opts_in, image_handler)

    def inpaint_sample(self, opts_in, image_handler):
        return self.do_sample("inpaint", opts_in, image_handler)

    def get_unprocessed_options(self, options_in):
        if options_in["reqtype"] == "Text2Image":
            return self.txt2img_options
        elif options_in["reqtype"] == "Image2Image":
            return self.img2img_options
        elif options_in["reqtype"] == "Inpainting":
            return self.txt2img_options
        else:
            raise Exception("Unknown request type")

    def get_sample_generator(self, model_name):
        if model_name == "generator":
            return self.generator
        elif model_name == "inpaint":
            return self.inpaint
        else:
            raise Exception("Unknown model name")

    def do_sample(self, model_name, opts_in, image_handler):
        self.activate_model(model_name)
        unprocessed_opts = self.get_unprocessed_options(opts_in)
        processed_opts = self.process_options(unprocessed_opts, opts_in)
        sample_generator = self.get_sample_generator(model_name)
        return sample_generator.sample(
            options=processed_opts,
            image_handler=image_handler,
            reqtype=opts_in["reqtype"]
        )

    def post_sample(self, options, image):
        pass

    def cancel(self):
        """
        Cancels a request even if image is generating.
        :return: None
        """
        if self.generator.current_sampler:
            self.generator.current_sampler.cancel_event = True

    def activate_model(self, model_name):
        # if not settingsManager.is_cuda:
        #     return
        # logger.info("Activating model %s" % model_name)
        #
        # # move everything that isn't the selected model to cpu
        # for key, model in self.models.items():
        #     if not model:
        #         continue
        #     if key != model_name:
        #         model.change_device("cpu")
        #
        # # move the selected model to cuda
        # self.models[model_name].change_device("cuda")
        pass

    def change_model(self, model):
        logger.info("Switching models to %s" % model)
        if self.generator.device == "cuda":
            self.generator.change_device("cpu")
            self.inpaint.change_device("cuda")
        else:
            self.inpaint.change_device("cpu")
            self.generator.change_device("cuda")

    def load(self):
        logger.info("sdrunner load")
        if self.txt2img_options is None:
            raise Exception("txt2img_options is required")
        if self.img2img_options is None:
            raise Exception("img2img_options is required")

        self.models = {
            "generator": StableDiffusionEngine(
                options=self.txt2img_options,
                device=self.device,
                args=self.t2iargs,
                model_name=self.model_name,
                model_version=self.model_version,
                safety_model_path=self.safety_model_path,
                safety_feature_extractor_path=self.safety_feature_extractor_path
            ),
            # "inpaint": StableDiffusionEngine(
            #     options=self.txt2img_options,
            #     device="cpu",
            #     args=self.t2iargs,
            #     model_name=self.model_name,
            #     model_version="%s-inpainting" % self.model_version,
            #     safety_model_path=self.safety_model_path,
            #     safety_feature_extractor_path=self.safety_feature_extractor_path
            # ),
            "gfpgan": None,
            "microsoft": None,
        }

    def __init__(self, *args, **kwargs):
        """
        Initialize the runner.
        """
        self.txt2img_options = kwargs.get("txt2img_options", None)
        self.img2img_options = kwargs.get("img2img_options", None)
        self.safety_model_path = kwargs.get("safety_model_path")
        self.safety_feature_extractor_path = kwargs.get("safety_feature_extractor_path")
        self.model_name = kwargs.get("model_name", "model.pt")
        self.model_version = kwargs.get("model_version", "v1-4")
        # start a txt2img loader instance
        self.t2iargs = Txt2ImgArgs
        self.t2iargs.append({
            "arg": "fast_sample",
            "type": bool,
            "nargs": "?",
            "default": True,
            "help": ""
        })
        self.load()
