import torch
from diffusers import StableDiffusionPipeline

from surfer import ModelLoader


class BasicModelLoader(ModelLoader):
    @staticmethod
    def _get_pipe(model_id: str, device: str):
        # On GPU we load by default the model in half precision,
        # because it's faster and lighter.
        if device == "cuda":
            return StableDiffusionPipeline.from_pretrained(
                model_id,
                revision="fp16",
                torch_dtype=torch.float16,
            )

        return StableDiffusionPipeline.from_pretrained(model_id)

    def load_model(self, *args, **kwargs) -> any:
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._get_pipe(model_id, device)
