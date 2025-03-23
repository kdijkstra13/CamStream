from typing import List

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from PIL import Image
import re
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

class LavaInfer():
    def __init__(self, model_type="small"):
        disable_torch_init()
        # class variables
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.model_name = None
        self.conv_mode = None

        # params
        self.temperature = 0
        self.top_p = None
        self.num_beams = 1
        self.max_new_tokens = 512

        # init model
        self.create_model(model_type)
        self.set_conv_mode()

    def create_model(self, model_type):
        if model_type == "small":
            model_path = "liuhaotian/llava-v1.5-7b"
        elif model_type == "medium":
            model_path = "liuhaotian/llava-v1.5-13b"
        elif model_type == "large":
            model_path = "liuhaotian/llava-v1.5-13b-lora"
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=self.model_name
        )

    def set_conv_mode(self, conv_mode=None):
        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        if conv_mode is not None:
            if conv_mode != self.conv_mode:
                print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                        self.conv_mode, conv_mode, self.conv_mode)
                )
            self.conv_mode = conv_mode

    def to_query(self, text):
        qs = text
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        return qs


    def infer(self, text: str, images: List):
        qs = self.to_query(text)

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_sizes = [x.size for x in images]
        images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

def run_llava():

    image_files = ["/home/klaas/temp/zetros/far_left.jpg",
                   "/home/klaas/temp/zetros/left.jpg",
                   "/home/klaas/temp/zetros/left2.jpg",
                   "/home/klaas/temp/zetros/center.jpg",
                   "/home/klaas/temp/zetros/right.jpg",
                   "/home/klaas/temp/zetros/far_right.jpg",]

    llava = LavaInfer("small")

    for image_file in image_files:
        answer = llava.infer(subst('shoe', "simple"), [Image.open(image_file)])
        print(answer)

