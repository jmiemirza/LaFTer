import os.path as osp
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
from tqdm import tqdm
from pathlib import Path
_tokenizer = _Tokenizer()
from utils.model_utils import *

class CLIP_Zero_Shot_adapt(nn.Module):

    def __init__(self, model, classes, templates, device='cuda', dataset_name=None, log=None, txt_cls = None, cfg=None):
        super(CLIP_Zero_Shot_adapt, self).__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.device = device
        self.classes = classes
        self.model = model.to(device)
        self.log = log
        self.args = None
        self.txt_cls = txt_cls
        self.templates = templates
        self.txt_feas = self.txt_features(self.classes, self.templates)
        self.backbone_out_size = 512
        self.text_adapter = nn.Sequential(nn.Linear(int(self.backbone_out_size), len(classes), bias=False)).to(device)
        self.txt_features_for_text_cls, self.labels_for_text_cls = self.txt_features_for_text_cls()

    def txt_features(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # format with class
                texts = tokenize(texts).cuda()  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def txt_features_for_text_cls(self):

        if self.txt_cls== '0':
            gpt3_prompts = None
            desc, labels_for_descriptions = gen_labels_with_classes(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == '1':
            gpt3_prompts = self.templates

            desc, labels_for_descriptions = gen_labels_with_templates(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == '2':
            # targetted_prompts
            path_to_file = f'./descriptions/tap/{self.dataset_name}.json'

            with open(path_to_file) as f:
                gpt3_prompts = json.load(f)

            desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)

        elif self.txt_cls == '3':
            # generic prompts
            path_to_file = f'./descriptions/generic/{self.dataset_name}.json'

            with open(path_to_file) as f:
                gpt3_prompts = json.load(f)

            desc, labels_for_descriptions = gen_labels_with_descrptions(self.classes, descriptions=gpt3_prompts)


        else:
            raise ValueError('Invalid txt_cls argument')

        Path(f'embeddings_icassp').mkdir(parents=True, exist_ok=True)

        if os.path.isfile(f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt'):

            zeroshot_weights = torch.load(f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt')
            print('******** Loaded Already Saved Embeddings *********')
            labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

        else:
            print('******** No Embeddings Found --- Saving New Embeddings *********')

            labels_for_descriptions = torch.tensor(labels_for_descriptions).cuda()

            zeroshot_weights = []
            with torch.no_grad():
                for classname in tqdm(desc):
                    text = tokenize(classname).cuda()  # tokenize # (50, 77) --> 50 templates/texts from GPT
                    class_embeddings = self.model.encode_text(
                        text)  # embed with text encoder # (50, 512) --> embeddings for all 50 texts
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)  # L2 norm of the embeddings (dim 2)
                    zeroshot_weights.append(class_embeddings)
                zeroshot_weights = torch.stack(zeroshot_weights).cuda()  # (512, 10) --> 512 embeddings for 10 classes'
                torch.save(zeroshot_weights, f'embeddings/{self.txt_cls}_{self.dataset_name}_embeddings.pt')

        return zeroshot_weights.squeeze(), labels_for_descriptions


    def train_txt_clas(self, criteria):
        noise_std = 0.1
        noise = torch.randn(self.txt_features_for_text_cls.shape) * noise_std
        txt_feas = self.txt_features_for_text_cls
        txt_label = self.labels_for_text_cls
        feas = (self.text_adapter(txt_feas.to(torch.float32) + noise.cuda()))
        loss = criteria(feas, txt_label)
        return loss

    def eval_text_adapter(self, x1):
        with torch.no_grad():
            img_features_1 = self.image_features(x1.float())
            img_features_1 = self.text_adapter(img_features_1)
            return img_features_1

    def image_features(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def forward(self, x1):
        with torch.no_grad():
            img_features = self.image_features(x1)
            out = img_features.float() @ self.txt_feas.float()
        return out


@TRAINER_REGISTRY.register()
class clip_adapt(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")
        self.model = CLIP_Zero_Shot_adapt(model=clip_model, classes=classnames,
                                          templates=['a photo of a {}'], dataset_name = cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg)
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=tr_transforms)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def parse_batch_train(self, batch):

        if isinstance(batch, list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
            input = input.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
