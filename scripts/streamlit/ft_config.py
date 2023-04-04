import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageChops
import random
import math
import json as js
import json
import cv2
import numpy as np
import base64
import urllib

mtface = None

class FTConfig(object):
    """dream booth 参数"""
    def __init__(self, ):
        self.lr = 5e-5
        self.instance_dir = '/www/simple_ssd/lxn3/mtdreambooth/plugins/dreambooth/instance/hat5'
        self.instance_prompt = 'a sks cap'
        self.with_prior_preservation = True
        self.class_prompt = 'a cap'
        self.class_data_dir = '/www/simple_ssd/lxn3/mtdreambooth/plugins/dreambooth/class/outputhatonly5'     
        self.resolution = 768
        self.center_crop = False
        self.lr_scheduler = "constant_with_warmup"
        self.lr_warmup_steps = 50
        self.gradient_accumulation_steps = 1
        self.max_train_steps = 200
        self.num_train_epochs = 1
        self.face_detect = False

def collate_fn(examples, tokenizer, with_prior_preservation=True): #dataset 后处理 函数
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation: # 一个batch的数据里面包含class数据（预训练知识保留）和instance（新知识学习）
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
    return batch

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        self.instance_images_path = [x for x in list(Path(instance_data_root).iterdir()) if str(x).endswith(".jpg")]
        self.instance_jsons_path = [str(x) + ".json" for x in list(self.instance_images_path)]
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def image_offset(self, Img, xoff, yoff):
        width, height = Img.size
        c = ImageChops.offset(Img, xoff, yoff)
        c.paste((0, 0, 0), (0, 0, xoff, height))
        c.paste((0, 0, 0), (0, 0, width, yoff))
        return c

    def get_face_image(self, instance_image, box, roll, width, height):
        box[0], box[2] = box[0] * width, box[2] * width
        box[1], box[3] = box[1] * height, box[3] * height
        cx = box[0] + box[2] / 2
        cy = box[1] + box[3] / 2



        expect_ratio = 0.15

        ratio = (box[2] * box[3]) / (height * height)
        if ratio < expect_ratio:
            new_height = math.sqrt(box[2] * box[3] / expect_ratio)
            new_width = math.sqrt(box[2] * box[3] / expect_ratio)
        else:
            new_width = width
            new_height = height

        tx = cx - new_height / 2
        ty = cy - new_height / 2

        face_rect = (max(0, tx), max(0, ty), new_width, new_height)
        #随机旋转
        instance_image = instance_image.rotate(random.uniform(-10, 10) + roll, center=(face_rect[0] + face_rect[2] / 2, face_rect[1] + face_rect[3] / 2))
        # instance_image = self.image_offset(instance_image, int(random.uniform(-10, 10)), int(random.uniform(-10, 10)))
        face_image = instance_image.crop(box=(int(face_rect[0]), int(face_rect[1]), int(face_rect[0] + face_rect[2]), int(face_rect[1] + face_rect[3])))
        if random.random() < 0.5:
            face_image.transpose(Image.FLIP_LEFT_RIGHT)

        # cv_img = cv2.cvtColor(np.asarray(face_image), cv2.COLOR_RGB2BGR)
        # # cv2.rectangle(cv_img, (int(box[0]), int(box[1])),
        # #               (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 2)
        # cv2.imshow("", cv_img)
        # cv2.waitKey()

        return face_image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        with open(self.instance_jsons_path[index % self.num_instance_images], "r") as f:
            body = js.load(f)
            instance_box = body["box"]
            instance_roll = math.degrees(body["roll"])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        instance_image = self.get_face_image(instance_image, instance_box, instance_roll, instance_image.width, instance_image.height)
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example

def get_face_rect(box, width, height):
    cx = box[0] + box[2] / 2
    cy = box[1] + box[3] / 2

    if width < height:
        return (0, max(0, cy - width / 2), width, width), (
        float(box[0]) / width, float(box[1] - max(0, cy - width / 2)) / width, float(box[2]) / width,
        float(box[3]) / width)
    else:
        tx = cx - height / 2
        return (max(0, tx), 0, height, height), (
        float(box[0] - max(0, tx)) / height, float(box[1]) / height, float(box[2]) / height, float(box[3]) / height)

def detect_face(image):
    global detector
    # 其它模型

    # 5. 进行检测
    feature = mtface.mtface_feature_t()
    feature = mtface.mtface_detector_detect(detector, image, feature)
    boxes = []
    # ages = []
    # genders = []
    rolls = []

    # 6. 获取检测结果
    for i in range(mtface.mtface_feature_get_face_size(feature)):
        # 6.1 get bounding box(type: numpy.array)
        box = mtface.mtface_feature_get_face_box(feature, i).astype(np.int32)
        # age, gender = mtface.mtface_feature_get_face_attribute(feature, i, [mtface.mtface_attr_age, mtface.mtface_attr_gender_male])
        _, _, roll = mtface.mtface_feature_get_face_pose_euler(feature, i)
        boxes.append(box)
        # ages.append(age)
        # genders.append(gender)
        rolls.append(roll)
    return boxes, rolls


def resize_image(image, size=1024): # 把人脸居中 且 裁成 1024*1024
    boxes, rolls = detect_face(image)
    if len(boxes) <= 0:
        return None, None, None
    box = boxes[0]
    # age = ages[0]
    # gender = genders[0]
    roll = rolls[0]
    face_size, box = get_face_rect(box, image.shape[1], image.shape[0])
    face_image = cv2.resize(
        image[int(face_size[1]): int(face_size[1] + face_size[3]), int(face_size[0]): int(face_size[0] + face_size[2])],
        (size, size))

    return face_image, box, roll

def encodeJson_test(image_dir):
    """
    for test
    :param image_dir:
    :return:
    """

    data = {}
    data["parameter"] = {}
    data["media_info"] = {}

    data["parameter"]["rsp_media_type"] = "base64"

    with open(image_dir, mode='rb') as file:
        img = file.read()
    img_list = base64.b64encode(img).decode('ascii')

    data["media_info_list"] = []
    media_info = {}
    media_info["media_data"] = img_list
    media_info["media_profiles"] = {
        "media_data_type": "image"
    }
    data["media_info_list"].append(media_info)
    jsondata = json.dumps(data)
    return jsondata


def load_image(image_data, type):
    """
     load image-base64 or url
    :param image_data:
    :return: image-cv2
    """
    # first we try to decode image

    # if any error, we go to url download
    if 'jpg' in image_data:
        image_data = encodeJson_test(image_data)
        image_data = json.loads(image_data)['media_info_list'][0]['media_data']

    if type == "jpg" or type == "JPG" or type == "png" or type == "PNG":
        modelImgStr = base64.b64decode(image_data)
        modelImg = np.fromstring(modelImgStr, dtype=np.uint8)
        # raise()
    else:
        # try url download
        print("download =====")
        try:
            url = image_data
            request = urllib.request.Request(url)
            resp = urllib.request.urlopen(request, timeout=10)
            print("download")
        except urllib.request.URLError as e:
            print("img download error:", url)
            print(e)
            return None, [json.dumps({"ErrorCode": 20014, "ErrorMsg": "NOT_FOUND"}), 20014, "NOT_FOUND"]
        except urllib.request.HTTPError as e:
            print("img download error:", url)
            print(e)
            return None, [json.dumps({"ErrorCode": 20014, "ErrorMsg": "NOT_FOUND"}), 20014, "NOT_FOUND"]
        except:
            return None, [json.dumps({"ErrorCode": 20014, "ErrorMsg": "NOT_FOUND"}), 20008, "NOT_FOUND"]
        raw = resp.read()
        modelImg = np.asarray(bytearray(raw), dtype="uint8")

    print(modelImg)

    if isinstance(modelImg, np.ndarray):
        modelImg = cv2.imdecode(modelImg, 1)
    return modelImg, None