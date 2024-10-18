from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
import os
import random
def modeul (image_path):
    processor = Pix2StructProcessor.from_pretrained('google/deplot')
    model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot').to("cuda:0")
    image_path = image_path 
    image = Image.open(image_path)
    inputs = processor(images=image, text="Generate underlying data table of the figure below", return_tensors="pt").to("cuda:0")
    predictions_list = [] 
    num_cases = random.randint(8, 10)  

    for i in range(num_cases):
        # 随机生成 do_sample 参数 (True 或 False)
        do_sample = random.choice([True, False])
        
        # 随机生成 num_beams 参数，范围是1到5之间的整数，或者设为None
        num_beams = random.choice([None, random.randint(1, 5)])
        
        # 随机生成 temperature 参数，范围是0.7到1.0之间的小数
        temperature = random.uniform(0.7, 1.0)
        
        # 随机生成 top_k 参数，范围是10到50之间的整数
        top_k = random.randint(10, 50)
        
        # 随机生成 top_p 参数，范围是0.8到1.0之间的小数
        top_p = random.uniform(0.8, 1.0)
        
        # 构建生成参数
        generate_params = {
            'max_new_tokens': 512,
            'do_sample': do_sample,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p
        }

        if num_beams is not None:
            generate_params['num_beams'] = num_beams

        predictions = model.generate(
            **inputs,
            **generate_params
        )

        # 解码预测结果
        decoded_prediction = processor.decode(
            predictions[0], skip_special_tokens=True
        )

        predictions_list.append({
            'image_index': i,  
            'do_sample': do_sample,
            'num_beams': num_beams,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'prediction': decoded_prediction,
        })

    # 打印预测列表中的内容
    for pred in predictions_list:
        print(f"Image {pred['image_index']} - do_sample: {pred['do_sample']}, num_beams: {pred['num_beams']}, temperature: {pred['temperature']}, top_k: {pred['top_k']}, top_p: {pred['top_p']}")
        print(f"Prediction: {pred['prediction']}")
    return predictions_list
