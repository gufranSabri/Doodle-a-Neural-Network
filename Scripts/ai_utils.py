import os
import openai
import keras_ocr

def build_code(prompt):
    openai.api_key = "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"

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    print(completion.choices[0].message)
    return completion

def get_text(img_dir):
    pipeline = keras_ocr.pipeline.Pipeline()
    images = os.listdir(img_dir)
    res = []

    for i in images:
        if i == "res.png": continue
        img = keras_ocr.tools.read(os.path.join(img_dir, i))
        prediction_groups = pipeline.recognize([img])
        res.append({"name":i, "text":[]})

        predicted_image = prediction_groups[0]
        for text, _ in predicted_image:
            res[-1]["text"].append(text)
    
    return res