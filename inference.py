import argparse
import os
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model():
    print("Loading model... (may take a minute)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("Model loaded!")
    return model, processor


def predict_answer(model, processor, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image_path}",
                },
                {
                    "type": "text",
                    "text": """This image contains a multiple choice question about deep learning.
The options are labeled A, B, C, D.

Read the question and all four options carefully, then select the correct answer.

You MUST reply with ONLY a single digit:
1 -> if the answer is A
2 -> if the answer is B
3 -> if the answer is C
4 -> if the answer is D
5 -> if you are not confident enough to answer

Do not write anything else. Output exactly one digit only."""
                }
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    generated = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0].strip()

    print(f"    Model raw output: '{generated}'")

    # Safely parse output — default to 5 (skip) to avoid hallucination penalty
    option = 5
    for char in generated:
        if char in '12345':
            option = int(char)
            break

    return option


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Absolute path to the test directory')
    args = parser.parse_args()

    test_dir = args.test_dir
    print(f"Test directory: {test_dir}")

    # Read test.csv
    test_csv_path = os.path.join(test_dir, 'test.csv')
    test_df = pd.read_csv(test_csv_path)
    print(f"Found {len(test_df)} questions")

    # Handle different possible column names
    if 'image_name' in test_df.columns:
        image_col = 'image_name'
    elif 'image_id' in test_df.columns:
        image_col = 'image_id'
    else:
        image_col = test_df.columns[0]

    print(f"Using column: '{image_col}'")

    # Load model
    model, processor = load_model()

    results = []
    for idx, row in test_df.iterrows():
        image_name = str(row[image_col]).strip()
        image_path = os.path.join(test_dir, 'images', f'{image_name}.png')

        print(f"\n[{idx+1}/{len(test_df)}] {image_name}")

        if not os.path.exists(image_path):
            print(f"  WARNING: image not found, defaulting to 5")
            option = 5
        else:
            option = predict_answer(model, processor, image_path)

        print(f"  -> Answer: {option}")
        results.append({
            'id': image_name,
            'image_name': image_name,
            'option': option
        })

    # Save submission.csv in current directory
    submission_df = pd.DataFrame(results)
    submission_df.to_csv('submission.csv', index=False)
    print(f"\nDone! submission.csv saved with {len(results)} rows.")
    print(submission_df.to_string())


if __name__ == '__main__':
    main()