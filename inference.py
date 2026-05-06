import argparse
import os
import re
import torch
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model():
    print("Loading Qwen2.5-VL-72B-Instruct-AWQ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        local_files_only=True,
    )
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
                    "text": (
                        "This image contains a multiple choice question about deep learning.\n"
                        "The options are labeled A, B, C, D.\n\n"
                        "Carefully analyze the question step by step:\n"
                        "1. Identify exactly what concept or fact is being tested\n"
                        "2. Reason through each option using your deep learning knowledge\n"
                        "3. Eliminate incorrect options and confirm the best answer\n\n"
                        "At the very end of your response, write EXACTLY this line:\n"
                        "FINAL ANSWER: <digit>\n\n"
                        "Where digit means:\n"
                        "1 = A is correct\n"
                        "2 = B is correct\n"
                        "3 = C is correct\n"
                        "4 = D is correct\n"
                        "5 = not confident enough to answer"
                    )
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
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    generated = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    print(f"    Model output:\n{generated}\n")

    # Primary: parse FINAL ANSWER: X
    match = re.search(r"FINAL ANSWER:\s*([1-5])", generated, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback: take the last valid digit in the response
    for char in reversed(generated):
        if char in "12345":
            return int(char)

    return 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Absolute path to the test directory")
    args = parser.parse_args()

    test_dir = args.test_dir
    print(f"Test directory: {test_dir}")

    test_csv_path = os.path.join(test_dir, "test.csv")
    test_df = pd.read_csv(test_csv_path)
    print(f"Found {len(test_df)} questions")

    if "image_name" in test_df.columns:
        image_col = "image_name"
    elif "image_id" in test_df.columns:
        image_col = "image_id"
    else:
        image_col = test_df.columns[0]

    print(f"Using column: '{image_col}'")

    model, processor = load_model()

    results = []
    for idx, row in test_df.iterrows():
        image_name = str(row[image_col]).strip()
        image_path = os.path.join(test_dir, "images", f"{image_name}.png")

        print(f"\n[{idx+1}/{len(test_df)}] {image_name}")

        if not os.path.exists(image_path):
            print("  WARNING: image not found, defaulting to 5")
            option = 5
        else:
            try:
                option = predict_answer(model, processor, image_path)
            except Exception as e:
                print(f"  ERROR during inference: {e}, defaulting to 5")
                option = 5

        print(f"  -> Answer: {option}")
        results.append({"id": image_name, "image_name": image_name, "option": option})

    submission_df = pd.DataFrame(results)
    submission_df.to_csv("submission.csv", index=False)
    print(f"\nDone! submission.csv saved with {len(results)} rows.")
    print(submission_df.to_string())


if __name__ == "__main__":
    main()