import torch
import langid
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import LLMFeatureExtractor
from model import MGTDetectionModel

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str)
    parser.add_argument("--hf_token", type=str)

    args = parser.parse_args()
    return args


def get_models():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    unseen_language_detection_model = MGTDetectionModel()
    unseen_language_detection_model.load_state_dict(
        torch.load(
            "MGTDetectionModel.pt",
            weights_only=True,
            map_location=torch.device("cuda"),
        )
    )
    unseen_language_detection_model = unseen_language_detection_model.eval().to("cuda")
    seen_language_detection_model = AutoModelForSequenceClassification.from_pretrained(
        "nota-ai/multilingual-e5-large-MGT-finetuned",
        device_map="auto",
    ).eval()

    model_dict = {"seen": seen_language_detection_model, "unseen": unseen_language_detection_model}

    return model_dict, tokenizer


def main(args, extractor: LLMFeatureExtractor):
    seen_lang = ["en", "ru", "de", "zh", "ar", "bg", "id", "ur", "it"]

    model_dict, tokenizer = get_models()

    input_text = args.input_text
    detected_language, _ = langid.classify(input_text)
    tokens = tokenizer(input_text, return_tensors="pt")

    if detected_language in seen_lang:
        tokens = tokens.to("cuda")
        with torch.no_grad():
            outputs = model_dict["seen"](**tokens)

    else:
        input_dict = extractor.feature_extract(input_text)

        input_tensors = {k: v.to("cuda") for k, v in input_dict.items()}
        with torch.no_grad():
            outputs = model_dict["unseen"](**input_tensors)

    logit = outputs.logits[0]
    pred = logit.argmax().item()

    res = "Human" if pred == 0 else "Machine"
    print(f"{args.input_text} > {res} generated text")


if __name__ == "__main__":
    args = parse_args()
    extractor = LLMFeatureExtractor(args.hf_token)
    main(args, extractor)
