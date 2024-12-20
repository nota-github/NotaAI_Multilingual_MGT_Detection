import os
import torch
import langid
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import LLMFeatureExtractor
from model import MGTDetectionModel

import warnings

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(args, extractor: LLMFeatureExtractor):
    seen_lang = ["en", "ru", "de", "zh", "ar", "bg", "id", "ur", "it"]
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    unseen_model = MGTDetectionModel()
    unseen_model.load_state_dict(
        torch.load(
            "MGTDetectionModel.pt",
            weights_only=True,
            map_location=torch.device("cuda"),
        )
    )
    unseen_model = unseen_model.eval().to("cuda")

    seen_model = AutoModelForSequenceClassification.from_pretrained(
        "multilingual-e5-large-MGT-2",
        device_map="auto",
    ).eval()

    input_text = args.input_text
    lang, _ = langid.classify(input_text)
    tokens = tokenizer(input_text, return_tensors="pt")

    if lang in seen_lang:
        tokens = tokens.to("cuda")
        with torch.no_grad():
            outputs = seen_model(**tokens)

    else:
        input_dict = extractor.feature_extract(input_text)

        input_tensors = {k: v.to("cuda") for k, v in input_dict.items()}
        with torch.no_grad():
            outputs = unseen_model(**input_tensors)

    logit = outputs.logits[0]
    pred = logit.argmax().item()

    res = "Human" if pred == 0 else "Machine"
    print(f"{args.input_text} > {res} generated text")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str)
    parser.add_argument("--hf_token", type=str)
    args = parser.parse_args()
    extractor = LLMFeatureExtractor(args.hf_token)
    main(args, extractor)
