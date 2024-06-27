import argparse
from transformers import AutoTokenizer 
from transformers import TrainingArguments

from src.dataset import ArticleDataset
from src.model import HuggingfaceModel
from src.train import Trainer

def main(args):
    dataset = ArticleDataset(args.dataset_path)
    data = dataset.create_datasets()
    
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True,)
    # set pad token - to avoid error while training
    tokenizer.pad_token = tokenizer.eos_token

    hf_model = HuggingfaceModel()
    model = hf_model.from_pretrained(args.hf_model_path)

    trainer = Trainer()

    if  args.auto_summarization:
        text = trainer.test("", model, tokenizer)
        print(text)

    else:
        train_args = TrainingArguments(
                    per_device_train_batch_size=args.device_per_batch_size,
                    gradient_accumulation_steps=1,
                    max_steps=args.max_steps,
                    learning_rate=args.learning_rate,
                    fp16=True,
                    logging_steps=50,
                    logging_dir=f"{args.save_dir}/logs",
                    output_dir=f"{args.save_dir}",
                )

        trainer.train(model, data, tokenizer, train_args=train_args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data.json")
    parser.add_argument("--hf_model_path", type=str, default="beomi/KoAlpaca-Polyglot-5.8B")
    parser.add_argument("--load_checkpoint", type=str, default="./model_checkpoints")
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--device_per_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./model_save")
    parser.add_argument('--auto_summarization', action='store_true')
    parser.add_argument('--input_url', str=str, default="https://n.news.naver.com/article/022/0003945184?cds=news_media_pc&type=editn")

    args = parser.parse_args()
    main(args)

