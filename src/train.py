import os
from transformers import TrainingArguments

import transformers
from .tools.text_gen import preprocessing_function, preprocessing_function_test

class Trainer:
    def train(self, model, data, tokenizer, train_args=None):

        col_to_delete = data['train'].to_pandas().columns.to_list()
        # Apply the preprocessing function and remove the undesired columns
        tokenized_datasets = data.map(
            preprocessing_function, 
            batched=False, 
            remove_columns=col_to_delete, 
            num_proc=int(os.cpu_count()/2),
            fn_kwargs={"tokenizer": tokenizer})
        # Rename the target to label as for HugginFace standards
        # tokenized_datasets = tokenized_datasets.rename_column("target", "label")
        # Set to torch format
        tokenized_datasets.set_format("torch")

        if train_args==None:
            train_args = TrainingArguments(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                max_steps=5150,
                learning_rate=1e-4,
                fp16=True,
                logging_steps=10,
                logging_dir="./logs",
                output_dir="./",
                # optim="paged_adamw_8bit"
            )

        # needed for gpt-neo-x tokenizer
        tokenizer.pad_token = tokenizer.eos_token

        trainer = transformers.Trainer(
            model=model,
            train_dataset=tokenized_datasets['train'],
            args=train_args,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train()        

    def test(self):
        pass


# def main(args):
#     dataset = ArticleDataset(args.dataset_path)
#     data = dataset.create_datasets()
    
#     tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True,)
#     # set pad token - to avoid error while training
#     tokenizer.pad_token = tokenizer.eos_token

#     hf_model = HuggingfaceModel()
#     model = hf_model.from_pretrained(args.hf_model_path)

#     trainer = Trainer()

#     train_args = TrainingArguments(
#                 per_device_train_batch_size=args.device_per_batch_size,
#                 gradient_accumulation_steps=1,
#                 max_steps=args.max_steps,
#                 learning_rate=args.learning_rate,
#                 fp16=True,
#                 logging_steps=50,
#                 logging_dir=f"{args.save_dir}/logs",
#                 output_dir=f"{args.save_dir}",
#                 # optim="paged_adamw_8bit"
#             )


#     trainer.train(model, data, tokenizer, train_args=train_args)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_path", type=str, default="./data.json")
#     parser.add_argument("--hf_model_path", type=str, default="beomi/KoAlpaca-Polyglot-5.8B")
#     parser.add_argument("--load_checkpoint", type=str, default="./model_checkpoints")
#     parser.add_argument("--max_steps", type=int, default=40000)
#     parser.add_argument("--learning_rate", type=float, default=1e-4)
#     parser.add_argument("--save_dir", type=str, default="./model_save")


#     args = parser.parse_args()
#     main(args)

