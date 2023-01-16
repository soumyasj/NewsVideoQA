## Training / Testing BERT QA models on NewsVideoQA

For finetuning pretrained BERT models, we use simpletransformers which in turn is based on transformers package. 

### Installing simpletransformers
For installing simpletransformers please follow original instructions We dont use fp16 while finetuning BERT models in our experiments. You need not install Apex if you dont want to use fp16 training.

### Fine tune a pretrained BERT model on DocVQA
See the script finetune_pretrained_model.py

### Make predictions using an existing BERT QA model
More details can be found in test_model.py
``` python test_model.py ```


(**Note that the data should be in SQUAD format)
