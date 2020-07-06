from fast_bert.data_cls import BertDataBunch

DATA_PATH = "data"
LABEL_PATH = "label"

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='train_sample.csv',
                          val_file='val_sample.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=[	'location',
										'chat',
										'time',
										'personal',
										'camera',
										'app',
										],
                          batch_size_per_gpu=1,
                          max_seq_length=512,
                          multi_gpu=False,
                          multi_label=True,
                          model_type='bert')


import torch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging

logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]
OUTPUT_DIR = "output"

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=100,
						multi_gpu=False,
						is_fp16=False,
						multi_label=True,
						logging_steps=50)


learner.fit(epochs=10,
			lr=6e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")


learner.save_model()

# import pickle
# with open('learner.pickle', 'wb') as f:
# 	pickle.dump(learner, f)


texts = ['where am i',
		 'what is my location',
		 'what do i like most',
		 'where are the restaurants',
		 'what snacks might be available',
		 'what are the snacks']
predictions = learner.predict_batch(texts)

print(predictions)
