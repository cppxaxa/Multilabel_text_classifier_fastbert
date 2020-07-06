from fast_bert.prediction import BertClassificationPredictor

MODEL_PATH = "output/model_out"

LABEL_PATH = "label"

predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=True,
				model_type='bert',
				do_lower_case=True)

# Single prediction
# single_prediction = predictor.predict("where to get food")
# print(single_prediction)

for i in range(10):
	text = input("Enter: ")
	if len(text) < 2: break
	single_prediction = predictor.predict(text)
	print(single_prediction)
	print()


# # Batch predictions
# texts = [
# 	"this is the first text",
# 	"this is the second text"
# 	]

# multiple_predictions = predictor.predict_batch(texts)

# print(multiple_predictions)
