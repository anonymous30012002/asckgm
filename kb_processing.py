import json
import torch
import transformers
from transformers import BertTokenizer, BertModel, BertConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import torch.nn as nn
import glob
import argparse

tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")
model=BertModel.from_pretrained("bert-base-uncased")
# encoder_config = BertConfig(
#     num_hidden_layers=2,
#     vocab_size=50000,
#     hidden_size=512,
#     num_attention_heads=8
# )
# model = BertModel(encoder_config)

model=model.to('cuda')
print(model)
def review_func(dialogues):
	review_list=[]
	# print(dialogues)
	for item in dialogues:
		# print(item)
		if item['utterance']['kb']!='null':
			for r in item['utterance']['kb']:
				if r==[]:
					continue
				else:
					for k in r:
						review_list.append(k['review_body'])
	return review_list

def bert_token(sentence):
	inputs = tokenizer(sentence,padding=True, truncation=True,return_tensors="pt")
	inputs=inputs.to('cuda')
	outputs= model(**inputs)
	# print(outputs)
	emb= outputs[1]
	return emb

def main(args):
	all_files=glob.glob(args.file_dir+"/*.json")
	print("Reading files")
	print("writing files")

	for dialogue_json in all_files:
		# print(dialogue_json)
		out_file= dialogue_json.split('/')[-1]
		# print(type(args.out_dir_path +'/'+out_file))
		dialogues=json.load(open(dialogue_json))
		#print(dialogues)
		review_list=review_func(dialogues)
		# print(review_list)
		if review_list !=[]:
			for item in dialogues:
				kb_rev_list=[]

				# print(item['utterance']['nlg'])
				sentence=bert_token(item['utterance']['nlg'])
				max_val=0
				cos_list=[]
				for rev in review_list:
					# print(rev)
					review_token=bert_token(rev)
					cos = nn.CosineSimilarity(dim=1, eps=1e-6)
					cos_sim= cos(sentence,review_token)
					cos_list.append(cos_sim)
					if cos_sim>max_val:
						max_val=cos_sim

				for index,num in enumerate(cos_list):
					if num==max_val:
						# print(index)
						# print(review_list[index])
						kb_rev=review_list[index]
						# print(kb_rev)
						item['utterance']['kb']= kb_rev
					else:
						continue

				with open(args.out_dir_path+'/'+out_file, "w") as outfile:
					json.dump(dialogues, outfile)
		else:
			with open(args.out_dir_path+'/'+out_file, "w") as outfile:
				json.dump(dialogues, outfile)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-file_dir', help='Input file directory path')
	parser.add_argument('-out_dir_path', type=str, help='out_dir_path')
	args = parser.parse_args()
	main(args)