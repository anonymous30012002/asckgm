import os
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import json
import pickle as pkl
import random
import time
import math
import numpy as np
import logging
import argparse
from annoy import AnnoyIndex
# Absolute imports
import mmd_final.utils.utils as utils
import mmd_final.modules.torch_utils as torch_utils
import mmd_final.modules.models as models

def check_cuda(seed):
	"""Check cuda"""
	if torch.cuda.is_available():
		use_cuda = True
		torch.cuda.manual_seed(seed)
	else:
		use_cuda = False
	return use_cuda
'''
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y
    
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)
'''
class NoamOpt:
	"Optim wrapper that implements rate."
	def __init__(self, model_size, factor, warmup, optim):
		self.optim = optim
		self._step = 0
		self.warmup = warmup
		self.factor = factor
		self.model_size = model_size
		self._rate = 0

	def zero_grad(self):
		self.optim.zero_grad()

	def step(self):
		"Update parameters and rate"
		self._step += 1
		rate = self.rate()
		for p in self.optim.param_groups:
			p['lr'] = rate
		self._rate = rate
		self.optim.step()

	def rate(self, step = None):
		"Implement `lrate` above"
		if step is None:
			step = self._step
		return self.factor * \
		(self.model_size ** (-0.5) *
			min(step ** (-0.5), step * self.warmup ** (-1.5)))


def optimizer_wrapper(model, optim_type, lr, weight_decay):
	"""Wrapper for different optimizers"""
	# TODO model.parameters() as input instead of model
	if optim_type == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optim_type == 'adadelta':
		optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optim_type == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optim_type == 'noam_opt':
		optimizer= NoamOpt(512, 2, 8000,
			optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))
	else:
		raise NotImplementedError("Not avaiable optimizer")
	# Initialize optimizers and criterion differently for encoder and decoder
	# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
	return optimizer

# Printing model paramters
def print_model(model):
	for name, param in model.named_parameters():
		if param.requires_grad:
			print (name)
	# params = list(model.parameters())
	# print(params)
	# https://discuss.pytorch.org/t/how-to-print-models-parameters-with-its-name-and-requires-grad-value/10778
	# for name, param in model.state_dict().items():
	# or 
	# for param in model.parameters():
	# 	print(param)
	### Printing model
	print(model)

def evaluate(valid_pkl_path, loss_criterion, model, config, vocab_size, vocab, annoyIndex, 
			annoyPkl, use_cuda=False, use_kb=False, use_aspect=False, use_sentiment= False,
			valid_aspect_path=None, valid_kb_path=None, valid_sentiment_path=None, valid_celeb_path=None):
	model.eval()
	# with torch.no_grad():
	valid_data = pkl.load(open(valid_pkl_path,'rb'))
	batch_size = config['data']['batch_size']
	total_samples = len(valid_data)
	num_valid_batch = int(math.ceil(float(total_samples)/float(batch_size)))
	total_loss =0.
	n_total_words = 0.
	correct = 0
	total = 0

	kb_len = None
	celeb_len = None
	kb_vec = None
	celeb_vec = None
	aspect_len=None
	aspect_vec=None
	sentiment_len = None
	sentiment_vec = None
	if use_kb:
		# celeb_data = pkl.load(open(valid_celeb_path,'rb'))
		kb_data = pkl.load(open(valid_kb_path,'rb'))
	if use_aspect:
		aspect_data = pkl.load(open(valid_aspect_path,'rb'))
	if use_sentiment:
		sentiment_data = pkl.load(open(valid_sentiment_path, "rb"))

	valid_start = time.time()
	for batch_id in range(num_valid_batch):
		batch_data = valid_data[batch_id*batch_size:(batch_id+1)*batch_size]
		if use_kb:
			kb_len = np.array(kb_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
			kb_len = utils.convert_states_to_torch(kb_len, use_cuda=use_cuda)
			kb_vec = np.array(kb_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
			kb_vec = utils.convert_states_to_torch(kb_vec, use_cuda=use_cuda)
			# Celebs
			# celeb_len = np.array(celeb_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
			# celeb_len = utils.convert_states_to_torch(celeb_len, use_cuda=use_cuda)
			# celeb_vec = np.array(celeb_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
			# celeb_vec = utils.convert_states_to_torch(celeb_vec, use_cuda=use_cuda)
		if use_aspect:
			aspect_len = np.array(aspect_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
			aspect_len =  utils.convert_states_to_torch(aspect_len, use_cuda=use_cuda)
			aspect_vec = np.array(aspect_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
			apect_vec = utils.convert_states_to_torch(aspect_vec, use_cuda=use_cuda)

		if use_sentiment:
			sentiment_len = np.array(sentiment_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
			sentiment_len =  utils.convert_states_to_torch(sentiment_len, use_cuda=use_cuda)
			sentiment_vec = np.array(sentiment_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
			sentiment_vec = utils.convert_states_to_torch(sentiment_vec, use_cuda=use_cuda)

		text_enc_input, text_enc_in_len, image_enc_input, dec_text_input,\
			dec_out_seq, dec_seq_length= utils.get_batch_mmd_data(batch_data, config['data']['start_id'],
						config['data']['end_id'], config['data']['pad_id'],
						config['data']['image_rep_size'], annoyIndex, annoyPkl,
						use_cuda=use_cuda, volatile=True)
		dec_output_prob = model(text_enc_input, image_enc_input, text_enc_in_len, 
						dec_text_input, dec_out_seq, context_size=args.context_size, 
						teacher_forcing_ratio=1, decode=False,use_cuda=use_cuda,
						kb_vec=kb_vec, celeb_vec=celeb_vec, kb_len=kb_len, celeb_len=celeb_len,
						aspect_vec=aspect_vec, aspect_len=aspect_len, sentiment_vec=sentiment_vec, sentiment_len=sentiment_len)
		'''
		if batch_id == 0:
			sentences=[]

			dec_output_prob_trans = model(text_enc_input, image_enc_input, text_enc_in_len, 
							context_size=args.context_size,
							teacher_forcing_ratio=0, decode=True, use_cuda=use_cuda,
							kb_vec=kb_vec, celeb_vec=celeb_vec, kb_len=kb_len,
							celeb_len=celeb_len)
			# print(dec_output_prob)
			dec_output_seq = dec_output_prob_trans[:,0,:].data.cpu().numpy()
			# loss = loss_criterion(dec_output_prob.contiguous().view(-1, vocab_size), #config['model']['tgt_vocab_size']),
			# 	dec_out_seq.view(-1))
			# dec_output_seq = dec_output_prob.data.cpu().numpy().argmax(axis=2) # argmax for each timestep
			for sequence in dec_output_seq:
				words = []
				for word_id in sequence:
					if word_id == config['data']['end_id']:
						break
					word = vocab[word_id]
					words.append(word)
				sentence = ' '.join(words)
				sentences.append(sentence)
			print(sentences)
		'''
		loss_val = loss_criterion(dec_output_prob.contiguous().view(-1, vocab_size), #config['model']['tgt_vocab_size']),
			dec_out_seq.view(-1))
		n_words = dec_seq_length.float().sum().data.item()
		n_total_words += n_words
		# batch_loss = loss_val.data[0]/n_words	
		total_loss += loss_val.data.item()
		dec_out_model = dec_output_prob.data.cpu().numpy().argmax(axis=2) # argmax for each timestep
		# print(dec_out_model)
		# print(dec_out_model.shape)

		# Multi task learning
		# _, predicted = torch.max(outputs.data, 1)
		# total += labels.size(0)
		# correct += (predicted == labels).sum().item()

	# Printing epoch loss		
	# epoch_loss = total_loss / num_valid_batch		
	epoch_loss = total_loss / n_total_words
	valid_elapsed = (time.time() - valid_start)/60
	# if (epoch+1) % config['training']['log_every'] == 0:
	print('Valid Loss: Loss: %.6f, Perplexity: %5.4f, Run Time:%5.4f'
		  %(epoch_loss, np.exp(epoch_loss), valid_elapsed))
	print("")
	model.train()

def main(args):
	config = utils.read_json_config(args.config_file_path)
	torch.manual_seed(config['training']['seed']) # Seed for reproducability
	use_cuda = check_cuda(config['training']['seed'])
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		# filename='log/%s' % (experiment_name),
		filemode='w')
	# define a new Handler to log to console as well
	console = logging.StreamHandler()
	# optional, set the logging level
	console.setLevel(logging.INFO)
	# set a format which is the same for console use
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	# tell the handler to use this format
	console.setFormatter(formatter)
	# add the handler to the root logger
	# logging.getLogger('').addHandler(console)
	# print 'Reading data ...'
	print(config)
	vocab = pkl.load(open(args.vocab_path,'rb'))[1]
	# print(vocab)
	vocab_size = len(vocab)
	# Server
	annoyIndex = AnnoyIndex(4096, metric='euclidean')
	annoyIndex.load(args.annoy_file_path)
	print(args.annoy_file_path)
	annoyPkl = pkl.load(open(args.annoy_pkl_path,'rb'))
	# # Local
	# annoyIndex = ""
	# annoyPkl = ""
	model_type = getattr(models, args.model_type)

	kb_vec = None
	use_kb = False
	celeb_vec = None
	kb_size = None
	celeb_vec_size = None
	kb_len = None
	celeb_len = None
	aspect_vec = None
	aspect_len = None
	aspect_size = None
	use_aspect = False
	use_review = False
	sentiment_vec = None
	sentiment_len = None
	sentiment_size = None
	use_sentiment = False

	if args.use_kb=='True':
		use_kb = True
		# celeb_data = pkl.load(open(args.train_celeb_path,'r'))
		kb_data = pkl.load(open(args.train_kb_path,'rb'))	
		print(len(kb_data[0]))	
		## TODO - copy in all
		kb_vocab = pkl.load(open(args.kb_vocab_path,'rb'))
		# celeb_vocab = pkl.load(open(args.celeb_vocab_path,'r'))
		kb_size = int(len(kb_vocab[0]))
		# print(kb_vocab)
		# print(kb_size)
		# celeb_vec_size = len(celeb_vocab[0])
		del kb_vocab

	if args.use_aspect=='True':
		use_aspect=True
		aspect_data = pkl.load(open(args.train_aspect_path,'rb'))
		# print(args.train_aspect_path)
		# print(aspect_data[0])
		# print(len(aspect_data[0]))
		aspect_size = vocab_size

	if args.use_sentiment=='True':
		use_sentiment=True
		sentiment_data = pkl.load(open(args.train_sentiment_path,'rb'))
		sentiment_size = vocab_size + 1
		print('sss',sentiment_size)

	if args.use_review == "True":
		use_review = True

	model = model_type(src_vocab_size=vocab_size, #config['model']['src_vocab_size'],
				tgt_vocab_size=vocab_size, #config['model']['tgt_vocab_size'],
				src_emb_dim=config['model']['src_emb_dim'],
				tgt_emb_dim=config['model']['tgt_emb_dim'],
				enc_hidden_size=config['model']['enc_hidden_size'],
				dec_hidden_size=config['model']['dec_hidden_size'],
				context_hidden_size=config['model']['context_hidden_size'],
				batch_size=config['data']['batch_size'],
				image_in_size=config['model']['image_in_size'],
				bidirectional_enc=config['model']['bidirectional_enc'],
				bidirectional_context=config['model']['bidirectional_context'],
				num_enc_layers=config['model']['num_enc_layers'],
				num_dec_layers=config['model']['num_dec_layers'],
				num_context_layers=config['model']['num_context_layers'],
				dropout_enc=config['model']['dropout_enc'],
				dropout_dec=config['model']['dropout_dec'],
				dropout_context=config['model']['dropout_context'],
				max_decode_len=config['model']['max_decode_len'],
				non_linearity=config['model']['non_linearity'],
				enc_type=config['model']['enc_type'],
				dec_type=config['model']['dec_type'],
				context_type=config['model']['context_type'],
				use_attention=config['model']['use_attention'],
				decode_function=config['model']['decode_function'],
				num_states=args.num_states,
				use_kb=use_kb, kb_size=kb_size, celeb_vec_size=celeb_vec_size,
				use_aspect=use_aspect, aspect_size=aspect_size,
				use_sentiment=use_sentiment, sentiment_size=sentiment_size, use_review = use_review
				)
	model = torch_utils.gpu_wrapper(model, use_cuda=use_cuda)
	# model = nn.DataParallel(model)
	# print_model(model)
	optimizer = optimizer_wrapper(model, config['training']['optimizer'], config['training']['lr'],
				config['training']['lr_decay'])
	#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	losses = []
	# Weight masking for cross entropy loss; can also use ignore_index
	# weight_mask = torch.ones(config['model']['tgt_vocab_size'])
	# weight_mask = torch_utils.gpu_wrapper(weight_mask, use_cuda=False)
	# weight_mask[3] = 0 # weight_mask[tgt_dic['word2id']['<pad>']] = 0
	loss_criterion = nn.CrossEntropyLoss(ignore_index=config['data']['pad_id']) 
	#loss_criterion = LabelSmoothingCrossEntropy()

	#weight=weight_mask) nn.CrossEntropyLoss
	loss_criterion = torch_utils.gpu_wrapper(loss_criterion, use_cuda=use_cuda)
	# Load all training data
	train_data = pkl.load(open(args.train_pkl_path,'rb'))
	# train_state_data = pkl.load(open(args.train_state_pkl_path,'r'))

	batch_size = config['data']['batch_size']
	total_samples = len(train_data)
	print(total_samples)
	num_train_batch = int(math.ceil(float(total_samples)/float(batch_size)))
	print('pp',num_train_batch)
	for epoch in range(config['training']['num_epochs']):
		total_loss = 0. 
		n_total_words = 0.
		epoch_start = time.time()
		for batch_id in range(num_train_batch):
			batch_start = time.time()
			# print('b',batch_id*batch_size,(batch_id+1)*batch_size)
			batch_data = train_data[batch_id*batch_size:(batch_id+1)*batch_size]
			if use_kb:
				kb_len = np.array(kb_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
				kb_len = utils.convert_states_to_torch(kb_len, use_cuda=use_cuda)
				kb_vec = np.array(kb_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
				kb_vec = utils.convert_states_to_torch(kb_vec, use_cuda=use_cuda)
				# Celebs
				# celeb_len = np.array(celeb_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
				# celeb_len = utils.convert_states_to_torch(celeb_len, use_cuda=use_cuda)
				# celeb_vec = np.array(celeb_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
				# celeb_vec = utils.convert_states_to_torch(celeb_vec, use_cuda=use_cuda)
				# print(kb_len)
				# # kb_array = np.array(kb_array)
				# # kb_len = np.array(kb_array[0])
				# kb_vec = np.array(kb_array[1])				
				# kb_vec = utils.convert_states_to_torch(kb_vec, use_cuda=use_cuda)
				# celeb_vec = celeb_data[batch_id*batch_size:(batch_id+1)*batch_size]
			# batch_state = train_state_data[batch_id*batch_size:(batch_id+1)*batch_size]
			if use_aspect:
				aspect_len = np.array(aspect_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
				aspect_len = utils.convert_states_to_torch(aspect_len, use_cuda=use_cuda)
				aspect_vec = np.array(aspect_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
				aspect_vec = utils.convert_states_to_torch(aspect_vec, use_cuda=use_cuda)

			if use_sentiment:
				# print(sentiment_data[0][0])
				# print(sentiment_data[1][0])
				sentiment_len = np.array(sentiment_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
				sentiment_len = utils.convert_states_to_torch(sentiment_len, use_cuda=use_cuda)
				sentiment_vec = np.array(sentiment_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
				sentiment_vec = utils.convert_states_to_torch(sentiment_vec, use_cuda=use_cuda)
			# print(aspect_vec.size())
			text_enc_input, text_enc_in_len, image_enc_input, dec_text_input,\
				dec_out_seq, dec_seq_length= utils.get_batch_mmd_data(batch_data, 
						config['data']['start_id'],
						config['data']['end_id'], config['data']['pad_id'],
						config['data']['image_rep_size'], annoyIndex, annoyPkl,\
						use_cuda=use_cuda)
			# print(text_enc_input)
			# Forward + Backward + Optimize
			# model.zero_grad() ??? zero grad model or optim?
			# https://discuss.pytorch.org/t/do-i-need-to-do-optimizer-zero-grad-when-using-adam-solver/3235
			optimizer.zero_grad()
			dec_output_prob = model(text_enc_input, image_enc_input, text_enc_in_len, 
							dec_text_input, dec_out_seq, context_size=args.context_size, 
							teacher_forcing_ratio=1, use_cuda=use_cuda, kb_vec=kb_vec, 
							celeb_vec=celeb_vec, kb_len=kb_len, celeb_len=celeb_len,
							aspect_vec=aspect_vec, aspect_len=aspect_len, 
							sentiment_vec=sentiment_vec, sentiment_len=sentiment_len)
			# loss = loss_criterion(dec_output_prob, dec_out_seq)
			
			# print(dec_output_prob.size())
			# print(dec_out_seq.size())

			loss = loss_criterion(dec_output_prob.contiguous().view(-1, vocab_size), #config['model']['tgt_vocab_size']),
				dec_out_seq.view(-1))

			# target_toks = dec_out_seq.ne(config['data']['pad_id']).long().sum().data[0]
			n_words = dec_seq_length.float().sum().data.item()
			n_total_words += n_words
			loss.backward()
			# Gradient clipping to avoid exploding gradients
			nn.utils.clip_grad_norm(model.parameters(), config['training']['clip_grad'])			
			optimizer.step()
			# exp_lr_scheduler.step()
			batch_elapsed = (time.time() - batch_start)/60
			batch_loss = loss.data.item()/n_words # @TODO
			if (batch_id+1) % config['training']['log_every'] == 0:
				print('Batch Loss: Epoch [%d], Batch [%d], Loss: %.6f, Perplexity: %5.5f, Batch Time:%5.4f'
					  %(epoch+1, batch_id+1, batch_loss, np.exp(batch_loss), batch_elapsed)) 
			total_loss += loss.data.item()
			losses.append(batch_loss)
			# break
		epoch_loss = total_loss / n_total_words
		
		epoch_elapsed = time.time() - epoch_start
		# if (epoch+1) % config['training']['log_every'] == 0:
		print('Epoch Loss: Epoch [%d], Loss: %.6f, Perplexity: %5.5f, Epoch Time:%5.4f'
			  %(epoch+1, epoch_loss, np.exp(epoch_loss), epoch_elapsed)) 
		if (epoch+1) % config['training']['evaluate_every'] == 0:
			print("\nEvaluation:")
			evaluate(args.valid_pkl_path, loss_criterion, model, config, vocab_size, vocab,\
					annoyIndex, annoyPkl, use_cuda=use_cuda, use_kb=use_kb, use_aspect=use_aspect, 
					use_sentiment=use_sentiment, valid_aspect_path=args.valid_aspect_path, valid_kb_path=args.valid_kb_path, 
					valid_sentiment_path= args.valid_sentiment_path, valid_celeb_path=args.valid_celeb_path)
		# Save the models
		if (epoch+1) % config['training']['save_every'] == 0:
			# Save and load only the model parameters(recommended).
			torch.save(model.state_dict(), os.path.join(args.model_path, 
									'model_params_%d.pkl' %(epoch+1)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-model_path', type=str, default='./models/',
						help='path for saving trained models')
	parser.add_argument('-vocab_path', type=str, default='./data/vocab.pkl',
						help='path for vocabulary wrapper')
	parser.add_argument('-config_file_path', help='path to json config', required=True)
	parser.add_argument('-train_pkl_path', type=str, help='data directory path')
	parser.add_argument('-valid_pkl_path', type=str, help='data directory path')
	parser.add_argument('-annoy_file_path', type=str, help='annoy path')
	parser.add_argument('-annoy_pkl_path', type=str, help='annoy pkl')
	parser.add_argument('-model_type', type=str, default='MultimodalHRED', help='model type')
	parser.add_argument('-context_size', type=int, default=2, help='model type')
	parser.add_argument('-train_state_pkl_path', type=str, help='model type')
	parser.add_argument('-valid_state_pkl_path', type=str, help='model type')
	parser.add_argument('-use_kb', type=str, default='False', help='whether to use kb')
	parser.add_argument('-use_aspect', type=str, default='False', help='whether to use aspect')
	parser.add_argument('-use_review', type=str, default='False', help='whether to use review')
	parser.add_argument('-use_sentiment', type=str, default='False', help='whether to use sentiment')
	parser.add_argument('-train_kb_path', type=str, help='model type')
	parser.add_argument('-valid_kb_path', type=str, help='model type')
	parser.add_argument('-train_aspect_path', type=str, help='model type')
	parser.add_argument('-valid_aspect_path', type=str, help='model type')
	parser.add_argument('-train_sentiment_path', type=str, help='model type')
	parser.add_argument('-valid_sentiment_path', type=str, help='model type')	
	parser.add_argument('-train_celeb_path', type=str, help='celeb path')
	parser.add_argument('-valid_celeb_path', type=str, help='celeb path')
	parser.add_argument('-celeb_vocab_path', type=str, help='celeb path')
	parser.add_argument('-kb_vocab_path', type=str, help='celeb path')
	parser.add_argument('-num_states', type=int, help='num states for multitasking')
	args = parser.parse_args()
	main(args)
