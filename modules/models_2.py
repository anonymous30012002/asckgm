import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .encoderRNN import EncoderRNN
from .image_encoder import ImageEncoder
from .bridge import BridgeLayer
from .contextRNN import ContextRNN
from .decoder import DecoderRNN
from . import torch_utils as torch_utils
from .torch_utils import to_var
from .kb_encoder import KbEncoder
from .enc_dec_tformer import encoder,decoder,MTrans_img_kb_Encoder,MTrans_img_Encoder,MTrans_kb_Encoder,context_encoder, \
MTrans_aspect_Encoder, MTrans_kb_aspect_Encoder, MTrans_img_aspect_Encoder, MTrans_img_kb_aspect_Encoder, Review_Transformer, \
MTrans_aspect_sentiment_Encoder, MTrans_img_aspect_sentiment_Encoder, MTrans_kb_aspect_sentiment_Encoder,MTrans_img_kb_aspect_sentiment_Encoder

torch.set_printoptions(edgeitems=10)

class MultimodalHRED(nn.Module):
	r""" HRED model
	Args:
	Inputs: 
	Outputs: 
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, src_emb_dim, tgt_emb_dim, 
				enc_hidden_size, dec_hidden_size, context_hidden_size, batch_size, 
				image_in_size, bidirectional_enc=True, bidirectional_context=False, 
				num_enc_layers=1, num_dec_layers=1, num_context_layers=1, 
				dropout_enc=0.4, dropout_dec=0.4, dropout_context=0.4, max_decode_len=40, 
				non_linearity='tanh', enc_type='GRU', dec_type='GRU', context_type='GRU', 
				use_attention=True, decode_function='softmax', sos_id=2, eos_id=3, 
				tie_embedding=True, activation_bridge='Tanh', num_states=None,
				use_kb=False, kb_size=None, celeb_vec_size=None,use_aspect=False,
				aspect_size=None):
		super(MultimodalHRED, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.src_emb_dim = src_emb_dim
		self.tgt_emb_dim = tgt_emb_dim
		self.batch_size = batch_size
		self.bidirectional_enc = bidirectional_enc
		self.bidirectional_context = bidirectional_context
		self.num_enc_layers = num_enc_layers
		self.num_dec_layers = num_dec_layers
		self.num_context_layers = num_context_layers
		self.dropout_enc = dropout_enc #dropout prob for encoder
		self.dropout_dec = dropout_dec #dropout prob for decoder
		self.dropout_context = dropout_context #dropout prob for context
		self.non_linearity = non_linearity # default nn.tanh(); nn.relu()
		self.enc_type = enc_type
		self.dec_type = dec_type
		self.context_type = context_type
		self.sos_id = sos_id # start token		
		self.eos_id = eos_id # end token
		self.decode_function = decode_function # @TODO: softmax or log softmax 
		self.max_decode_len = max_decode_len # max timesteps for decoder
		self.attention_size = dec_hidden_size # Same as enc/dec hidden size!!
		# self.context_hidden_size = context_hidden_size
		# self.enc_hidden_size = enc_hidden_size
		# All implementations have encoder hidden size halved
		self.num_directions = 2 if bidirectional_enc else 1
		self.enc_hidden_size = enc_hidden_size // self.num_directions
		self.num_directions = 2 if bidirectional_context else 1
		self.context_hidden_size = context_hidden_size // self.num_directions
		self.dec_hidden_size = dec_hidden_size
		self.use_attention = use_attention
		self.image_in_size = image_in_size
		self.image_out_size = self.dec_hidden_size # Project on same size as enc hidden

		## TODO - copy this to all
		self.use_kb = use_kb 
		self.kb_size = kb_size 
		self.celeb_vec_size = celeb_vec_size
		self.use_aspect = use_aspect 
		self.aspect_size = aspect_size
		# Equating to emb_size = tgt_emb_dim for now 
		# Default to hidden_size = dec_hidden_size for now. 
		self.kb_emb_size = self.tgt_emb_dim
		self.kb_hidden_size = self.dec_hidden_size
		self.aspect_emb_size = self.tgt_emb_dim
		self.aspect_hidden_size = self.dec_hidden_size
		if self.use_kb:
			self.kb_encoder = KbEncoder(self.kb_size, self.kb_emb_size, self.kb_hidden_size,
			                	rnn_type='GRU', num_layers=1, batch_first=True,
			                	dropout=0, bidirectional=False)
			# Same for kb and celebs for now.
			# self.celeb_encoder = KbEncoder(self.celeb_vec_size, self.kb_emb_size, self.kb_hidden_size,
			#                 	rnn_type='GRU', num_layers=1, batch_first=True,
			#                 	dropout=0, bidirectional=False)
		if self.use_aspect:
			self.aspect_encoder = KbEncoder(self.aspect_size, self.aspect_emb_size, self.aspect_hidden_size,
				                rnn_type='GRU', num_layers=1, batch_first=True,
			    	            dropout=0, bidirectional=False)
		# Initialize encoder
		self.encoder = EncoderRNN(self.src_vocab_size, self.src_emb_dim, self.enc_hidden_size, 
						self.enc_type, self.num_enc_layers, batch_first=True, dropout=self.dropout_enc, 
						bidirectional=self.bidirectional_enc)
		self.image_encoder = ImageEncoder(self.image_in_size, self.image_out_size)
		# Initialize bridge layer 
		self.activation_bridge = activation_bridge
		# self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size, self.activation_bridge)
		self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size)
		# Initialize context encoder
		self.context_input_size = self.image_out_size + enc_hidden_size # image+text
		self.context_encoder = ContextRNN(self.context_input_size, self.context_hidden_size, 
								self.context_type, self.num_context_layers, batch_first=True,
								dropout=self.dropout_context, bidirectional=self.bidirectional_context)
		# Initialize RNN decoder
		self.decoder = DecoderRNN(self.tgt_vocab_size, self.tgt_emb_dim, self.dec_hidden_size, 
						self.dec_type, self.num_dec_layers, self.max_decode_len,  
						self.dropout_dec, batch_first=True, use_attention=self.use_attention, 
						attn_size = self.attention_size, sos_id=self.sos_id, eos_id=self.eos_id,
						use_input_feed=True,
						use_kb=self.use_kb, kb_size=self.kb_hidden_size, use_aspect=self.use_aspect,
						aspect_size=self.aspect_hidden_size, celeb_vec_size=self.kb_hidden_size)						
		if tie_embedding:
			self.decoder.embedding = self.encoder.embedding
		# Initialize parameters
		self.init_params()

	def forward(self, text_enc_input, image_enc_input, text_enc_in_len=None, dec_text_input=None,
				dec_out_seq=None, context_size=2, teacher_forcing_ratio=1, decode=False, 
				use_cuda=False, beam_size=1, kb_vec=None, celeb_vec=None, kb_len=None, celeb_len=None,
				aspect_vec=None, aspect_len=None):
		# text_enc_input == (turn, batch, seq_len) ==> will project it to features through RNN
		# text_enc_in_len == (turn, batch) # np.array
		assert (text_enc_input.size(0)==context_size), "Context size not equal to first dimension"
		# Define variables to store outputs
		batch_size = text_enc_input.size(1)
		# https://github.com/pytorch/pytorch/issues/5552
		context_enc_input_in_place = Variable(torch.zeros(batch_size, context_size, \
							self.dec_hidden_size*2), requires_grad=True)
		# https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/6
		# https://discuss.pytorch.org/t/how-to-copy-a-variable-in-a-network-graph/1603/6
		context_enc_input = context_enc_input_in_place.clone()
		context_enc_input = torch_utils.gpu_wrapper(context_enc_input, use_cuda=use_cuda) # Port to cuda
		for turn in range(0,context_size):
			# pytorch multiple packedsequence input ordering
			text_input = text_enc_input[turn,:] #3D to 2D (batch, seq_len) == regular input
			# Pass through encoder: 
			# text_enc_in_len[turn,:] # 2D to 1D
			encoder_outputs, encoder_hidden = self.encoder(text_input, text_enc_in_len[turn])
			# Bridge layer to pass encoder outputs to context RNN # (layers*directions, batch, features)
			# [-1] => (B,D); select the last, unsqueeze to (B,1,D)
			text_outputs = self.bridge(encoder_hidden, 
								bidirectional_encoder=self.bidirectional_enc)[-1] # (B,dim)

			image_input = image_enc_input[turn,:] #4D to 3D (batch, seq_len = num_images =1, features=4096*5=in_size)
			image_outputs = self.image_encoder(image_input).squeeze(1)
			# image_outputs = image_outputs.contiguous() # Error in py2
			combined_enc_input = self.combine_enc_outputs(text_outputs, image_outputs, dim=1)
			context_enc_input[:,turn,:] = combined_enc_input # (batch, 1, features)
		# Context RNN	
		context_enc_outputs, context_enc_hidden = self.context_encoder(context_enc_input)
		context_projected_hidden = self.bridge(context_enc_hidden, 
								bidirectional_encoder=self.bidirectional_context)#.unsqueeze(0) 
								# (B,D) => (Layer,B,D)
		# TODO: copy here and in decode as well.
		kb_outputs = None
		celeb_outputs = None
		aspect_outputs = None
		if self.use_kb:
			_, kb_hidden = self.kb_encoder(kb_vec, kb_len)
			kb_outputs = self.bridge(kb_hidden, 
					bidirectional_encoder=False)[-1] # (B,dim)
			# _, celeb_hidden = self.celeb_encoder(celeb_vec, celeb_len)
			# celeb_outputs = self.bridge(celeb_hidden, 
			# 		bidirectional_encoder=False)[-1] # (B,dim)
		if self.use_aspect:
			_, aspect_hidden = self.aspect_encoder(aspect_vec, aspect_len)
			aspect_outputs = self.bridge(aspect_hidden, 
					bidirectional_encoder=False)[-1]
		# print('aspect_output',aspect_outputs.shape)
		# print('kb_output',kb_outputs.shape)
		if not decode:
			decoder_outputs = self.decoder(dec_text_input,
								init_h=context_projected_hidden,
								encoder_outputs = encoder_outputs,
								input_valid_length = text_enc_in_len[turn],
								context_enc_outputs = context_enc_outputs,
							    kb_vec = kb_outputs,
							    aspect_vec = aspect_outputs,
							    celeb_vec = celeb_outputs, 
								decode=decode)
			return decoder_outputs
		else:
			prediction = self.decoder(init_h=context_projected_hidden,
								encoder_outputs = encoder_outputs,
								input_valid_length = text_enc_in_len[turn],
								context_enc_outputs = context_enc_outputs,
							    kb_vec = kb_outputs,
							    aspect_vec = aspect_outputs,
							    celeb_vec = celeb_outputs, 
								decode=decode)
			return prediction

	def combine_enc_outputs(self, text_outputs, image_outputs, dim=2):
		"""Combine tensors across specified dimension. """
		encoded_both = torch.cat([image_outputs, text_outputs],dim)
		return encoded_both

	def softmax_prob(self, logits):
		"""Return probability distribution over words."""
		soft_probs = torch_utils.softmax_3d(logits)
		return soft_probs

	def init_params(self, initrange=0.1):
		for name, param in self.named_parameters():
			if param.requires_grad:
				param.data.uniform_(-initrange, initrange)

class HRED(nn.Module):
	r""" HRED model
	Args:
	Inputs: 
	Outputs: 
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, src_emb_dim, tgt_emb_dim, 
				enc_hidden_size, dec_hidden_size, context_hidden_size, batch_size, 
				image_in_size, bidirectional_enc=True, bidirectional_context=False, 
				num_enc_layers=1, num_dec_layers=1, num_context_layers=1, 
				dropout_enc=0.4, dropout_dec=0.4, dropout_context=0.4, max_decode_len=40, 
				non_linearity='tanh', enc_type='GRU', dec_type='GRU', context_type='GRU', 
				use_attention=True, decode_function='softmax', sos_id=2, eos_id=3, 
				tie_embedding=True, activation_bridge='Tanh', num_states=None,
				use_kb=False, kb_size=None, celeb_vec_size=None,use_aspect=False,
				aspect_size=None):
		super(HRED, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.src_emb_dim = src_emb_dim
		self.tgt_emb_dim = tgt_emb_dim
		self.batch_size = batch_size
		self.bidirectional_enc = bidirectional_enc
		self.bidirectional_context = bidirectional_context
		self.num_enc_layers = num_enc_layers
		self.num_dec_layers = num_dec_layers
		self.num_context_layers = num_context_layers
		self.dropout_enc = dropout_enc #dropout prob for encoder
		self.dropout_dec = dropout_dec #dropout prob for decoder
		self.dropout_context = dropout_context #dropout prob for context
		self.non_linearity = non_linearity # default nn.tanh(); nn.relu()
		self.enc_type = enc_type
		self.dec_type = dec_type
		self.context_type = context_type
		self.sos_id = sos_id # start token		
		self.eos_id = eos_id # end token
		self.decode_function = decode_function # @TODO: softmax or log softmax 
		self.max_decode_len = max_decode_len # max timesteps for decoder
		self.attention_size = dec_hidden_size # Same as enc/dec hidden size!!
		# self.context_hidden_size = context_hidden_size
		# self.enc_hidden_size = enc_hidden_size
		# All implementations have encoder hidden size halved
		self.num_directions = 2 if bidirectional_enc else 1
		self.enc_hidden_size = enc_hidden_size // self.num_directions
		self.num_directions = 2 if bidirectional_context else 1
		self.context_hidden_size = context_hidden_size // self.num_directions
		self.dec_hidden_size = dec_hidden_size
		self.use_attention = use_attention
		self.image_in_size = image_in_size
		self.image_out_size = self.dec_hidden_size # Project on same size as enc hidden

		self.use_kb = use_kb 
		self.kb_size = kb_size 
		self.celeb_vec_size = celeb_vec_size
		self.use_aspect = use_aspect 
		self.aspect_size = aspect_size
		# Equating to emb_size = tgt_emb_dim for now 
		# Default to hidden_size = dec_hidden_size for now. 
		self.kb_emb_size = self.tgt_emb_dim
		self.kb_hidden_size = self.dec_hidden_size
		self.aspect_emb_size = self.tgt_emb_dim
		self.aspect_hidden_size = self.dec_hidden_size
		if self.use_kb:
			self.kb_encoder = KbEncoder(self.kb_size, self.kb_emb_size, self.kb_hidden_size,
				                rnn_type='GRU', num_layers=1, batch_first=True,
			    	            dropout=0, bidirectional=False)
			# Same for kb and celebs for now.
			# self.celeb_encoder = KbEncoder(self.celeb_vec_size, self.kb_emb_size, self.kb_hidden_size,
			# 	                rnn_type='GRU', num_layers=1, batch_first=True,
			#     	            dropout=0, bidirectional=False)
		if self.use_aspect:
			self.aspect_encoder = KbEncoder(self.aspect_size, self.aspect_emb_size, self.aspect_hidden_size,
				                rnn_type='GRU', num_layers=1, batch_first=True,
			    	            dropout=0, bidirectional=False)


		# Initialize encoder
		self.encoder = EncoderRNN(self.src_vocab_size, self.src_emb_dim, self.enc_hidden_size, 
						self.enc_type, self.num_enc_layers, batch_first=True, dropout=self.dropout_enc, 
						bidirectional=self.bidirectional_enc)
		# self.image_encoder = ImageEncoder(self.image_in_size, self.image_out_size)
		# Initialize bridge layer 
		self.activation_bridge = activation_bridge
		# self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size, self.activation_bridge)
		self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size)
		# Initialize context encoder
		self.context_input_size = enc_hidden_size #self.image_out_size + enc_hidden_size # image+text
		self.context_encoder = ContextRNN(self.context_input_size, self.context_hidden_size, 
								self.context_type, self.num_context_layers, batch_first=True,
								dropout=self.dropout_context, bidirectional=self.bidirectional_context)
		# Initialize RNN decoder
		self.decoder = DecoderRNN(self.tgt_vocab_size, self.tgt_emb_dim, self.dec_hidden_size, 
						self.dec_type, self.num_dec_layers, self.max_decode_len,  
						self.dropout_dec, batch_first=True, use_attention=self.use_attention, 
						attn_size = self.attention_size, sos_id=self.sos_id, eos_id=self.eos_id,
						use_input_feed=True,
						use_kb=self.use_kb, kb_size=self.kb_hidden_size, use_aspect=self.use_aspect,
						aspect_size=self.aspect_hidden_size,celeb_vec_size=self.kb_hidden_size)						

		if tie_embedding:
			self.decoder.embedding = self.encoder.embedding
		# Initialize parameters
		self.init_params()

	def forward(self, text_enc_input, image_enc_input, text_enc_in_len=None, dec_text_input=None,
				dec_out_seq=None, context_size=2, teacher_forcing_ratio=1, decode=False, 
				use_cuda=False, beam_size=1, kb_vec=None, celeb_vec=None, kb_len=None, celeb_len=None,
				aspect_vec=None, aspect_len=None):
		# text_enc_input == (turn, batch, seq_len) ==> will project it to features through RNN
		# text_enc_in_len == (turn, batch) # np.array
		assert (text_enc_input.size(0)==context_size), "Context size not equal to first dimension"
		# Define variables to store outputs
		batch_size = text_enc_input.size(1)
		# https://github.com/pytorch/pytorch/issues/5552
		context_enc_input_in_place = Variable(torch.zeros(batch_size, context_size, \
							self.context_input_size), requires_grad=True)
		# https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/6
		# https://discuss.pytorch.org/t/how-to-copy-a-variable-in-a-network-graph/1603/6
		context_enc_input = context_enc_input_in_place.clone()
		context_enc_input = torch_utils.gpu_wrapper(context_enc_input, use_cuda=use_cuda) # Port to cuda
		for turn in range(0,context_size):
			# pytorch multiple packedsequence input ordering
			text_input = text_enc_input[turn,:] #3D to 2D (batch, seq_len) == regular input

			# Pass through encoder: 
			# text_enc_in_len[turn,:] # 2D to 1D
			encoder_outputs, encoder_hidden = self.encoder(text_input, text_enc_in_len[turn])
			# Bridge layer to pass encoder outputs to context RNN # (layers*directions, batch, features)
			text_outputs = self.bridge(encoder_hidden, bidirectional_encoder=self.bidirectional_enc)[-1] # (B,dim)
			# image_input = image_enc_input[turn,:] #4D to 3D (batch, seq_len = num_images, features)
			# image_outputs = self.image_encoder(image_input).squeeze(1)
			# image_outputs = image_outputs.contiguous() # Error in py2
			# combined_enc_input = self.combine_enc_outputs(text_outputs, image_outputs, dim=1)
			context_enc_input[:,turn,:] = text_outputs # (batch, 1, features)
		# Context RNN	
		context_enc_outputs, context_enc_hidden = self.context_encoder(context_enc_input)
		context_projected_hidden = self.bridge(context_enc_hidden, 
								bidirectional_encoder=self.bidirectional_context)#.unsqueeze(0) 
								# (B,D) => (Layer,B,D)
		kb_outputs = None
		celeb_outputs = None
		aspect_outputs = None
		if self.use_kb:
			_, kb_hidden = self.kb_encoder(kb_vec, kb_len)
			kb_outputs = self.bridge(kb_hidden, 
					bidirectional_encoder=False)[-1] # (B,dim)
			# _, celeb_hidden = self.celeb_encoder(celeb_vec, celeb_len)
			# celeb_outputs = self.bridge(celeb_hidden, 
			# 		bidirectional_encoder=False)[-1] # (B,dim)
		if self.use_aspect:
			_, aspect_hidden = self.aspect_encoder(aspect_vec, aspect_len)
			aspect_outputs = self.bridge(aspect_hidden, 
					bidirectional_encoder=False)[-1]
		# print('aspect_output',aspect_outputs.shape)
		# print('kb_output',kb_outputs.shape)
		if not decode:
			# print('dec',dec_text_input)
			decoder_outputs = self.decoder(dec_text_input,
										   init_h=context_projected_hidden,
										   encoder_outputs = encoder_outputs,
										   input_valid_length = text_enc_in_len[turn],
										   context_enc_outputs = context_enc_outputs,
										    kb_vec = kb_outputs,
										    aspect_vec = aspect_outputs,
										    celeb_vec = celeb_outputs, 
											decode=decode)
			return decoder_outputs
		else:
			prediction = self.decoder(init_h=context_projected_hidden,
								encoder_outputs = encoder_outputs,
								input_valid_length = text_enc_in_len[turn],
								context_enc_outputs = context_enc_outputs,
							    kb_vec = kb_outputs,
							    aspect_vec = aspect_outputs,
							    celeb_vec = celeb_outputs, 
								decode=decode)
			# prediction, final_score, length = self.decoder.beam_decode(beam_size=beam_size, 
			# 									init_h=context_projected_hidden,
			# 								   encoder_outputs = encoder_outputs,
			# 								   input_valid_length = text_enc_in_len[turn],
			# 								   context_enc_outputs = context_enc_outputs,
			# 								   kb_vec = kb_outputs,
			# 								   celeb_vec = celeb_outputs)
			return prediction

	def init_params(self, initrange=0.1):
		for name, param in self.named_parameters():
			if param.requires_grad:
				param.data.uniform_(-initrange, initrange)

class Transformer(nn.Module):
	r""" transformer model 
	Args:
	Inputs: 
	Outputs: 
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, src_emb_dim, tgt_emb_dim, 
				enc_hidden_size, dec_hidden_size, context_hidden_size, batch_size, 
				image_in_size, bidirectional_enc=False, bidirectional_context=False, 
				num_enc_layers=1, num_dec_layers=1, num_context_layers=	None, 
				dropout_enc=0.4, dropout_dec=0.4, dropout_context=0.4, max_decode_len=40, 
				non_linearity=None, enc_type=None, dec_type=None, context_type=None, 
				use_attention=False, decode_function=None, sos_id=2, eos_id=3, 
				tie_embedding=False, activation_bridge=None, num_states=None,
				use_kb=False, kb_size=None, celeb_vec_size=None, use_aspect=False,
				aspect_size=None, use_sentiment=None,
				sentiment_size=None, use_review = False):

		super(Transformer, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.tgt_emb_dim = tgt_emb_dim
		self.enc_pf_dim = 2048
		self.dec_pf_dim = 2048
		self.enc_heads=8
		self.dec_heads=8
		self.batch_size = batch_size
		self.num_enc_layers = num_enc_layers
		self.num_dec_layers = num_dec_layers
		print()
		self.kb_layers=1
		self.aspect_layers=1
		self.sentiment_layers=1
		self.use_kb = use_kb
		self.use_aspect = use_aspect
		self.use_review = use_review
		self.use_sentiment = use_sentiment
		self.dropout_enc = dropout_enc #dropout prob for encoder
		self.dropout_dec = dropout_dec #dropout prob for decoder
		self.non_linearity = non_linearity # default nn.tanh(); nn.relu()
		self.sos_id = sos_id # start token		
		self.eos_id = eos_id # end token
		self.max_decode_len = max_decode_len # max timesteps for decoder
		self.attention_size = dec_hidden_size # Same as enc/dec hidden size!!
		self.pad_ind=0
		# self.context_hidden_size = context_hidden_size
		# self.enc_hidden_size = enc_hidden_size
		# All implementations have encoder hidden size halved
		#self.num_directions = 2 if bidirectional_enc else 1
		self.enc_hidden_size = enc_hidden_size
		self.dec_hidden_size = dec_hidden_size

		# Initialize encoder
		self.encoder = encoder(self.src_vocab_size, self.enc_hidden_size, 
			self.num_enc_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)

		self.kb_size = kb_size
		self.aspect_size = aspect_size
		self.sentiment_size = sentiment_size
		# Equating to emb_size = tgt_emb_dim for now 
		# Default to hidden_size = dec_hidden_size for now. 
		self.kb_emb_size = self.tgt_emb_dim
		self.kb_hidden_size = self.dec_hidden_size

		self.aspect_emb_size = self.tgt_emb_dim
		self.aspect_hidden_size = self.dec_hidden_size

		self.sentiment_hidden_size = self.dec_hidden_size
		self.linear = nn.Linear(enc_hidden_size,256)
		self.linear_review = nn.Linear(self.tgt_vocab_size,self.enc_hidden_size)


		if self.use_kb:
			self.kb_encoder = encoder(self.kb_size, self.kb_hidden_size, 
				                    self.kb_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)

		if self.use_aspect:
			self.aspect_encoder = encoder(self.aspect_size, self.aspect_hidden_size, 
				                    self.aspect_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)

		# print(sentiment_size,'sentiment_size')
		if self.use_sentiment:
			self.sentiment_encoder = encoder(self.sentiment_size, self.sentiment_hidden_size, 
				                    self.sentiment_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)

		self.mtrans_kb_encoder=MTrans_kb_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                    self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.mtrans_aspect_sentiment_encoder = MTrans_aspect_sentiment_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                    self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.mtrans_aspect_encoder=MTrans_aspect_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                    self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.mtrans_kb_aspect_encoder=MTrans_kb_aspect_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                          self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.mtrans_kb_aspect_sentiment_encoder = MTrans_kb_aspect_sentiment_Encoder(self.src_vocab_size, self.enc_hidden_size, 
			                                            self.num_enc_layers, self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.context_encoder=context_encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                    self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.decoder = decoder(self.tgt_vocab_size, self.dec_hidden_size, 
			self.num_dec_layers, self.dec_pf_dim, self.dec_heads,self.max_decode_len,self.dropout_dec)

		if self.use_review:
			self.review_gen = Review_Transformer(self.src_vocab_size, self.tgt_vocab_size, self.enc_hidden_size, self.dec_hidden_size,
		                                     self.num_enc_layers, self.num_dec_layers, self.dropout_enc, self.dropout_dec, 
		                                     self.max_decode_len, self.sos_id, self.eos_id,self.pad_ind, use_review = True)

		self.init_params()

		# model.apply(initialize_weights);

	def forward(self, text_enc_input, image_enc_input, text_enc_in_len=None, dec_text_input=None,
				dec_out_seq=None, context_size=2, teacher_forcing_ratio=1, decode=False, 
				use_cuda=False, beam_size=1, kb_vec=None, celeb_vec=None, kb_len=None, celeb_len=None,
				aspect_vec=None, aspect_len=None, sentiment_vec=None, sentiment_len=None):
		# text_enc_input == (batch, context_len*src_len)
		# print('tt',text_enc_input[0])
		# print(text_enc_input[29])
		# print(src_mask[29])
		#src_mask = [batch size, 1, 1, src len]
		# print(src_mask.size())
		# print(src_mask)
		#src_mask>>[batch_size,1,1,src_len]
		his_bank=None
		his_mask=None
		# print(aspect_vec.shape)
		# aspect_vec == (batch, aspect_len)
		for turn in range(0,context_size):
			# pytorch multiple packedsequence input ordering
			text_input = text_enc_input[turn,:] #3D to 2D (batch, seq_len) == regular input
			src_mask = (text_input != self.pad_ind).unsqueeze(1).unsqueeze(2)
			#src_mask = [batch size, 1, 1, src len]
			# Pass through encoder: 
			# text_enc_in_len[turn,:] # 2D to 1D
			# print(self.use_aspect)
			if (self.use_kb and not self.use_aspect and not self.use_sentiment):
				kb_mask = (kb_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				kb_outputs=self.kb_encoder(kb_vec,kb_mask)
				emb,his_bank,his_mask = self.mtrans_kb_encoder(text_input,src_mask,his_bank,
				                                                            his_mask,kb_outputs,kb_mask)
				# print(emb.shape)
			elif (self.use_aspect and not self.use_kb and not self.use_sentiment):
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				# sentiment_mask = (sentiment_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				# sentiment_outputs=self.sentiment_encoder(sentiment_vec,sentiment_mask)
				# print(text_input)
				emb,his_bank,his_mask = self.mtrans_aspect_encoder(text_input,src_mask,his_bank,
					                                              his_mask,aspect_outputs,aspect_mask)
				# print(emb.shape)
			elif (self.use_aspect and not self.use_kb and self.use_sentiment):
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				sentiment_mask = (sentiment_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				# print(sentiment_mask)
				# print(sentiment_vec)
				sentiment_outputs=self.sentiment_encoder(sentiment_vec,sentiment_mask)
				# print(text_input)
				emb,his_bank,his_mask = self.mtrans_aspect_sentiment_encoder(text_input, src_mask,his_bank,
					                                              his_mask,aspect_outputs,aspect_mask, sentiment_outputs, sentiment_mask)

			elif (self.use_aspect and self.use_kb and not self.use_sentiment):
				# print(type(kb_vec))
				# print(type(aspect_vec))
				kb_mask = (kb_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				kb_outputs=self.kb_encoder(kb_vec,kb_mask)
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				emb,his_bank,his_mask = self.mtrans_kb_aspect_encoder(text_input,src_mask,his_bank,
				                                       aspect_outputs, aspect_mask ,his_mask,kb_outputs,kb_mask)
				# print(emb.shape)
			elif (self.use_aspect and self.use_kb and self.use_sentiment):
				# print(type(kb_vec))
				# print(type(aspect_vec))
				# print("hello")
				kb_mask = (kb_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				kb_outputs=self.kb_encoder(kb_vec,kb_mask)
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				sentiment_mask = (sentiment_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				sentiment_outputs=self.sentiment_encoder(sentiment_vec,sentiment_mask)
				emb,his_bank,his_mask = self.mtrans_kb_aspect_sentiment_encoder(text_input,src_mask,his_bank,
				                            aspect_outputs, aspect_mask ,his_mask,kb_outputs,kb_mask,sentiment_outputs, sentiment_mask)
			else:
				emb,his_bank,his_mask=self.context_encoder(text_input, src_mask, his_bank, his_mask)

			# print("aspect",aspect_outputs.size())
			# print("context",his_bank.size())
			# his_bank = his_bank.reshape(his_bank.shape[0], -1)
			# print("his bank", his_bank[:,:,-1].size())
			# print("his mask", his_mask.size())

			if self.use_review:
				# checkpoint = torch.load("/home1/deeksha/mmd_sentiment_aspect/Review_generator/review_gen_model_1_adam.pt")
				checkpoint = torch.load("/home1/deeksha/mmd_sentiment_aspect/review_gen_without_sentiment/model/review_gen_model_50_adam.pt")
				model_dict = self.review_gen.state_dict()
				# print(model_dict.keys(),'\n')
# 
				# print(checkpoint.keys())
				keys = ["encoder.tos_embed.weight", "encoder.pos_embed.weight", "decoder.tos_embedding.weight",
				        "decoder.pos_embedding.weight", "decoder.fc_out.weight", "decoder.fc_out.bias"]
				checkpoint = {k: checkpoint[k] for k,v in checkpoint.items() if k not in keys}
				pretrained_dict = {(k if k in checkpoint else k) : (checkpoint[k] if k in checkpoint else v)  for k, v in model_dict.items()}
				# print(pretrained_dict.keys())
				model_dict.update(pretrained_dict)
				self.review_gen.load_state_dict(pretrained_dict)
				his_bank_r = self.linear(his_bank)
				# print(his_bank_r.size())
				# print(his_mask.size())
				his_mask_r = his_mask
				review_kb = self.review_gen(his_bank_r, 0, his_mask_r, decode = True, review = True)
				# print("review", review_kb.size())
				review_kb = self.linear_review(review_kb)
				# print("review", review_kb.size())
				

		#print(src.shape)
		# print(src[29])
		#src>>[batch_size,src_len,hid_dim]fs
		if not decode:
			if self.use_review:
				# print('rr',review_kb)
				output, attention=self.decoder(his_bank, dec_text_input, his_mask, self.sos_id, review_kb, decode)
			else:
				output, attention=self.decoder(his_bank, dec_text_input, his_mask, self.sos_id, None,decode)
			#output>>[batch_size,trg_len,output_dim]
			#attention>>[batch_size,n_heads,trg_len,src_len]
			return output
		else:
			# print(src[0])
			# pred_trgs=self.decoder(src,0,src_mask,self.sos_id,decode)
			if self.use_review:
				pred_trgs=self.decoder(his_bank,0,his_mask, self.sos_id,review_kb,decode)
			else:
				pred_trgs=self.decoder(src,0,src_mask,self.sos_id,None,decode)
			
			pred_trgs = torch.LongTensor([pred_trgs]).cuda().unsqueeze(1)
			return pred_trgs


	def init_params(self, initrange=0.1):
		for name, param in self.named_parameters():
			if param.dim() > 1:
				nn.init.xavier_uniform_(param)


class MTransformer(nn.Module):
	r""" HRED model
	Args:
	Inputs: 
	Outputs: 
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, src_emb_dim, tgt_emb_dim, 
				enc_hidden_size, dec_hidden_size, context_hidden_size, batch_size, 
				image_in_size, bidirectional_enc=False, bidirectional_context=False, 
				num_enc_layers=1, num_dec_layers=1, num_context_layers=	None, 
				dropout_enc=0.4, dropout_dec=0.4, dropout_context=0.4, max_decode_len=40, 
				non_linearity=None, enc_type=None, dec_type=None, context_type=None, 
				use_attention=False, decode_function=None, sos_id=2, eos_id=3, 
				tie_embedding=False, activation_bridge=None, num_states=None,
				use_kb=False, kb_size=None, celeb_vec_size=None,use_aspect=False,
				aspect_size=None, use_sentiment=None,
				sentiment_size=None, use_review = False):
		super(MTransformer, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.tgt_emb_dim = tgt_emb_dim
		self.enc_pf_dim = 1024
		self.dec_pf_dim = 1024
		self.enc_heads=8
		self.dec_heads=8
		self.batch_size = batch_size
		self.num_enc_layers = num_enc_layers
		self.num_dec_layers = num_dec_layers
		self.kb_layers=1
		self.sentiment_layers=1
		self.aspect_layers=1
		self.review_kb_layers=1
		self.use_kb = use_kb
		self.use_aspect = use_aspect
		self.use_review = use_review
		self.use_sentiment = use_sentiment
		self.dropout_enc = dropout_enc #dropout prob for encoder
		self.dropout_dec = dropout_dec #dropout prob for decoder
		self.non_linearity = non_linearity # default nn.tanh(); nn.relu()
		self.sos_id = sos_id # start token		
		self.eos_id = eos_id # end token
		self.max_decode_len = max_decode_len # max timesteps for decoder
		self.attention_size = dec_hidden_size # Same as enc/dec hidden size!!
		self.pad_ind=0
		# self.context_hidden_size = context_hidden_size
		# self.enc_hidden_size = enc_hidden_size
		# All implementations have encoder hidden size halved
		#self.num_directions = 2 if bidirectional_enc else 1
		self.enc_hidden_size = enc_hidden_size
		self.dec_hidden_size = dec_hidden_size

		self.image_in_size = image_in_size
		self.image_out_size = self.dec_hidden_size # Project on same size as enc hidden

		self.image_encoder = ImageEncoder(self.image_in_size, self.image_out_size)
		self.kb_size = kb_size
		self.aspect_size = aspect_size
		self.sentiment_size = sentiment_size
		# Equating to emb_size = tgt_emb_dim for now 
		# Default to hidden_size = dec_hidden_size for now. 
		self.kb_emb_size = self.tgt_emb_dim
		self.kb_hidden_size = self.dec_hidden_size

		self.aspect_emb_size = self.tgt_emb_dim
		self.aspect_hidden_size = self.dec_hidden_size

		self.sentiment_hidden_size = self.dec_hidden_size
		self.linear = nn.Linear(enc_hidden_size,256)
		self.linear_review = nn.Linear(self.tgt_vocab_size,self.enc_hidden_size)
		self.review_kb_size =  self.enc_hidden_size

		self.review_kb_hidden_size = self.dec_hidden_size

		if self.use_kb:
			self.kb_encoder = encoder(self.kb_size, self.kb_hidden_size, 
				                      self.kb_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)
		if self.use_aspect:
			self.aspect_encoder = encoder(self.aspect_size, self.aspect_hidden_size, 
				                    self.aspect_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)

		if self.use_sentiment:
			self.sentiment_encoder = encoder(self.sentiment_size, self.sentiment_hidden_size, 
				                    self.sentiment_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)

		self.review_kb_encoder = encoder(self.review_kb_size, self.review_kb_hidden_size, 
				                    self.review_kb_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)

		self.mtrans_img_kb_encoder=MTrans_img_kb_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                          self.enc_heads, self.enc_pf_dim, self.dropout_enc)
		
		self.mtrans_img_encoder=MTrans_img_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                          self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.mtrans_img_aspect_encoder=MTrans_img_aspect_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                          self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.mtrans_img_kb_aspect_encoder=MTrans_img_kb_aspect_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                          self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.mtrans_img_aspect_sentiment_encoder= MTrans_img_aspect_sentiment_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                          self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.mtrans_img_kb_aspect_sentiment_encoder = MTrans_img_kb_aspect_sentiment_Encoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                          self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.decoder = decoder(self.tgt_vocab_size, self.dec_hidden_size, 
			self.num_dec_layers, self.dec_pf_dim, self.dec_heads,self.max_decode_len,self.dropout_dec)

		if self.use_review:
			self.review_gen = Review_Transformer(self.src_vocab_size, self.tgt_vocab_size, self.enc_hidden_size, self.dec_hidden_size,
		                                     self.num_enc_layers, self.num_dec_layers, self.dropout_enc, self.dropout_dec, 
		                                     self.max_decode_len, self.sos_id, self.eos_id,self.pad_ind, use_review = True)
			self.checkpoint = torch.load("/home1/deeksha/mmd_sentiment_aspect/review_gen_sentiment/review_gen_sentiment/model/review_gen_adam_30.pt")
			self.model_dict = self.review_gen.state_dict()
			keys = ["encoder.tos_embed.weight", "encoder.pos_embed.weight", "decoder.tos_embedding.weight",
				    "decoder.pos_embedding.weight", "decoder.fc_out.weight", "decoder.fc_out.bias"]
			self.checkpoint = {k: self.checkpoint[k] for k,v in self.checkpoint.items() if k not in keys}
			self.pretrained_dict = {(k if k in self.checkpoint else k) : (self.checkpoint[k] if k in self.checkpoint else v)  for k, v in self.model_dict.items()}
			self.model_dict.update(self.pretrained_dict)
		# Initialize parameters
		self.init_params()

	def forward(self, text_enc_input, image_enc_input, text_enc_in_len=None, dec_text_input=None,
				dec_out_seq=None, context_size=2, teacher_forcing_ratio=1, decode=False, 
				use_cuda=False, beam_size=1, kb_vec=None, celeb_vec=None, kb_len=None, celeb_len=None,
				aspect_vec=None, aspect_len=None, sentiment_vec=None, sentiment_len=None):
		# text_enc_input == (turn, batch, seq_len) ==> will project it to features through RNN
		# text_enc_in_len == (turn, batch) # np.array

		# Define variables to store outputs
		batch_size = text_enc_input.size(1)
		# https://github.com/pytorch/pytorch/issues/5552
		his_bank=None
		his_mask=None
		img_mask=None
		for turn in range(0,context_size):
			# pytorch multiple packedsequence input ordering
			text_input = text_enc_input[turn,:] #3D to 2D (batch, seq_len) == regular input
			src_mask = (text_input != self.pad_ind).unsqueeze(1).unsqueeze(2)
			#src_mask = [batch size, 1, 1, src len]
			# Pass through encoder: 
			# text_enc_in_len[turn,:] # 2D to 1D
			image_input = image_enc_input[turn,:] #4D to 3D (batch, seq_len = num_images =1, features=4096*5=in_size)
			image_outputs = self.image_encoder(image_input)
			kb_outputs = None
			if (self.use_kb and not self.use_aspect and not self.use_sentiment and not self.use_review):
				kb_mask = (kb_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				kb_outputs=self.kb_encoder(kb_vec,kb_mask)
				emb,his_bank,his_mask = self.mtrans_img_kb_encoder(text_input,src_mask,his_bank,
				                                       image_outputs, img_mask ,his_mask,kb_outputs,kb_mask)
				# print(emb.size())
			elif (self.use_aspect and not self.use_kb and not self.use_sentiment and not self.use_review):
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				emb,his_bank,his_mask = self.mtrans_img_aspect_encoder(text_input,src_mask,his_bank,
				                                       image_outputs, img_mask ,his_mask,aspect_outputs,aspect_mask)
				# print(emb.size())

			elif (self.use_aspect and self.use_kb and self.use_sentiment and not self.use_review):
				kb_mask = (kb_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				kb_outputs=self.kb_encoder(kb_vec,kb_mask)
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				sentiment_mask = (sentiment_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				sentiment_outputs=self.sentiment_encoder(sentiment_vec,sentiment_mask)
				# print("input size",self.src_vocab_size)
				emb,his_bank,his_mask = self.mtrans_img_kb_aspect_sentiment_encoder(text_input,src_mask,his_bank,
				                                       image_outputs, img_mask ,his_mask,kb_outputs,kb_mask,
				                                        aspect_outputs, aspect_mask,sentiment_outputs,sentiment_mask)
				# print(emb.size())

			elif (self.use_aspect and not self.use_kb and self.use_sentiment and self.use_review):
				# print("hello")
				self.review_gen.load_state_dict(self.pretrained_dict)
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				sentiment_mask = (sentiment_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				sentiment_outputs=self.sentiment_encoder(sentiment_vec,sentiment_mask)
				# print("input size",self.src_vocab_size)
				emb,his_bank,his_mask = self.mtrans_img_aspect_sentiment_encoder(text_input,src_mask,his_bank,image_outputs, 
					                       img_mask ,his_mask,aspect_outputs,aspect_mask, sentiment_outputs, sentiment_mask)
				# loading review kb
				his_bank_r = self.linear(his_bank)
				his_mask_r = his_mask
				review_kb = self.review_gen(his_bank_r, 0, his_mask_r, decode = True, review = True)
				review_kb = self.linear_review(review_kb)
				# print("review_kb size",review_kb.size())
				# print("hid dim",self.review_kb_hidden_size)
				emb,his_bank,his_mask = self.mtrans_img_kb_aspect_sentiment_encoder(text_input,src_mask,his_bank,
				                                       image_outputs, img_mask ,his_mask,review_kb,None,
				                                        aspect_outputs, aspect_mask,sentiment_outputs,sentiment_mask)
				
				# print("encoder output", his_bank.size())

			elif (self.use_aspect and not self.use_kb and self.use_sentiment and not self.use_review):
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				sentiment_mask = (sentiment_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				sentiment_outputs=self.sentiment_encoder(sentiment_vec,sentiment_mask)
				emb,his_bank,his_mask = self.mtrans_img_aspect_sentiment_encoder(text_input,src_mask,his_bank,image_outputs, 
					                       img_mask ,his_mask,aspect_outputs,aspect_mask, sentiment_outputs, sentiment_mask)

			elif (self.use_aspect and self.use_kb and not self.use_sentiment and not self.use_review):
				kb_mask = (kb_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				kb_outputs=self.kb_encoder(kb_vec,kb_mask)
				aspect_mask = (aspect_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
				aspect_outputs=self.aspect_encoder(aspect_vec,aspect_mask)
				emb,his_bank,his_mask = self.mtrans_img_kb_aspect_encoder(text_input,src_mask,his_bank,
				                                       image_outputs, img_mask ,his_mask,kb_outputs,kb_mask,
				                                                                aspect_outputs, aspect_mask)
			else:
				emb,his_bank,his_mask = self.mtrans_img_encoder(text_input,src_mask,his_bank,
				                                              image_outputs, img_mask ,his_mask)
				# print(emb.size())
			'''
			if self.use_review:
				# checkpoint = torch.load("/home1/deeksha/mmd_sentiment_aspect/Review_generator/review_gen_model_1_adam.pt")
				checkpoint = torch.load("/home1/deeksha/mmd_sentiment_aspect/review_gen_without_sentiment/model/review_gen_model_50_adam.pt")
				model_dict = self.review_gen.state_dict()
				# print(model_dict.keys(),'\n')
# 
				# print(checkpoint.keys())
				keys = ["encoder.tos_embed.weight", "encoder.pos_embed.weight", "decoder.tos_embedding.weight",
				        "decoder.pos_embedding.weight", "decoder.fc_out.weight", "decoder.fc_out.bias"]
				checkpoint = {k: checkpoint[k] for k,v in checkpoint.items() if k not in keys}
				pretrained_dict = {(k if k in checkpoint else k) : (checkpoint[k] if k in checkpoint else v)  for k, v in model_dict.items()}
				# print(pretrained_dict.keys())
				model_dict.update(pretrained_dict)
				self.review_gen.load_state_dict(pretrained_dict)
				his_bank_r = self.linear(his_bank)
				# print(his_bank_r.size())
				# print(his_mask.size())
				his_mask_r = his_mask
				review_kb = self.review_gen(his_bank_r, 0, his_mask_r, decode = True, review = True)
				# print("review", review_kb.size())
				review_kb = self.linear_review(review_kb)
			'''

		if not decode:
			if self.use_review:
				output, attention=self.decoder(his_bank, dec_text_input, his_mask, self.sos_id, review_kb, decode)
			else:
				if kb_outputs is not None:
					output, attention=self.decoder(his_bank, dec_text_input, his_mask, self.sos_id,kb_outputs, decode)
				else:
					output, attention=self.decoder(his_bank, dec_text_input, his_mask, self.sos_id,None, decode)
			#output>>[batch_size,trg_len,output_dim]
			#attention>>[batch_size,n_heads,trg_len,src_len]
			return output
		else:
			# print(src[0])
			if self.use_review:
				pred_trgs=self.decoder(his_bank,0,his_mask,self.sos_id,review_kb,decode)
			else:
				if kb_outputs is not None:
					pred_trgs=self.decoder(his_bank,0,his_mask,self.sos_id,kb_outputs,decode)
				else:
					pred_trgs=self.decoder(his_bank,0,his_mask,self.sos_id,None,decode)
			pred_trgs = torch.LongTensor([pred_trgs]).cuda().unsqueeze(1)
			return pred_trgs

	def init_params(self, initrange=0.1):
		for name, param in self.named_parameters():
			if param.dim() > 1:
				nn.init.xavier_uniform_(param)

'''
class MKbTransformer(nn.Module):
	r""" HRED model
	Args:
	Inputs: 
	Outputs: 
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, src_emb_dim, tgt_emb_dim, 
				enc_hidden_size, dec_hidden_size, context_hidden_size, batch_size, 
				image_in_size, bidirectional_enc=False, bidirectional_context=False, 
				num_enc_layers=1, num_dec_layers=1, num_context_layers=	None, 
				dropout_enc=0.4, dropout_dec=0.4, dropout_context=0.4, max_decode_len=40, 
				non_linearity=None, enc_type=None, dec_type=None, context_type=None, 
				use_attention=False, decode_function=None, sos_id=2, eos_id=3, 
				tie_embedding=False, activation_bridge=None, num_states=None,
				use_kb=False, kb_size=None, celeb_vec_size=None):
		super(MKbTransformer, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.tgt_emb_dim = tgt_emb_dim
		self.enc_pf_dim = 1024
		self.dec_pf_dim = 1024
		self.enc_heads=8
		self.dec_heads=8
		self.batch_size = batch_size
		self.num_enc_layers = num_enc_layers
		self.num_dec_layers = num_dec_layers
		self.kb_layers=1
		self.use_kb = use_kb 
		self.dropout_enc = dropout_enc #dropout prob for encoder
		self.dropout_dec = dropout_dec #dropout prob for decoder
		self.non_linearity = non_linearity # default nn.tanh(); nn.relu()
		self.sos_id = sos_id # start token		
		self.eos_id = eos_id # end token
		self.max_decode_len = max_decode_len # max timesteps for decoder
		self.attention_size = dec_hidden_size # Same as enc/dec hidden size!!
		self.pad_ind=0
		# self.context_hidden_size = context_hidden_size
		# self.enc_hidden_size = enc_hidden_size
		# All implementations have encoder hidden size halved
		#self.num_directions = 2 if bidirectional_enc else 1
		self.enc_hidden_size = enc_hidden_size
		self.dec_hidden_size = dec_hidden_size

		self.image_in_size = image_in_size
		self.image_out_size = self.dec_hidden_size # Project on same size as enc hidden

		self.use_kb = use_kb 
		self.kb_size = kb_size 
		# Equating to emb_size = tgt_emb_dim for now 
		# Default to hidden_size = dec_hidden_size for now. 
		self.kb_emb_size = self.tgt_emb_dim
		self.kb_hidden_size = self.dec_hidden_size
		if self.use_kb:
			self.kb_encoder = encoder(self.kb_size, self.kb_hidden_size, 
			self.kb_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)

		#encoder(self.src_vocab_size, self.enc_hidden_size, 
		#	self.num_enc_layers=1, self.enc_heads, self.enc_pf_dim,self.dropout_enc)
		self.image_encoder = ImageEncoder(self.image_in_size, self.image_out_size)

		self.mtrans_encoder=MTransEncoder(self.src_vocab_size, self.enc_hidden_size, self.num_enc_layers,
			                                          self.enc_heads, self.enc_pf_dim, self.dropout_enc)

		self.decoder = decoder(self.tgt_vocab_size, self.dec_hidden_size, 
			self.num_dec_layers, self.dec_pf_dim, self.dec_heads,self.max_decode_len,self.dropout_dec)

		# Initialize parameters
		self.init_params()

	def forward(self, text_enc_input, image_enc_input, text_enc_in_len=None, dec_text_input=None,
				dec_out_seq=None, context_size=2, teacher_forcing_ratio=1, decode=False, 
				use_cuda=False, beam_size=1, kb_vec=None, celeb_vec=None, kb_len=None, celeb_len=None):
		# text_enc_input == (turn, batch, seq_len) ==> will project it to features through RNN
		# text_enc_in_len == (turn, batch) # np.array

		# Define variables to store outputs
		# print('hid_size',self.kb_hidden_size)
		# print('emb_size',self.kb_emb_size)
		batch_size = text_enc_input.size(1)
		# https://github.com/pytorch/pytorch/issues/5552
		his_bank=None
		his_mask=None
		img_mask=None
		kb_outputs = None
		for turn in range(0,context_size):
			# pytorch multiple packedsequence input ordering

			text_input = text_enc_input[turn,:] #3D to 2D (batch, seq_len) == regular input
			src_mask = (text_input != self.pad_ind).unsqueeze(1).unsqueeze(2)
			kb_mask = (kb_vec != self.pad_ind).unsqueeze(1).unsqueeze(2)
			#src_mask = [batch size, 1, 1, src len]
			# Pass through encoder: 
			# text_enc_in_len[turn,:] # 2D to 1D
			image_input = image_enc_input[turn,:] #4D to 3D (batch, seq_len = num_images =1, features=4096*5=in_size)
			image_outputs = self.image_encoder(image_input)
			
			if self.use_kb:
				kb_outputs = self.kb_encoder(kb_vec,kb_mask) 
			emb,his_bank,his_mask = self.mtrans_encoder(text_input,src_mask,his_bank,
				                                       image_outputs, img_mask ,his_mask,kb_outputs,kb_mask)

		if not decode:

			output, attention=self.decoder(his_bank, dec_text_input, his_mask, self.sos_id, decode)
			#output>>[batch_size,trg_len,output_dim]
			#attention>>[batch_size,n_heads,trg_len,src_len]
			return output
		else:
			# print(src[0])
			pred_trgs=self.decoder(his_bank,0,his_mask,self.sos_id,decode)
			pred_trgs = torch.LongTensor([pred_trgs]).cuda().unsqueeze(1)
			return pred_trgs

	def init_params(self, initrange=0.1):
		for name, param in self.named_parameters():
			if param.dim() > 1:
				nn.init.xavier_uniform_(param)
'''