import torch
import torch.nn as nn
import torch.nn.functional as F
from .layer_tformer import encoder_layer, decoder_layer,MTrans_img_kb_encoder_layer,MTrans_img_encoder_layer,MTrans_kb_encoder_layer,\
context_encoder_layer, MTrans_aspect_encoder_layer, MTrans_kb_aspect_encoder_layer, MTrans_img_aspect_encoder_layer, \
MTrans_img_kb_aspect_encoder_layer, review_encoder_layer, review_decoder_layer, MTrans_aspect_sentiment_encoder_layer, \
MTrans_img_aspect_sentiment_encoder_layer,MTrans_kb_aspect_sentiment_encoder_layer,MTrans_img_kb_aspect_sentiment_encoder_layer

class encoder(nn.Module):
	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):
		super().__init__()

		self.tos_embed=nn.Embedding(input_dim,hid_dim)
		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.encoder_layer=nn.ModuleList([encoder_layer(hid_dim,pf_dim,n_heads,drop_prob) for i in range(n_layers)])
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self,src,src_mask):
		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		##src>>[batch_size,src_len,hid_dim]
		for layer in self.encoder_layer:
			src=layer(src,src_mask)

		return src


class context_encoder(nn.Module):
	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):
		super(context_encoder,self).__init__()

		self.tos_embed=nn.Embedding(input_dim,hid_dim)
		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.context_encoder_layer=nn.ModuleList([context_encoder_layer(hid_dim,pf_dim,n_heads,drop_prob) for i in range(n_layers)])
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		

	def forward(self,src,src_mask,history_bank, his_mask):
		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		##src>>[batch_size,src_len,hid_dim]
		for layer in self.context_encoder_layer:
			src=layer(src,src_mask,history_bank , his_mask)
		out = self.layer_norm(src)

		return token_embedding,out.contiguous(),src_mask



class decoder(nn.Module):
	def __init__(self,output_dim,hid_dim,n_layers,pf_dim,n_heads,max_unroll=40,drop_prob=0.1):
		super().__init__()
		self.tos_embedding=nn.Embedding(output_dim,hid_dim)
		self.pos_embedding=nn.Embedding(output_dim,hid_dim)

		self.layers=nn.ModuleList([decoder_layer(hid_dim,n_heads,pf_dim,drop_prob) for _ in range(n_layers)])

		self.fc_out=nn.Linear(hid_dim,output_dim)
		self.sample = False

		# self.fc_out = nn.Linear(hid_dim, output_dim)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()
		self.max_unroll=max_unroll
		self.pad_ind=0

	def init_token(self, batch_size, sos_id=8):
		"""Get Variable of <SOS> Index (batch_size)"""
		# x = torch.LongTensor([sos_id] * batch_size).cuda().unsqueeze(1)
		x =([sos_id] * batch_size)
		return x

	# def trg_mask(self,x):
	# 	tgt_words = x.unsqueeze(2)[:, :, 0]
	# 	tgt_batch, tgt_len = tgt_words.size()
	# 	trg_pad_mask = tgt_words.data.eq(self.pad_ind).unsqueeze(1)
	# 	trg_pad_mask=trg_pad_mask.unsqueeze(2)
	# 	#trg_pad_mask>[batch_size,1,1,trg_len]
	# 	trg_len=x.shape[1]
	# 	trg_sub_mask=torch.tril(torch.ones((trg_len,trg_len)))
	# 	#trg_sub_mask>>[trg_len,trg_len]

	# 	trg_pad_mask=torch.gt(trg_pad_mask,0).cuda()
	# 	trg_sub_mask=torch.gt(trg_sub_mask,0).cuda()

	# 	trg_mask=trg_pad_mask & trg_sub_mask
	# 	#print('trg_mask',trg_mask)
	# 	trg_mask=~trg_mask
		
	# 	#print('reverse_trg_mask',trg_mask)
	# 	#trg_mask>[batch_size,1,trg_len,trg_len]
	# 	return trg_mask

	def trg_mask(self,trg):
		#trg>>[batch_size,trg_len]
		trg_pad_mask=(trg!=self.pad_ind).unsqueeze(1).unsqueeze(2)
		#trg_pad_mask>[batch_size,1,1,trg_len]
		trg_len=trg.shape[1]
		trg_sub_mask=torch.tril(torch.ones((trg_len,trg_len)).cuda()).bool()#.type(torch.uint8)
		#trg_sub_mask>>[trg_len,trg_len]
		trg_mask=trg_pad_mask & trg_sub_mask
		#trg_mask>[batch_size,1,trg_len,trg_len]
		return trg_mask

	def decode(self, out):
		"""
		Args:
			out: unnormalized word distribution [batch_size, vocab_size]
		Return:
			x: word_index [batch_size]
		"""

		# Sample next word from multinomial word distribution
		
		if self.sample:
			# x: [batch_size] - word index (next input)
			x = torch.multinomial(self.softmax(out / self.temperature), 2).view(-1)
		# Greedy sampling
		else:
			# x: [batch_size] - word index (next input)
			_, x = out.max(dim=2)
		return x[:,-1].item()
		
		#return out.argmax(2)[:,-1].unsqueeze(1).unsqueeze(2)

	def forward_step(self, enc_src, trg, src_mask, review_kb):
		trg_mask=self.trg_mask(trg)
		# print(trg_mask[3])
		# print(trg_mask.size())
		# print(trg_mask[0])
		batch_size=trg.shape[0]
		trg_len=trg.shape[1]

		pos=torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embed=self.pos_embedding(pos)
		tos_embed=self.tos_embedding(trg).cuda()
		trg=self.dropout(tos_embed*self.scale+pos_embed)
		##trg>>[batch_size,trg_len,hid_dim]
		
		for layer in self.layers:
			output,attention=layer(enc_src, trg, src_mask, trg_mask, review_kb)
		output=self.fc_out(output)

		#output>>[batch_size,trg_len,output_dim]
		return output,attention

	def forward(self, enc_src, trg, src_mask, sos_id, review_kb=None, decode=False):
		##enc_src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		##trg_mask>>[batch_size,1,trg_len,trg_len]
		##trg>>[batch_size,trg_len]

		batch_size=enc_src.shape[0]
		trg_indexes = self.init_token(batch_size,sos_id)
		# print(trg[0])
		# print(trg_indexes.size())
		#x>[batch_size]
		#x=self.init_token(batch_size,sos_id)
		#x=x.unsqueeze(1).cuda()
		# print('rrrr',review_kb.size())
	
		if not decode:
			output, attention = self.forward_step(enc_src, trg, src_mask, review_kb)
			return output, attention
		else:
			# trg_indexes=[]
			
			for i in range(self.max_unroll):
				x = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()

				with torch.no_grad():
					output, attention = self.forward_step(enc_src, x, src_mask, review_kb)

				# print(output[0][0])
				pred_token = self.decode(output)
				trg_indexes.append(pred_token)

				if pred_token == 3:
					break
			return trg_indexes[1:]

class MTrans_kb_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_kb_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_kb_encoder_layer = nn.ModuleList(
			[MTrans_kb_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , his_mask, kb_bank,kb_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_kb_encoder_layer[i](src, src_mask, history_bank, his_mask, kb_bank, kb_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_aspect_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_aspect_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_aspect_encoder_layer = nn.ModuleList(
			[MTrans_aspect_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , his_mask, aspect_bank, aspect_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_aspect_encoder_layer[i](src, src_mask, history_bank, his_mask, aspect_bank,
				                                          aspect_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_aspect_sentiment_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_aspect_sentiment_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_aspect_sentiment_encoder_layer = nn.ModuleList(
			[ MTrans_aspect_sentiment_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , his_mask, aspect_bank, aspect_mask, sentiment_outputs, sentiment_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		# print(src)
		# print(src_mask)
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_aspect_sentiment_encoder_layer[i](src, src_mask, history_bank, his_mask, aspect_bank,
				                                          aspect_mask, sentiment_outputs, sentiment_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_kb_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_kb_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_kb_encoder_layer = nn.ModuleList(
			[MTrans_img_kb_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask, kb_bank,kb_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_kb_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask, kb_bank, kb_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_kb_aspect_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_kb_aspect_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_kb_aspect_encoder_layer = nn.ModuleList(
			[MTrans_kb_aspect_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , aspect_bank, aspect_mask, his_mask, kb_bank,kb_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_kb_aspect_encoder_layer[i](src, src_mask, aspect_bank, aspect_mask, history_bank, his_mask, kb_bank, kb_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_kb_aspect_sentiment_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_kb_aspect_sentiment_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_kb_aspect_sentiment_encoder_layer = nn.ModuleList(
			[MTrans_kb_aspect_sentiment_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , aspect_bank, aspect_mask, his_mask, kb_bank,kb_mask,sentiment_bank, sentiment_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		# print(src.size(),aspect_bank.size(),kb_bank.size(),sentiment_bank.size())
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_kb_aspect_sentiment_encoder_layer[i](src, src_mask, aspect_bank, aspect_mask, history_bank, 
				                                         his_mask, kb_bank, kb_mask,sentiment_bank, sentiment_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_encoder_layer = nn.ModuleList(
			[MTrans_img_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_aspect_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_aspect_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_aspect_encoder_layer = nn.ModuleList(
			[MTrans_img_aspect_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask, aspect_bank, aspect_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_aspect_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask, 
				                                                                           aspect_bank, aspect_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_aspect_sentiment_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_aspect_sentiment_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_aspect_sentiment_encoder_layer = nn.ModuleList(
			[MTrans_img_aspect_sentiment_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask, aspect_bank, aspect_mask, sentiment_outputs, sentiment_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_aspect_sentiment_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask, 
				                                                          aspect_bank, aspect_mask, sentiment_outputs, sentiment_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_kb_aspect_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_kb_aspect_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_kb_aspect_encoder_layer = nn.ModuleList(
			[MTrans_img_kb_aspect_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask, kb_bank,kb_mask,aspect_bank, aspect_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_kb_aspect_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask, kb_bank, kb_mask,
				                                                                                        aspect_bank, aspect_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class MTrans_img_kb_aspect_sentiment_Encoder(nn.Module):

	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):

		super(MTrans_img_kb_aspect_sentiment_Encoder, self).__init__()
		self.n_layers=n_layers
		self.tos_embed=nn.Embedding(input_dim,hid_dim)

		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.mtrans_img_kb_aspect_sentiment_encoder_layer = nn.ModuleList(
			[MTrans_img_kb_aspect_sentiment_encoder_layer(hid_dim, pf_dim, n_heads, drop_prob)
			 for _ in range(n_layers)])
		self.layer_norm = nn.LayerNorm(hid_dim, eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self, src, src_mask, history_bank , img_bank, img_mask, his_mask, kb_bank,kb_mask,aspect_bank, aspect_mask,sentiment_bank,sentiment_mask):
		""" See :obj:`EncoderBase.forward()`"""

		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		batch_size=src.shape[0]
		src_len=src.shape[1]
		pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
		token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
		src=self.dropout(token_embedding*(self.scale)+pos_embedding)

		# Run the forward pass of every layer of the tranformer.
		for i in range(self.n_layers):
			out = self.mtrans_img_kb_aspect_sentiment_encoder_layer[i](src, src_mask, img_bank, img_mask, history_bank, his_mask, kb_bank, kb_mask,
				                                                                                 aspect_bank, aspect_mask,sentiment_bank,sentiment_mask)
		out = self.layer_norm(out)

		return token_embedding, out.contiguous(), src_mask

class Review_Transformer(nn.Module):
	r""" transformer model 
	Args:
	Inputs: 
	Outputs: 
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, 
				enc_hidden_size, dec_hidden_size,  
				num_enc_layers, num_dec_layers, 
				dropout_enc, dropout_dec,
				max_decode_len, sos_id, eos_id, pad_id, use_review = False):

		super(Review_Transformer, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.enc_pf_dim = 2048
		self.dec_pf_dim = 2048
		self.enc_heads=8
		self.dec_heads=8
		self.num_enc_layers = 2#num_enc_layers
		self.num_dec_layers = 2#num_dec_layers
		self.dropout_enc = dropout_enc #dropout prob for encoder
		self.dropout_dec = dropout_dec #dropout prob for decoder
		self.sos_id = sos_id # start token		
		self.eos_id = eos_id # end token
		self.max_decode_len = max_decode_len # max timesteps for decoder
		self.pad_ind= pad_id
		self.enc_hidden_size = 256#enc_hidden_size
		self.dec_hidden_size = 256#dec_hidden_size

		# Initialize encoder
		self.encoder = review_encoder(self.src_vocab_size, self.enc_hidden_size, 
			self.num_enc_layers, self.enc_heads, self.enc_pf_dim,self.dropout_enc)


		self.decoder = review_decoder(self.tgt_vocab_size, self.dec_hidden_size, 
			self.num_dec_layers, self.dec_pf_dim, self.dec_heads,self.max_decode_len,self.dropout_dec)

		self.init_params()

		# model.apply(initialize_weights);

	def forward(self, src, trg, src_mask = None, decode = False, review = False):
		# text_enc_input == (batch, context_len*src_len)
		#print('tt',text_enc_input[0])
		# print(text_enc_input[29])
		# print(src_mask[29])
		#src_mask = [batch size, 1, 1, src len]
		# print(src_mask.size())
		# print(src_mask)
		#src_mask>>[batch_size,1,1,src_len]
		# print("src##", src.size())
		if not review:
			src_mask = (src != self.pad_ind).unsqueeze(1).unsqueeze(2)
			src = self.encoder(src, src_mask)
		else:
			src = self.encoder(src, src_mask,review = True)

		#src_mask = [batch size, 1, 1, src len]
		# Pass through encoder: 
		# text_enc_in_len[turn,:] # 2D to 1D
		# print(self.use_aspect)
		#src>>[batch_size,src_len,hid_dim]fs
		# print("src", src.size())
		# print("src_mask", src_mask.size())
		if not decode:
			# print("hello")
			output, attention=self.decoder(src, trg, src_mask, self.sos_id, decode)
			#output>>[batch_size,trg_len,output_dim]
			#attention>>[batch_size,n_heads,trg_len,src_len]
			# print("oo1", output.size())
			return output
		else:
			# print(src[0])
			# pred_trgs=self.decoder(src,0,src_mask,self.sos_id,decode)
			# print("src", src.size())
			# print("src_maks")
			pred_trgs=self.decoder(src,0, src_mask,self.sos_id,decode)
			
			# pred_trgs = torch.LongTensor([pred_trgs]).cuda().unsqueeze(1)
			# pred_trgs = torch.LongTensor(pred_trgs).cuda()
			return pred_trgs


	def init_params(self, initrange=0.1):
		for name, param in self.named_parameters():
			if param.dim() > 1:
				nn.init.xavier_uniform_(param)

class review_encoder(nn.Module):
	def __init__(self,input_dim,hid_dim,n_layers,n_heads,pf_dim,drop_prob=0.1):
		super().__init__()

		self.tos_embed=nn.Embedding(input_dim,hid_dim)
		self.pos_embed=nn.Embedding(input_dim,hid_dim)

		self.encoder_layer=nn.ModuleList([review_encoder_layer(hid_dim,pf_dim,n_heads,drop_prob) for i in range(n_layers)])
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()

	def forward(self,src,src_mask, review = False):
		##src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		
		if not review:
			batch_size=src.shape[0]
			src_len=src.shape[1]
			pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).cuda()
			pos_embedding = self.pos_embed(pos) ##[batch_size,src_len,hid_dim]
			token_embedding = self.tos_embed(src) ##[batch_size,src_len,hid_dim]
			src=self.dropout(token_embedding*(self.scale)+pos_embedding)
		else:
			src = src
		# print("src",src.size())

		##src>>[batch_size,src_len,hid_dim]
		for layer in self.encoder_layer:
			src=layer(src,src_mask)

		return src


class review_decoder(nn.Module):
	def __init__(self,output_dim,hid_dim,n_layers,pf_dim,n_heads,max_unroll=40,drop_prob=0.1):
		super().__init__()
		self.tos_embedding=nn.Embedding(output_dim,hid_dim)
		self.pos_embedding=nn.Embedding(output_dim,hid_dim)

		self.layers=nn.ModuleList([review_decoder_layer(hid_dim,n_heads,pf_dim,drop_prob) for _ in range(n_layers)])

		self.fc_out=nn.Linear(hid_dim,output_dim)
		self.sample = False

		# self.fc_out = nn.Linear(hid_dim, output_dim)
		self.dropout=nn.Dropout(p=drop_prob)
		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()
		self.max_unroll=max_unroll
		self.pad_ind=0

	def init_token(self, batch_size, sos_id=8,t=False):
		"""Get Variable of <SOS> Index (batch_size)"""
		# x = torch.LongTensor([sos_id] * batch_size).cuda().unsqueeze(1)
		# old code
		if t:
			x =([sos_id] * batch_size)
		# new code
		else:
			x = ([[sos_id] for _ in range(batch_size)])
		return x

	def trg_mask(self,trg):
		#trg>>[batch_size,trg_len]
		trg_pad_mask=(trg!=self.pad_ind).unsqueeze(1).unsqueeze(2)
		#trg_pad_mask>[batch_size,1,1,trg_len]
		trg_len=trg.shape[1]
		trg_sub_mask=torch.tril(torch.ones((trg_len,trg_len)).cuda()).bool()#.type(torch.uint8)
		#trg_sub_mask>>[trg_len,trg_len]
		trg_mask=trg_pad_mask & trg_sub_mask
		#trg_mask>[batch_size,1,trg_len,trg_len]
		return trg_mask

	def decode(self, out,t=False):
		"""
		Args:
			out: unnormalized word distribution [batch_size, vocab_size]
		Return:
			x: word_index [batch_size]
		"""

		# Sample next word from multinomial word distribution
		
		if self.sample:
			# x: [batch_size] - word index (next input)
			x = torch.multinomial(self.softmax(out / self.temperature), 2).view(-1)
		# Greedy sampling
		else:
			# x: [batch_size] - word index (next input)
			if t:
			# new code
				# print('out',out.size())
				_, x = out.max(dim=2)
				# print('x',x)
				x = x.tolist()
				return x
			# old code
			else:
				# print('out',out.size(/))
				_, x = out.max(dim=2)
				# print('x',x)/

				return x[:,-1].item()
		
		#return out.argmax(2)[:,-1].unsqueeze(1).unsqueeze(2)

	def forward_step(self, enc_src, trg, src_mask,  output_layer = True):
		trg_mask=self.trg_mask(trg)
		# print(trg_mask[3])
		# print(trg_mask.size())
		# print(trg_mask[0])
		batch_size=trg.shape[0]
		trg_len=trg.shape[1]

		pos=torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size,1).cuda()
		pos_embed=self.pos_embedding(pos)
		tos_embed=self.tos_embedding(trg).cuda()
		trg=self.dropout(tos_embed*self.scale+pos_embed)
		##trg>>[batch_size,trg_len,hid_dim]
		# print("trg", trg.size())
		
		for layer in self.layers:
			output,attention=layer(enc_src,trg,src_mask,trg_mask)
		if output_layer:
			output=self.fc_out(output)
		# output=self.fc_out(output)

		#output>>[batch_size,trg_len,output_dim]
		return output,attention

	def forward(self, enc_src, trg, src_mask, sos_id, decode=False):
		##enc_src>>[batch_size,src_len]
		##src_mask>>[batch_size,1,1,src_len]
		##trg_mask>>[batch_size,1,trg_len,trg_len]
		##trg>>[batch_size,trg_len]

		batch_size=enc_src.shape[0]
		# print('batch_size', batch_size)
		trg_indexes = self.init_token(batch_size,sos_id)
		trg_indexes_old=self.init_token(batch_size,sos_id,t=True)
		# print(trg[0])
		# print(trg_indexes.size())
		#x>[batch_size]
		#x=self.init_token(batch_size,sos_id)
		#x=x.unsqueeze(1).cuda()

		if not decode:
			output, attention = self.forward_step(enc_src, trg, src_mask)
			# print("oo",output.size())
			return output, attention
		else:
			# new code
			translations_done = [0] * batch_size
			for i in range(self.max_unroll):
				# print('i',i)
				# x = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()
				x = torch.LongTensor(trg_indexes).cuda()
				# print("batch", batch_size)
				# print("iiinitial trg", x.size())
				# print(enc_src.size())
				# print(src_mask.size())
				with torch.no_grad():
					output, attention = self.forward_step(enc_src, x, src_mask, output_layer = True)

				# print(output[0][0])
				pred_token = self.decode(output, t=True)
				# print('pred_token', pred_token)
				# trg_indexes.append(pred_token)
				for idx,pred_token_i in enumerate(pred_token):
					trg_indexes[idx].append(pred_token[idx][-1])
					if pred_token_i == 3:
						translations_done[i] = 1

				# if pred_token == 3:
					# break
				# if i == 9:
				# 	break
				if all(translations_done):
					break

			# print(trg_indexes)
			# return trg_indexes[1:]
			# return [trg_indexes[idx][1:] for idx,i in enumerate(trg_indexes)]
			return output

			# old code
			
			# trg_indexes=[]
			# translations_done = [0] * batch_size
			# for i in range(self.max_unroll):
			# 	print('ii',i)
			# 	x = torch.LongTensor(trg_indexes_old).unsqueeze(0).cuda()
			# 	# x = torch.LongTensor(trg_indexes).cuda()
			# 	# print("batch", batch_size)
			# 	# print("initial trg", x.size())
			# 	# print(enc_src.size())
			# 	# print(src_mask.size())
			# 	with torch.no_grad():
			# 		output, attention = self.forward_step(enc_src, x, src_mask)

			# 	# print(output[0][0])
			# 	pred_token = self.decode(output)
			# 	print('pred_token ok', pred_token)

			# 	trg_indexes_old.append(pred_token)
			# 	# for idx,pred_token_i in enumerate(pred_token):
			# 		# trg_indexes[idx].append(pred_token[idx][0])
			# 		# if pred_token_i == 3:
			# 			# translations_done[i] = 1

			# 	if pred_token == 3:
			# 		break
			# 	# if all(translations_done):
			# 		# break
			# print('old',trg_indexes_old[1:])
			# return trg_indexes_old[1:]
			# return [trg_indexes[idx][1:] for idx,i in enumerate(trg_indexes)]
			

# class review_decoder(nn.Module):
# 	def __init__(self,output_dim,hid_dim,n_layers,pf_dim,n_heads,max_unroll=40,drop_prob=0.1):
# 		super().__init__()
# 		self.tos_embedding=nn.Embedding(output_dim,hid_dim)
# 		self.pos_embedding=nn.Embedding(output_dim,hid_dim)
# 		# self.output_dim = output_dim
# 		self.hid_dim = hid_dim

# 		self.layers=nn.ModuleList([review_decoder_layer(hid_dim,n_heads,pf_dim,drop_prob) for _ in range(n_layers)])

# 		self.fc_out=nn.Linear(hid_dim,output_dim)
# 		self.sample = False

# 		# self.fc_out = nn.Linear(hid_dim, output_dim)
# 		self.dropout=nn.Dropout(p=drop_prob)
# 		self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).cuda()
# 		self.max_unroll=max_unroll
# 		self.pad_ind=0

# 	def init_token(self, batch_size, sos_id=2):
# 		"""Get Variable of <SOS> Index (batch_size)"""
# 		# x = torch.LongTensor([sos_id] * batch_size).cuda().unsqueeze(1)
# 		x =([sos_id] * batch_size)
# 		# x = ([[sos_id] for _ in range(batch_size)])
# 		return x

# 	def trg_mask(self,trg):
# 		#trg>>[batch_size,trg_len]
# 		trg_pad_mask=(trg!=self.pad_ind).unsqueeze(1).unsqueeze(2)
# 		#trg_pad_mask>[batch_size,1,1,trg_len]
# 		trg_len=trg.shape[1]
# 		trg_sub_mask=torch.tril(torch.ones((trg_len,trg_len)).cuda()).bool()#.type(torch.uint8)
# 		#trg_sub_mask>>[trg_len,trg_len]
# 		trg_mask=trg_pad_mask & trg_sub_mask
# 		#trg_mask>[batch_size,1,trg_len,trg_len]
# 		return trg_mask

# 	def decode(self, out):
# 		"""
# 		Args:
# 			out: unnormalized word distribution [batch_size, vocab_size]
# 		Return:
# 			x: word_index [batch_size]
# 		"""

# 		# Sample next word from multinomial word distribution
		
# 		if self.sample:
# 			# x: [batch_size] - word index (next input)
# 			x = torch.multinomial(self.softmax(out / self.temperature), 2).view(-1)
# 		# Greedy sampling
# 		else:
# 			# x: [batch_size] - word index (next input)
# 			_, x = out.max(dim=2)
# 			# x = x.tolist()
# 		return x[:,-1].item()
# 		# return x
		
# 		#return out.argmax(2)[:,-1].unsqueeze(1).unsqueeze(2)

# 	def forward_step(self, enc_src, trg, src_mask, output_layer = True):
# 		# print("helllo")
# 		trg_mask=self.trg_mask(trg)
# 		# print(trg_mask[3])
# 		# print(trg_mask.size())
# 		# print(trg_mask[0])
# 		batch_size=trg.shape[0]
# 		trg_len=trg.shape[1]

# 		pos=torch.arange(0,trg_len).unsqueeze(0).repeat(batch_size,1).cuda()
# 		pos_embed=self.pos_embedding(pos)
# 		tos_embed=self.tos_embedding(trg).cuda()
# 		trg=self.dropout(tos_embed*self.scale+pos_embed)
# 		##trg>>[batch_size,trg_len,hid_dim]
# 		# print("trg forward", trg.size())
		
# 		for layer in self.layers:
# 			output,attention=layer(enc_src,trg,src_mask,trg_mask)
# 		# output=self.fc_out(output)
# 		if output_layer:
# 			output=self.fc_out(output)

# 		#output>>[batch_size,trg_len,output_dim]
# 		return output,attention

# 	def forward(self, enc_src, trg, src_mask, sos_id, decode=False):
# 		##enc_src>>[batch_size,src_len]
# 		##src_mask>>[batch_size,1,1,src_len]
# 		##trg_mask>>[batch_size,1,trg_len,trg_len]
# 		##trg>>[batch_size,trg_len]

# 		batch_size=enc_src.shape[0]
# 		trg_indexes = self.init_token(batch_size,sos_id)
# 		# print(batch_size)
# 		# print(trg[0])
# 		# print(trg_indexes.size())
# 		#x>[batch_size]
# 		#x=self.init_token(batch_size,sos_id)
# 		#x=x.unsqueeze(1).cuda()
	
# 		if not decode:
# 			output, attention = self.forward_step(enc_src, trg, src_mask, True)
# 			# print("oo",output.size())
# 			return output, attention
# 		else:
# 			# trg_indexes=[]
# 			# outputs = torch.zeros(batch_size, self.max_unroll, self.hid_dim).cuda()
# 			# translations_done = [0] * batch_size

# 			for i in range(self.max_unroll):
# 				x = torch.LongTensor(trg_indexes).unsqueeze(0).cuda()
# 				# x = torch.LongTensor(trg_indexes).cuda()

# 				with torch.no_grad():
# 					# print("hi")
# 					# print("initial trg", x.size())
# 					output, attention = self.forward_step(enc_src, x, src_mask, False)

# 				# print(output[0][0])
# 				pred_token = self.decode(output)
# 				trg_indexes.append(pred_token)
# 				# for idx,pred_token_i in enumerate(pred_token):
# 					# if pred_token_i == 3:
# 						# translations_done[i] = 1
# 					# trg_indexes[idx].append(pred_token[idx][0])
				
# 				# if all(translations_done):
# 					# break

# 				if pred_token == 3:
# 					break
# 			# print("trg index", len(trg_indexes[0]))
# 			# print(trg_indexes[0][0], trg_indexes[1][0])
# 			# print("success")
# 			return trg_indexes[1:]
# 			# return [trg_indexes[idx][1:] for idx,i in enumerate(trg_indexes)]
# 			# return output