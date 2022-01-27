import torch
import torch.nn as nn
from .sublayers_tformer import multi_head_attention, position_feed_forward, review_multi_head_attention, review_position_feed_forward


class encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super().__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.layer_norm_self_attn=nn.LayerNorm(hid_dim)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm=nn.LayerNorm(hid_dim)

		self.dropout=nn.Dropout(p=drop_prob)
	def forward(self,src,src_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		_src,attn=self.self_attention(src,src,src,src_mask)
		src=self.layer_norm_self_attn(self.dropout(_src)+src)
		##src>>[batch_size,src_len,hidden]
		_src=self.position_feed(src)
		##src>>[batch_size,src_len,hid_dim]
		src=self.layer_norm(self.dropout(_src)+src)
		##src>>[batch_size,src_len,hid_dim]
		return src

class context_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(context_encoder_layer,self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim)
		self.layer_norm_2=nn.LayerNorm(hid_dim)
		self.dropout=nn.Dropout(p=drop_prob)
	def forward(self,src,src_mask,his_bank,his_mask):

		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		##query>>[batch_size,src_len,hidden]

		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			out_norm = self.layer_norm_2(query)
			out,attn3=self.context_attn(out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+query
			return self.position_feed(out)
		else:
			return self.position_feed(query)

class MTrans_kb_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_kb_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,his_bank,his_mask,kb_bank, kb_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]

		knl_out,attn3=self.knowledge_attn(query_norm,kb_bank,kb_bank,kb_mask)
		knl_out=self.dropout(knl_out)+query
		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			knl_out_norm = self.layer_norm_4(knl_out)
			out,attn3=self.context_attn(knl_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+knl_out
			return self.position_feed(out)
		else:
			return self.position_feed(knl_out)

class MTrans_aspect_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_aspect_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		# self.sentiment_attn= multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		# self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,his_bank,his_mask,aspect_bank, aspect_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]

		knl_out,attn3=self.knowledge_attn(query_norm,aspect_bank,aspect_bank,aspect_mask)
		knl_out=self.dropout(knl_out)+query
		# knl_out_norm = self.layer_norm_3(knl_out)

		# sentiment_out, attn4 = self.sentiment_attn(knl_out_norm, sentiment_bank, sentiment_bank, sentiment_mask)
		# sentiment_out = self.dropout(sentiment_out)+query
		##knl_out>>[batch_size,src_len,hid_dim]
		# if his_bank is not None:
		# 	sentiment_out_norm = self.layer_norm_4(sentiment_out)
		# 	out,attn3=self.context_attn(sentiment_out,his_bank,his_bank,his_mask)
		# 	out=self.dropout(out)+sentiment_out
		# 	return self.position_feed(out)
		# else:
		# 	return self.position_feed(sentiment_out)

		if his_bank is not None:
			knl_out_norm = self.layer_norm_4(knl_out)
			out,attn3=self.context_attn(knl_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+knl_out
			return self.position_feed(out)
		else:
			return self.position_feed(knl_out)

class MTrans_aspect_sentiment_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_aspect_sentiment_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.sentiment_attn= multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,his_bank,his_mask,aspect_bank, aspect_mask, sentiment_bank, sentiment_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]

		knl_out,attn3=self.knowledge_attn(query_norm,aspect_bank,aspect_bank,aspect_mask)
		knl_out=self.dropout(knl_out)+query
		knl_out_norm = self.layer_norm_3(knl_out)

		sentiment_out, attn4 = self.sentiment_attn(knl_out_norm, sentiment_bank, sentiment_bank, sentiment_mask)
		sentiment_out = self.dropout(sentiment_out)+query
		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			sentiment_out_norm = self.layer_norm_4(sentiment_out)
			out,attn3=self.context_attn(sentiment_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+sentiment_out
			return self.position_feed(out)
		else:
			return self.position_feed(sentiment_out)

class MTrans_img_kb_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_img_kb_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,img_input,img_mask,his_bank,his_mask,kb_bank, kb_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]


		img_out,attn2=self.image_attn(query_norm,img_input,img_input,img_mask)
		img_out=self.dropout(img_out)+query
		img_out_norm=self.layer_norm_3(img_out)
		knl_out,attn3=self.knowledge_attn(img_out_norm,kb_bank,kb_bank,kb_mask)
		knl_out=self.dropout(knl_out)+img_out
		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			knl_out_norm = self.layer_norm_4(knl_out)
			out,attn3=self.context_attn(knl_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+knl_out
			return self.position_feed(out)
		else:
			return self.position_feed(knl_out)

class MTrans_kb_aspect_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_kb_aspect_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.aspect_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,aspect_input,aspect_mask,his_bank,his_mask,kb_bank, kb_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]


		aspect_out,attn2=self.aspect_attn(query_norm,aspect_input,aspect_input,aspect_mask)
		aspect_out=self.dropout(aspect_out)+query
		aspect_out_norm=self.layer_norm_3(aspect_out)
		knl_out,attn3=self.knowledge_attn(aspect_out_norm,kb_bank,kb_bank,kb_mask)
		knl_out=self.dropout(knl_out)+aspect_out
		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			knl_out_norm = self.layer_norm_4(knl_out)
			out,attn3=self.context_attn(knl_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+knl_out
			return self.position_feed(out)
		else:
			return self.position_feed(knl_out)

class MTrans_kb_aspect_sentiment_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_kb_aspect_sentiment_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.aspect_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.sentiment_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_5=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,aspect_input,aspect_mask,his_bank,his_mask,kb_bank, kb_mask,sentiment_bank, sentiment_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]


		aspect_out,attn2=self.aspect_attn(query_norm,aspect_input,aspect_input,aspect_mask)
		aspect_out=self.dropout(aspect_out)+query
		aspect_out_norm=self.layer_norm_3(aspect_out)
		knl_out,attn3=self.knowledge_attn(aspect_out_norm,kb_bank,kb_bank,kb_mask)
		knl_out=self.dropout(knl_out)+aspect_out
		knl_out_norm = self.layer_norm_4(knl_out)
		sentiment_out,attn4=self.sentiment_attn(knl_out_norm,sentiment_bank,sentiment_bank,sentiment_mask)
		sentiment_out=self.dropout(sentiment_out)+knl_out

		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			sentiment_out_norm = self.layer_norm_5(sentiment_out)
			out,attn5=self.context_attn(sentiment_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+sentiment_out
			return self.position_feed(out)
		else:
			return self.position_feed(sentiment_out)

class MTrans_img_aspect_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_img_aspect_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,img_input,img_mask,his_bank,his_mask,aspect_bank, aspect_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]


		img_out,attn2=self.image_attn(query_norm,img_input,img_input,img_mask)
		img_out=self.dropout(img_out)+query
		img_out_norm=self.layer_norm_3(img_out)
		aspect_out,attn3=self.knowledge_attn(img_out_norm,aspect_bank,aspect_bank,aspect_mask)
		aspect_out=self.dropout(aspect_out)+img_out
		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			aspect_out_norm = self.layer_norm_4(aspect_out)
			out,attn3=self.context_attn(aspect_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+aspect_out
			return self.position_feed(out)
		else:
			return self.position_feed(aspect_out)

class MTrans_img_aspect_sentiment_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_img_aspect_sentiment_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.sentiment_attn= multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_5=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,img_input,img_mask,his_bank,his_mask,aspect_bank, aspect_mask,sentiment_bank, sentiment_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]


		img_out,attn2=self.image_attn(query_norm,img_input,img_input,img_mask)
		img_out=self.dropout(img_out)+query
		img_out_norm=self.layer_norm_3(img_out)
		aspect_out,attn3=self.knowledge_attn(img_out_norm,aspect_bank,aspect_bank,aspect_mask)
		aspect_out=self.dropout(aspect_out)+img_out
		aspect_out_norm = self.layer_norm_4(aspect_out)
		sentiment_out,attn4=self.sentiment_attn(aspect_out_norm,sentiment_bank, sentiment_bank, sentiment_mask)
		sentiment_out=self.dropout(sentiment_out)+aspect_out
		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			sentiment_out_norm = self.layer_norm_5(sentiment_out)
			out,attn5=self.context_attn(sentiment_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+sentiment_out
			return self.position_feed(out)
		else:
			return self.position_feed(sentiment_out)

class MTrans_img_kb_aspect_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_img_kb_aspect_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.aspect_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_5=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,img_input,img_mask,his_bank,his_mask,kb_bank, kb_mask,aspect_bank, aspect_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]


		img_out,attn2=self.image_attn(query_norm,img_input,img_input,img_mask)
		img_out=self.dropout(img_out)+query
		img_out_norm=self.layer_norm_3(img_out)
		knl_out,attn3=self.knowledge_attn(img_out_norm,kb_bank,kb_bank,kb_mask)
		knl_out=self.dropout(knl_out)+img_out
		knl_out_norm = self.layer_norm_4(knl_out)
		##knl_out>>[batch_size,src_len,hid_dim]
		aspect_out,attn4=self.aspect_attn(knl_out_norm,aspect_bank,aspect_bank,aspect_mask)
		aspect_out=self.dropout(aspect_out)+knl_out
		if his_bank is not None:
			aspect_out_norm = self.layer_norm_5(aspect_out)
			out,attn5=self.context_attn(aspect_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+aspect_out
			return self.position_feed(out)
		else:
			return self.position_feed(aspect_out)

class MTrans_img_kb_aspect_sentiment_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_img_kb_aspect_sentiment_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.knowledge_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.sentiment_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.aspect_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_4=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_5=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_6=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,img_input,img_mask,his_bank,his_mask,kb_bank, kb_mask,aspect_bank, aspect_mask,sentiment_bank,sentiment_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]


		img_out,attn2=self.image_attn(query_norm,img_input,img_input,img_mask)
		img_out=self.dropout(img_out)+query
		img_out_norm=self.layer_norm_3(img_out)
		knl_out,attn3=self.knowledge_attn(img_out_norm,kb_bank,kb_bank,kb_mask)
		knl_out=self.dropout(knl_out)+img_out
		knl_out_norm = self.layer_norm_4(knl_out)
		##knl_out>>[batch_size,src_len,hid_dim]
		aspect_out,attn4=self.aspect_attn(knl_out_norm,aspect_bank,aspect_bank,aspect_mask)
		aspect_out=self.dropout(aspect_out)+knl_out
		aspect_out_norm = self.layer_norm_5(aspect_out)

		sentiment_out,attn5= self.sentiment_attn(aspect_out_norm,sentiment_bank,sentiment_bank,sentiment_mask)
		sentiment_out=self.dropout(sentiment_out)+aspect_out
		if his_bank is not None:
			sentiment_out_norm = self.layer_norm_6(sentiment_out)
			out,attn6=self.context_attn(sentiment_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+sentiment_out
			return self.position_feed(out)
		else:
			return self.position_feed(sentiment_out)

class MTrans_img_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super(MTrans_img_encoder_layer, self).__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.image_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.context_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm_1=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_2=nn.LayerNorm(hid_dim,eps=1e-6)
		self.layer_norm_3=nn.LayerNorm(hid_dim,eps=1e-6)
		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,src,src_mask,img_input,img_mask,his_bank,his_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		input_norm=self.layer_norm_1(src)
		query,attn1=self.self_attention(input_norm,input_norm,input_norm,src_mask)
		#query == [batch_size,src_len,hid_dim]
		query=self.dropout(query)+src
		query_norm=self.layer_norm_2(query)
		##query>>[batch_size,src_len,hidden]


		knl_out,attn2=self.image_attn(query_norm,img_input,img_input,img_mask)
		knl_out=self.dropout(knl_out)+query
		##knl_out>>[batch_size,src_len,hid_dim]
		if his_bank is not None:
			knl_out_norm = self.layer_norm_3(knl_out)
			out,attn3=self.context_attn(knl_out_norm,his_bank,his_bank,his_mask)
			out=self.dropout(out)+knl_out
			return self.position_feed(out)
		else:
			return self.position_feed(knl_out)

class decoder_layer(nn.Module):
	def __init__(self,hid_dim,n_heads,pf_dim,drop_prob=0.1):
		super().__init__()
		self.self_attention=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.self_attn_layer_norm=nn.LayerNorm(hid_dim)
		self.encoder_attn=multi_head_attention(hid_dim,n_heads,drop_prob)
		self.enc_attn_layer_norm=nn.LayerNorm(hid_dim)
		self.kb_attn = multi_head_attention(hid_dim,n_heads,drop_prob)
		self.kb_attn_layer_norm = nn.LayerNorm(hid_dim)

		self.fc_layer_norm=nn.LayerNorm(hid_dim)
		self.position_feed=position_feed_forward(hid_dim,pf_dim,drop_prob)

		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self, enc_src, trg, src_mask, trg_mask, review_kb=None):
		#enc_src>[batch_size,src_len,hid_dim]
		#trg>[batch_size,trg_len,hid_dim]
		#src_mask>[batch_size,1,1,src_len]
		#trg_mask>{batch_size,1,trg_len,trg_len}
		# print('gg',review_kb.size())

		_trg,attn=self.self_attention(trg,trg,trg,trg_mask)
		trg=self.self_attn_layer_norm(self.dropout(_trg)+trg)
		##trg>>[batch_size,trg_len,hid_dim]
		_trg,attn=self.encoder_attn(trg,enc_src,enc_src,src_mask)
		##_trg>>[batch_size,trg_len,hid_dim]
		trg=self.enc_attn_layer_norm(self.dropout(_trg)+trg)
		#trg>>[batch_size,trg_len,hid_dim]
		if review_kb is not None:
			__trg, kb_attn = self.kb_attn(trg, review_kb, review_kb)
			trg = self.kb_attn_layer_norm(self.dropout(__trg)+ trg)
			_trg=self.position_feed(trg)
			trg=self.fc_layer_norm(self.dropout(_trg)+trg)
			return trg, kb_attn
		else:
			_trg=self.position_feed(trg)
			trg=self.fc_layer_norm(self.dropout(_trg)+trg)
			return trg, attn

class review_encoder_layer(nn.Module):
	def __init__(self,hid_dim,pf_dim,n_heads,drop_prob=0.1):
		super().__init__()
		self.self_attention= review_multi_head_attention(hid_dim,n_heads,drop_prob)
		self.layer_norm_self_attn=nn.LayerNorm(hid_dim)
		self.position_feed= review_position_feed_forward(hid_dim,pf_dim,drop_prob)
		self.layer_norm=nn.LayerNorm(hid_dim)

		self.dropout=nn.Dropout(p=drop_prob)
	def forward(self,src,src_mask):
		##src>>[batch_size,src_len,hidden]
		##src_mask>>[batch_size,1,1,src_len]
		# print("src", src.size())
		# print("src mask", src_mask.size())
		_src,attn=self.self_attention(src,src,src,src_mask)
		src=self.layer_norm_self_attn(self.dropout(_src)+src)
		##src>>[batch_size,src_len,hidden]
		_src=self.position_feed(src)
		##src>>[batch_size,src_len,hid_dim]
		src=self.layer_norm(self.dropout(_src)+src)
		##src>>[batch_size,src_len,hid_dim]
		return src


class review_decoder_layer(nn.Module):
	def __init__(self,hid_dim,n_heads,pf_dim,drop_prob=0.1):
		super().__init__()
		self.self_attention= review_multi_head_attention(hid_dim,n_heads,drop_prob)
		self.self_attn_layer_norm=nn.LayerNorm(hid_dim)
		self.encoder_attn= review_multi_head_attention(hid_dim,n_heads,drop_prob)
		self.enc_attn_layer_norm=nn.LayerNorm(hid_dim)
		self.fc_layer_norm=nn.LayerNorm(hid_dim)
		self.position_feed= review_position_feed_forward(hid_dim,pf_dim,drop_prob)

		self.dropout=nn.Dropout(p=drop_prob)

	def forward(self,enc_src,trg,src_mask,trg_mask):
		#enc_src>[batch_size,src_len,hid_dim]
		#trg>[batch_size,trg_len,hid_dim]
		#src_mask>[batch_size,1,1,src_len]
		#trg_mask>{batch_size,1,trg_len,trg_len}

		_trg,attn=self.self_attention(trg,trg,trg,trg_mask)
		trg=self.self_attn_layer_norm(self.dropout(_trg)+trg)
		##trg>>[batch_size,trg_len,hid_dim]
		# print("review_decoder_layer")
		# print("trg", trg.size())
		# print("enc_src", enc_src.size())
		# print("src mask", src_mask.size())
		_trg,attn=self.encoder_attn(trg,enc_src,enc_src,src_mask)
		##_trg>>[batch_size,trg_len,hid_dim]
		trg=self.enc_attn_layer_norm(self.dropout(_trg)+trg)
		#trg>>[batch_size,trg_len,hid_dim]
		_trg=self.position_feed(trg)
		trg=self.fc_layer_norm(self.dropout(_trg)+trg)
		return trg,attn