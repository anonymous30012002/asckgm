import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class multi_head_attention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

# class multi_head_attention(nn.Module):
# 	def __init__(self,hid_dim,n_heads,drop_prob=0.1):
# 		super().__init__()
# 		#assert hid_dim%n_heads==0

# 		self.hid_dim=hid_dim
# 		self.n_heads=n_heads
# 		self.head_dim=self.hid_dim // self.n_heads
# 		self.fc_q=nn.Linear(hid_dim,hid_dim)
# 		self.fc_k=nn.Linear(hid_dim,hid_dim)
# 		self.fc_v=nn.Linear(hid_dim,hid_dim)
# 		self.fc_o=nn.Linear(hid_dim,hid_dim)

# 		self.dropout=nn.Dropout(p=drop_prob)
# 		self.scale=torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()


# 	def forward(self,query,key,value,mask=None):
# 		#query>>[batch_size,query_len,hid_dim]
# 		#key>>[batch_size,key_len,hid_dim]
# 		#value>>[batch_size,val_len,hid_dim]
# 		q_len=query.shape[1]
# 		v_len=value.shape[1]
# 		k_len=key.shape[1]
		
# 		batch_size=query.shape[0]
# 		print('batch_size',batch_size)
# 		Q=self.fc_q(query)
# 		K=self.fc_k(key)
# 		V=self.fc_v(value)

# 		Q=Q.reshape(batch_size,q_len,self.n_heads,self.head_dim) ##[batch_size,query_len,n_heads,head_dim]
# 		V=V.reshape(batch_size,v_len,self.n_heads,self.head_dim) ##[batch_size,val_len,n_heads,head_dim]
# 		K=K.reshape(batch_size,k_len,self.n_heads,self.head_dim) ##[batch_size,key_len,n_heads,head_dim]

# 		Q=Q.permute(0,2,1,3)##[batch_size,n_heads,query_len,head_dim]
# 		V=V.permute(0,2,1,3)##[batch_size,n_heads,val_len,head_dim]
# 		K=K.permute(0,2,1,3)##[batch_size,n_heads,key_len,head_dim]

# 		energy=torch.matmul(Q,K.permute(0,1,3,2))/self.scale
# 		##energy>>[batch_size,n_heads,query_len,key_len]
# 		if mask is not None:
# 			energy.masked_fill(mask==0,-1e10)

# 		attention=torch.softmax(energy,dim=-1)
# 		##attention>>[batch_size,n_heads,query_len,key_len]
# 		x=torch.matmul(self.dropout(attention),V)
# 		##x>>[batch_size,n_heads,query_len,head_dim]
# 		x=x.permute(0,2,1,3)
# 		##x>>[batch_size,query_len,n_heads,head_dim]
# 		x=x.reshape(batch_size,-1,self.hid_dim)
# 		##x>>[batch_size,query_len,hid_dim]
# 		x=self.fc_o(x)
# 		##x>>[batch_size,query_len,hid_dim]
# 		return x,attention



class position_feed_forward(nn.Module):
	def __init__(self,hid_dim,pf_dim,drop_prob=0.1):
		super().__init__()

		self.fc1=nn.Linear(hid_dim,pf_dim)
		self.fc2=nn.Linear(pf_dim,hid_dim)
		self.dropout=nn.Dropout(p=drop_prob)


	def forward(self,x):
		x=self.dropout(F.relu(self.fc1(x)))
		##x>>[batch_size,src_len,pf_dim]
		x=self.fc2(x)
		##x>>[batch_size,src_len,hid_dim]
		return x

class review_multi_head_attention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        
    def forward(self, query, key, value, mask = None):
        
        # print(query)
        batch_size = query.shape[0]
        # print("query", query.size())      
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        # print("key", key.size())
        # print("query", query.size())
        # print("value", value.size())        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        # print("energy", energy.size())
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention


class review_position_feed_forward(nn.Module):
    def __init__(self,hid_dim,pf_dim,drop_prob=0.1):
        super().__init__()

        self.fc1=nn.Linear(hid_dim,pf_dim)
        self.fc2=nn.Linear(pf_dim,hid_dim)
        self.dropout=nn.Dropout(p=drop_prob)


    def forward(self,x):
        x=self.dropout(F.relu(self.fc1(x)))
        ##x>>[batch_size,src_len,pf_dim]
        x=self.fc2(x)
        ##x>>[batch_size,src_len,hid_dim]
        return x