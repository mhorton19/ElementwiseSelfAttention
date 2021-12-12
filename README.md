# ElementwiseSelfAttention
A O(N) subtitute/augmentation to the transformer architecture inspired by self-attention

This architecture was trained from scratch to perform sentiment analysis on this 2015 kaggle challenge https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/overview based on this collab notebook https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta

This architecture with no pretraining performs similarly to many of the LSTM solutions (around 62% without ensembling)

GOALS:
1) Make a O(N) (wrt sequence length) non-recurrent model that can be trained end to end for NLP tasks with single outputs (as opposed to sequence outputs). DONE!
2) Use this architecture to finetine pretrained tranformers without isolating a single output vector (such as is done with BERT finetuning).
3) Use this architecture to augment current transformer models.  This model requires a similar amount of parameters to a standard transformer, but far less memory and computation.  Therefore, maybe it is a cheap way to increase parameters and improve performance without adding much compute.  It may also be a cheap way to reduce depth and number of heads because global context can do a lot of the work of increasing the "receptive field" of the self-attention operation.
4) Use this architecture to enable transformer models with a limited attention field to deal with very long sequences while still incorporating information from the whole sequence.

The main operation is implemented in elementwise_sa_layer.py for reference and is not too verbose.  

Here is how it works. Instead of implementing pairwise attention between every member of the sequence, there is a elementwise global attention vector (E) which incorporates information from the vectors in the sequence.  Each element of the sequence is linearly transformed to produce 

1) a value vector (V) the length of the global attention vector
2) an attention vector (A) the length of the global attention vector

softmax(A, dim=1) * V then produces E which has each of its elements set by a self-attention-esque operation across member of the sequence.

Once E is produced, each member of the sequence I then produces some gating value to multiply against E, which is then reduced via linear transform and added back to I as a residual value.  Before the final layer, E is returned as a summarization of the sequence.

Input (I): batch size x sequence length x input feature dimension (N, L, D)
Attention (A) via linear transformation of I: batch size x sequence length x expanded intermediate feature dimension (N, L, De)
Values (V) via linear transformation of I: batch size x sequence length x expanded intermediate feature dimension (N, L, De)
Elementwise global attention vector (E) via softmax(A, dim=1) * V: batch size x sequence length x expanded intermediate feature dimension (N, 1, De)
Retrieval gating values (R) via linear tranformation of I: batch size x sequence length x expanded intermediate feature dimension (N, L, De)
Output (O) via linear transformation of (R * tanh(E)): batch size x sequence length x input feature dimension (N, L, D)

I am also using residual connections as described by the prenorm layer https://arxiv.org/abs/2002.04745 as well as a fully connected portion similar to the original transformer model.

ROOM FOR IMPROVEMENT:
1. I want to find a better way to reincorporate information from E into each member of the output sequence.  Simply adding with no modification seems like the most natural solution, but this would keep increasing the correlation between each member of the output sequence which worries me. I tried 3 forms of gating to alleviate this: sigmoid, linear, and tanh. sigmoid would seem to be the most natural of these 3, but it still theoretically will be expected to produce highly correlated outputs.  Linear is strange considering that I have not seen elementwise vector multiplication used this way in deep learning literature. Tanh is the one I settled on, but I do not see a large performance difference between these three and I am not convinced that any is optimal.
2. Since E is a vector and not a sequence, should we use batch normalization? Does it matter?  
3. Should E be used to more directly modify the next layer's attention operation? Do residual connections across E make sense?
4. I am using layer norm on E, which is a departure from the original transformer architecture.  Is this a bad idea?

FUTURE WORK:
Combine with transformers and see what happens
