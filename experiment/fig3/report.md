Under llama7b, test the perplexity in pg19 datasets based on streamingllm, dense, windom attention strategy.
This experiment haven't conver the recomputation of windom attention. It is not covered in the provided code and need to further investigate in the corresponding paper.

等价于 fig3  关于 llama的实验 

注：模型加载时会把整个数据集下载下来才能用，但实在太慢了，所以稍微改了一下读取方式，变成streaming读取，只下载第一本书（seq_len:65134)，从而 data的接口也需要稍微修改。

Result:

dense:
5k: 339.380126953125
10K: 10286.7822265625
15K: 33157.78515625
20K: 28172.08984375

streaming:
5k: 54.57406997680664
10K: 123.4581527709961
15K: 171.97267150878906
20K: 197.3938751220703

window:
5k: 90.34803771972656
10K: 158.1236572265625
15K: 202.95611572265625
20K: 223.79295349121094
