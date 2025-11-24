from flash_attn_interface import flash_attn_varlen_func
from flash_attn.bert_padding import pad_input, unpad_input
from einops import rearrange

def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None):
    """
    qkv: Tensor of shape [B, S, 3, H, D]
    key_padding_mask: BoolTensor of shape [B, S]
    """
    B, S, three, H, D = qkv.shape

    # 1) 先把 QKV 三个张量合并到最后一个维度，方便 unpad
    x = rearrange(qkv, "b s three h d -> b s (three h d)")

    # 2) 去掉 padding
    #    x_unpad:    [NNZ, three * H * D]
    #    indices:    用于还原回原始位置
    #    cu_seqlens: [B+1] prefix sum，用来定位每个样本在 x_unpad 里的起止
    #    max_s:      去 pad 后最长序列长度
    x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(x, key_padding_mask)

    # 3) 恢复成 [NNZ, 3, H, D]，并拆成 Q, K, V 三块
    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=H)
    q_unpad = x_unpad[:, 0]  # [NNZ, H, D]
    k_unpad = x_unpad[:, 1]  # [NNZ, H, D]
    v_unpad = x_unpad[:, 2]  # [NNZ, H, D]

    # 4) 调用 FlashAttention 3 的变长前向接口
    #    注意：这里假设 q/k 使用相同的 cu_seqlens 和 max_s
    output_unpad, _ = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens,   # cu_seqlens_q
        cu_seqlens,   # cu_seqlens_k
        max_s,        # max_seqlen_q
        max_s,        # max_seqlen_k
        softmax_scale=softmax_scale,
        causal=causal,
    ) # 返回 [NNZ, H, D]

    # 5) 扁平化，再 pad 回去
    out_flat = rearrange(output_unpad, "nnz h d -> nnz (h d)")
    padded = pad_input(out_flat, indices, B, S)  # [B, S, H*D]

    # 6) 恢复成 [B, S, H, D]
    output = rearrange(padded, "b s (h d) -> b s h d", h=H)
    return output
