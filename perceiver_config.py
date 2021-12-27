class PerceiverConfig:
    def __init__(self) -> None:
        self.num_heads = 8
        self.widening_factor = 2
        self.dropout = 0.1
        self.ignore_first_cross_attention = True
        # first cross attention is not involved in weight sharing
        self.num_self_attentions = 6
        self.num_blocks = 8

        self.d_latents = 32
        self.num_latents = 32

        self.d_inputs = 64
        self.d_outputs = 64
        self.num_outputs = 128
