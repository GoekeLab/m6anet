import toml
from torch.nn import Module, Identity, Sequential
from .model_blocks.pooling_blocks import PoolingFilter


class MILModel(Module):

    def __init__(self, model_config):
        super(MILModel, self).__init__()

        self.model_config = model_config
        self.read_level_encoder = None
        self.pooling_filter = None
        self.decoder = None

        # Building model block by block
        self.build_model()

    def build_model(self):
        blocks = self.model_config['block']
        seq_model = []
        for block in blocks:
            block_type = block.pop('block_type')
            block_obj = self._build_block(block_type, **block)
        
            if isinstance(block_obj, PoolingFilter):
                if len(seq_model) > 0:
                    self.read_level_encoder = Sequential(*seq_model)
                else:
                    self.read_level_encoder = None

                self.pooling_filter = block_obj
                seq_model = []
            else:
                seq_model.append(block_obj)
        
        if (self.read_level_encoder is None) and (self.pooling_filter is None):
            self.read_level_encoder = Sequential(*seq_model)
            self.pooling_filter = Identity()
            self.decoder = Identity()
        else:
            if len(seq_model) == 0:
                self.decoder = Identity()
            else:
                self.decoder = Sequential(*seq_model)

    def _build_block(self, block_type, **kwargs):
        from . import model_blocks
        block_obj = getattr(model_blocks, block_type)
        return block_obj(**kwargs)

    def get_read_representation(self, x):
        if self.read_level_encoder is None:
            return x
        else:
            return self.read_level_encoder(x)
    
    def get_read_probability(self, x):
        read_representation = self.get_read_representation(x)
        return self.pooling_filter.predict_read_level_prob(read_representation)

    def get_site_representation(self, x):
        return self.pooling_filter(self.get_read_representation(x))
    
    def get_site_probability(self, x):
        return self.decoder(self.get_site_representation(x))

    def get_read_site_probability(self, x):
        read_representation = self.get_read_representation(x)
        read_level_probability = self.pooling_filter.predict_read_level_prob(read_representation)
        site_level_probability = self.decoder(self.pooling_filter(read_representation))
        return read_level_probability, site_level_probability, read_representation
    
    def get_attention_weights(self, x):
        if hasattr(self.pooling_filter, "get_attention_weights"):
            return self.pooling_filter.get_attention_weights(self.get_read_representation(x))
        else:
            raise ValueError("Pooling filter does not have attention weights")

    def forward(self, x):
        return self.get_site_probability(x)
