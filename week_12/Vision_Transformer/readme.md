## Vision Transformers(ViT's):



Explain with following things with reference to  [FILE](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py)

- Block

- Embeddings

- MLP

- Attention

- Encoder

  

### How the Vision Transformer works in a nutshell 

1. Split an image into patches
2. Flatten the patches
3. Produce lower-dimensional linear embeddings from the flattened patches
4. Add positional embeddings
5. Feed the sequence as an input to a standard transformer encoder
6. Pretrain the model with image labels (fully supervised on a huge dataset)
7. Finetune on the downstream dataset for image classification

Source: https://theaisummer.com/vision-transformer/

### Embedding

- The first step is to break-down the image into patches and flatten them.
- These patches are projected using a normal linear layer, a Conv2d layer is used for this for the same.
- Next step is to add the position embedding. 



**Step 1:**

![step_1](/home/adit/Documents/GitHub/Extensive-Vision-AI-Program-6/week_12/assets/step_1.jpeg)



**Step 2:** 

![step_2](/home/adit/Documents/GitHub/Extensive-Vision-AI-Program-6/week_12/assets/step_2.jpeg)



**Code Block:**

```
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

```



### MLP

The result of the self-attention operation is sent to MLP, which consists of two sequential linear layers with a GELU activation function applied to the output. The sequence is shown in the following chart.

**![mlp](/home/adit/Documents/GitHub/Extensive-Vision-AI-Program-6/week_12/assets/mlp.jpg)GeLU**

The **Gaussian Error Linear Unit**, or **GELU**, is an activation function. The GELU activation function is xΦ(x), where Φ(x) the standard Gaussian cumulative distribution function. The GELU nonlinearity weights inputs by their percentile, rather than gates inputs by their sign as in [ReLUs](https://paperswithcode.com/method/relu) (x1x>0). Consequently the GELU can be thought of as a smoother ReLU.

![gelu](/home/adit/Documents/GitHub/Extensive-Vision-AI-Program-6/week_12/assets/gelu.png)



**Code Block:**

```python
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

```



### **Block**

The block class combines both the attention module and the MLP module. Along with that it combines dropout, skip connections and layer norm. The sequence diagram is as follow:

![block](/home/adit/Documents/GitHub/Extensive-Vision-AI-Program-6/week_12/assets/block.jpg)

**Code block:**

```python
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

```



### **Attention**

The transformers' attention system allows them to have extraordinarily long-term memory. A transformer model can attend on all previously created tokens. The attention takes three inputs, the well-known queries, keys, and values, and computes the attention matrix from the queries and values, which it then uses to attend to the values.

![attn](/home/adit/Documents/GitHub/Extensive-Vision-AI-Program-6/week_12/assets/attn.jpg)

```
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
```



### Encoder



Encoder - Decoder Block is a part of Transformers. But in ViT we are using only encoder block.

As seen in the code, the purpose of encoder is to merge multiple attention in a Multi-head attention framework with a Multi-Layer Perceptron (MLP) framework in a sequential manner.



```python
class Encoder(nn.Module):
        def __init__(self, config, vis):
            super(Encoder, self).__init__()
            self.vis = vis
            self.layer = nn.ModuleList()
            self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
            for _ in range(config.transformer["num_layers"]):
                layer = Block(config, vis)
                self.layer.append(copy.deepcopy(layer))

        def forward(self, hidden_states):
            attn_weights = []
            for layer_block in self.layer:
                hidden_states, weights = layer_block(hidden_states)
                if self.vis:
                    attn_weights.append(weights)
            encoded = self.encoder_norm(hidden_states)
            return encoded, attn_weights
```

