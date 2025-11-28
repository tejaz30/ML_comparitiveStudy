class TabTransformer(nn.Module):
    def __init__(self, 
                 num_num_features, 
                 vocab_sizes, 
                 embedding_dim=32, 
                 num_transformer_blocks=2, 
                 num_heads=4):
        super(TabTransformer, self).__init__()
        
        # --- Embeddings ---
        # Create a dictionary of embeddings for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embedding_dim) 
            for vocab_size in vocab_sizes.values()
        ])
        self.num_cat_features = len(vocab_sizes)
        
        # --- Transformer Encoder ---
        # PyTorch has a built-in TransformerEncoderLayer class
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=embedding_dim * 2, 
            dropout=0.1, 
            batch_first=True,
            norm_first=True # Pre-Norm usually stabilizes training
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_blocks)
        
        # --- Numerical MLP ---
        self.num_mlp = nn.Sequential(
            nn.Linear(num_num_features, 16),
            nn.ReLU()
        )
        
        # --- Final Concatenation & Classification ---
        # Input dim = (Num_Cats * Emb_Dim) + Num_MLP_Out
        final_input_dim = (self.num_cat_features * embedding_dim) + 16
        
        self.final_mlp = nn.Sequential(
            nn.Linear(final_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Output 1 logit (BCEWithLogitsLoss handles sigmoid)
        )

    def forward(self, x_cat, x_num):
        # x_cat shape: (Batch, Num_Cats)
        # x_num shape: (Batch, Num_Nums)
        
        # 1. Embed Categoricals
        embeddings = []
        for i, emb_layer in enumerate(self.embeddings):
            # Grab the i-th column, embed it, unsqueeze to (Batch, 1, Dim)
            col_emb = emb_layer(x_cat[:, i]).unsqueeze(1)
            embeddings.append(col_emb)
            
        # Concat -> (Batch, Num_Cats, Dim)
        x = torch.cat(embeddings, dim=1)
        
        # 2. Transformer Pass
        x = self.transformer_encoder(x)
        
        # 3. Flatten Transformer Output -> (Batch, Num_Cats * Dim)
        x = x.flatten(start_dim=1)
        
        # 4. Numerical Pass
        y = self.num_mlp(x_num)
        
        # 5. Concatenate & Final Output
        concat = torch.cat([x, y], dim=1)
        output = self.final_mlp(concat)
        
        return output
