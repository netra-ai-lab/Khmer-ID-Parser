# model.py
import torch
import torch.nn as nn
from torchcrf import CRF


class CharCNN(nn.Module):
    def __init__(self, embed_dim, num_filters, kernel_sizes):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k,
                      padding='same')
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_outputs = [self.relu(conv(x)) for conv in self.convs]
        out = torch.cat(conv_outputs, dim=1)
        return out.permute(0, 2, 1)


class KhmerIDParserModel(nn.Module):
    def __init__(self, vocab_size, num_tags, embed_dim=64, cnn_filters=64,
                 cnn_kernel_sizes=[3, 5, 7] , lstm_hidden_dim=256, lstm_layers=2,
                 dropout=0.3, pad_idx=0):
        super().__init__()

        self.pad_idx = pad_idx

        # TagVocab layout:  <PAD>=0  O=1  B-X=2  I-X=3 ...
        # The CRF must NOT include <PAD> as a valid tag — it has no concept of
        # padding and treats every index as a learnable state.  At init its
        # transition scores are uniform, so it freely predicts tag-0 (<PAD>)
        # at real positions (~74 % of the time in tests).
        #
        # Fix: give the CRF (num_tags - 1) states, i.e. every tag except <PAD>.
        # We subtract 1 from tag ids before passing to the CRF, and add 1 back
        # after decoding — so the caller always works with original vocab ids.
        self.num_crf_tags = num_tags - 1   # O=0  B-X=1  I-X=2  ... inside CRF

        # 1. Embedding
        self.embedding   = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)

        # 2. CharCNN + LayerNorm
        self.char_cnn   = CharCNN(embed_dim, cnn_filters, cnn_kernel_sizes)
        cnn_out_dim     = cnn_filters * len(cnn_kernel_sizes)
        self.layer_norm1  = nn.LayerNorm(cnn_out_dim)
        self.cnn_dropout  = nn.Dropout(dropout)

        # 3. BiLSTM
        self.lstm = nn.LSTM(
            input_size    = cnn_out_dim,
            hidden_size   = lstm_hidden_dim,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0,
        )

        # 4. LayerNorm + projection
        self.layer_norm2  = nn.LayerNorm(lstm_hidden_dim * 2)
        self.lstm_dropout = nn.Dropout(dropout)
        self.hidden2tag   = nn.Linear(lstm_hidden_dim * 2, self.num_crf_tags)

        # 5. CRF — over (num_tags - 1) real tags only
        self.crf = CRF(self.num_crf_tags, batch_first=True)

    # ── Internal feature extraction ───────────────────────────────────────────
    def _get_features(self, x):
        embeds  = self.emb_dropout(self.embedding(x))

        cnn_out = self.char_cnn(embeds)
        cnn_out = self.cnn_dropout(self.layer_norm1(cnn_out))

        lstm_out, _ = self.lstm(cnn_out)
        lstm_out    = self.lstm_dropout(self.layer_norm2(lstm_out))

        return self.hidden2tag(lstm_out)   # (batch, seq, num_crf_tags)

    # ── Training forward — returns scalar NLL loss ────────────────────────────
    def forward(self, x, tags, mask=None):
        """
        x    : (batch, seq)  char ids
        tags : (batch, seq)  tag ids from TagVocab  (PAD=0, O=1, B-X=2, ...)
        mask : (batch, seq)  bool, True = real token
        """
        logits = self._get_features(x)

        if mask is None:
            mask = (x != self.pad_idx)

        # Shift tags into CRF space: subtract 1.
        # Masked positions keep their (meaningless) value — CRF ignores them.
        tags_crf = (tags - 1).clamp(min=0)

        loss = -self.crf(emissions=logits, tags=tags_crf, mask=mask, reduction='token_mean')
        return loss

    # ── Inference — returns tag ids in original TagVocab space ───────────────
    def decode(self, x, mask=None):
        """
        Returns List[List[int]] of tag ids in TagVocab space
        (O=1, B-X=2, ...) — never contains 0 (<PAD>).
        """
        logits = self._get_features(x)

        if mask is None:
            mask = (x != self.pad_idx)

        pred_crf = self.crf.decode(emissions=logits, mask=mask)

        # Shift back: add 1 to restore TagVocab ids
        return [[t + 1 for t in seq] for seq in pred_crf]