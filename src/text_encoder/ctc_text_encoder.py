import re
from string import ascii_lowercase
import torch
from pyctcdecode import build_ctcdecoder
from tokenizers import Tokenizer
from src.utils.io_utils import ROOT_PATH

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, use_bpe=True, **kwargs):
        """
        Args:
            use_bpe (bool): use bpe or not
        """
        self.use_bpe=use_bpe
        data_lm_lc_path = str(ROOT_PATH / 'pretrained_lm/lowercase_3-gram.pruned.1e-7.arpa')
        librispeech_vocab_path = str(ROOT_PATH / 'pretrained_lm/librispeech-vocab.txt')
        
        if self.use_bpe:
            tok_path = ROOT_PATH / 'pretrained_lm/bpe_tokenizer.json'
            self.tokenizer = Tokenizer.from_file(str(tok_path))
            self.char2ind = {k.lower(): v for k,v in  self.tokenizer.get_vocab().items()}
            self.ind2char = {v: k.lower() for k, v in self.char2ind.items()}
            self.vocab = [self.ind2char[ind] for ind in range(len(self.ind2char))]
        else:
            alphabet = list(ascii_lowercase + " ")
            self.use_bpe = use_bpe
            self.vocab = [self.EMPTY_TOK] + list(alphabet)
            self.ind2char = dict(enumerate(self.vocab))
            self.char2ind = {v: k for k, v in self.ind2char.items()}

        if librispeech_vocab_path is not None:
            with open(librispeech_vocab_path) as f:
                unigrams = [t.lower() for t in f.read().strip().split("\n")]

        self.decoder = build_ctcdecoder(
            labels=self.vocab,
            kenlm_model_path=data_lm_lc_path,
            unigrams=unigrams,
        )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            if self.use_bpe:
                return torch.Tensor(self.tokenizer.encode(text.lower()).ids).unsqueeze(0)
            else:
                return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            if ind == last_ind:
                continue
            elif ind != self.char2ind[self.EMPTY_TOK]:
                decoded.append(self.ind2char[ind])
            last_ind = ind
        return "".join(decoded)

    def ctc_beam_search(self, log_probs, beam_size=50) -> str:
        log_probs = log_probs.cpu().numpy()
        return self.decoder.decode(log_probs, beam_size)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
