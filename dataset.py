import json
import random
import string
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

MAX_LEN = 1024

# --- Vocabulary Classes ---
class CharVocab:
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<UNK>": 1}
    def add_char(self, char):
        if char not in self.char2idx: self.char2idx[char] = len(self.char2idx)
    def __len__(self): return len(self.char2idx)

class TagVocab:
    def __init__(self):
        self.tag2idx = {"<PAD>": 0, "O": 1}
    def add_tag(self, tag):
        if tag not in self.tag2idx: self.tag2idx[tag] = len(self.tag2idx)
    def __len__(self): return len(self.tag2idx)

def to_khmer_num(text):
    khmer_digits = "០១២៣៤៥៦៧៨៩"
    return ''.join(khmer_digits[int(c)] if c.isdigit() else c for c in str(text))

# --- On-The-Fly Augmentation Builder ---
class AugmentingTextBuilder:
    def __init__(self, is_train=True):
        self.text = ""
        self.tags = []
        self.is_train = is_train

    def add_noise_word(self):
        if self.is_train and random.random() < 0.05:
            noise = "".join(random.choices(string.ascii_uppercase + "ិីុូើា ", k=random.randint(3, 8)))
            self.text += noise + " "
            self.tags.extend(["O"] * (len(noise) + 1))

    def append(self, content, label=None, allow_drop=True):
        if not content: return
        if self.is_train and allow_drop and random.random() < 0.02:
            return
        self.add_noise_word()
        b_emitted = False
        for char in content:
            if self.is_train and random.random() < 0.01:
                continue
            if char == " " and self.is_train and random.random() < 0.10:
                continue
            self.text += char
            if label:
                if not b_emitted:
                    self.tags.append(f"B-{label}")
                    b_emitted = True
                else:
                    self.tags.append(f"I-{label}")
            else:
                self.tags.append("O")

def render_sample(item, is_train):
    b = AugmentingTextBuilder(is_train)

    # Helper: strip any matching administrative prefix.
    # Handles both rural (ឃុំ/ស្រុក/ខេត្ត) and urban (សង្កាត់/ខណ្ឌ/រាជធានី) forms.
    def strip_any(word, *prefixes):
        for pfx in prefixes:
            if is_train and random.random() < 0.3 and word.startswith(pfx):
                return word[len(pfx):]
        return word

    # 1. NID
    b.append(item["nid"], "ID_NUM")
    b.append("\n", None, allow_drop=False)

    # 2. Names
    b.append(random.choice(["គោន្តនាមនិងនាមៈ ", "គោត្តនាមនិងនាមៈ "]), None)
    b.append(item["name_kh"], "NAME_KH")
    b.append("\n", None, allow_drop=False)
    b.append(item["name_en"], "NAME_EN")
    b.append("\n", None, allow_drop=False)

    # 3. DOB, Gender, Height
    b.append("ថ្ងៃខែឆ្នាំកំណើតៈ ", None)
    date_sep = random.choice([".", ":"]) if is_train else "."
    dob = to_khmer_num(item["dob"].replace(".", date_sep))
    b.append(dob, "DOB")
    b.append(" នេះ ", None)
    b.append(item["gender"], "GENDER")
    b.append(" កំពស់ៈ ", None)
    b.append(to_khmer_num(item["height"]), "HEIGHT")
    b.append(random.choice([" ស.ប", " ស.ម", " ផ.ម"]) if is_train else " ស.ម", None)
    b.append("\n", None, allow_drop=False)

    # 4. Place of birth — pob[1]=commune  pob[2]=district  pob[3]=province
    b.append("ទីកន្លែងកំណើតៈ ", None)
    b.append(strip_any(item["pob"][1], "ឃុំ", "សង្កាត់"),  "POB_COMM")
    b.append(" ", None)
    b.append(strip_any(item["pob"][2], "ស្រុក", "ខណ្ឌ"),   "POB_DIST")
    b.append(" ", None)
    b.append(strip_any(item["pob"][3], "ខេត្ត", "រាជធានី"), "POB_PROV")
    b.append("\n", None, allow_drop=False)

    # 5. Current address — addr[0]=village  addr[1]=commune  addr[2]=district  addr[3]=province
    b.append("អាសយដ្ឋានៈ ", None)
    b.append(strip_any(item["addr"][0], "ភូមិ"),             "ADDR_VILL")
    b.append("\n", None, allow_drop=False)
    b.append(strip_any(item["addr"][1], "ឃុំ", "សង្កាត់"),  "ADDR_COMM")
    b.append(" ", None)
    b.append(strip_any(item["addr"][2], "ស្រុក", "ខណ្ឌ"),   "ADDR_DIST")
    b.append(" ", None)
    b.append(strip_any(item["addr"][3], "ខេត្ត", "រាជធានី"), "ADDR_PROV")
    b.append("\n", None, allow_drop=False)

    # 6. Validity
    b.append("សុពលភាពៈ ", None)
    b.append(to_khmer_num(item["issue_date"]), "ISSUE_DATE")
    b.append(" ដល់ថ្ងៃ ", None)
    b.append(to_khmer_num(item["exp_date"]), "EXP_DATE")
    b.append("\n", None, allow_drop=False)

    # 7. Marks
    b.append("ភិនភាគ\n", None)
    b.append(item["mark"], "MARKS")
    b.append("\n", None, allow_drop=False)

    # 8. MRZ
    b.append(item["mrz1"], "MRZ")
    b.append("\n", None, allow_drop=False)
    b.append(item["mrz2"], "MRZ")
    b.append("\n", None, allow_drop=False)
    b.append(item["mrz3"], "MRZ")

    return b.text, b.tags

# --- Dataset and DataLoader logic ---
def build_vocab(train_file):
    char_vocab, tag_vocab = CharVocab(), TagVocab()
    for c in string.printable + "០១២៣៤៥៦៧៨៩កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័៘៙៚៕។":
        char_vocab.add_char(c)
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            for _ in range(3):
                text, _ = render_sample(item, is_train=True)
                for c in text: char_vocab.add_char(c)
            text, tags = render_sample(item, is_train=False)
            for c in text: char_vocab.add_char(c)
            for t in tags:
                if t != "O": tag_vocab.add_tag(t)
    return char_vocab, tag_vocab

class KhmerIDDataset(Dataset):
    def __init__(self, json_file, char_vocab, tag_vocab, is_train=True):
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.is_train = is_train
        self.data = []
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text, tags = render_sample(item, self.is_train)
        char_ids = [self.char_vocab.char2idx.get(c, self.char_vocab.char2idx["<UNK>"]) for c in text]
        tag_ids  = [self.tag_vocab.tag2idx[t] for t in tags]
        return torch.tensor(char_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)

def collate_fn(batch):
    chars = [item[0][:MAX_LEN] for item in batch]
    tags  = [item[1][:MAX_LEN] for item in batch]
    chars_padded = pad_sequence(chars, batch_first=True, padding_value=0)
    tags_padded  = pad_sequence(tags,  batch_first=True, padding_value=0)
    mask = (chars_padded != 0)
    return chars_padded, tags_padded, mask