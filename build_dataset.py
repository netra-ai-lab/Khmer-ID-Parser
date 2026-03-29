import json
import random
from datetime import datetime, timedelta

def generate_nid(): return f"{random.randint(100000000, 999999999)}"

def random_date(start_year, end_year):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))

def mrz_check_digit(data):
    weights =[7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if char == '<': val = 0
        elif char.isdigit(): val = int(char)
        else: val = ord(char) - 55
        total += val * weights[i % 3]
    return str(total % 10)

def load_data():
    with open("names.json", "r", encoding="utf-8") as f: names = json.load(f)
    with open("marks.json", "r", encoding="utf-8") as f: marks = json.load(f)
    with open("data.json", "r", encoding="utf-8") as f: loc_raw = json.load(f)

    locations =[]
    for prov in loc_raw:
        p_name = prov["name"]["km"]
        if "districts" in prov and "values" in prov["districts"]:
            for dist in prov["districts"]["values"]:
                d_name = dist["name"]["km"]
                if "communes" in dist and "values" in dist["communes"]:
                    for comm in dist["communes"]["values"]:
                        c_name = comm["name"]["km"]
                        if "villages" in comm and "values" in comm["villages"]:
                            for vill in comm["villages"]["values"]:
                                locations.append((vill["name"]["km"], c_name, d_name, p_name))
    return names, marks, locations

def build_clean_datasets(num_samples=50000, split_ratio=0.8):
    names, marks, locations = load_data()
    dataset =[]

    print(f"Generating {num_samples} clean structured samples...")
    for _ in range(num_samples):
        name = random.choice(names)
        mark = random.choice(marks)
        pob = random.choice(locations)
        addr = random.choice(locations)
        
        dob = random_date(1950, 2005)
        issue_date = random_date(2012, 2024)
        try: exp_date = issue_date.replace(year=issue_date.year + 10)
        except ValueError: exp_date = issue_date.replace(year=issue_date.year + 10, month=2, day=28)

        nid = generate_nid()
        gender_mrz = "M" if name["gender"] == "ប្រុស" else "F"
        
        # MRZ Calculation
        dob_mrz, exp_mrz = dob.strftime("%y%m%d"), exp_date.strftime("%y%m%d")
        line1 = f"IDKHM{nid}{mrz_check_digit(nid)}".ljust(30, '<')
        line2 = f"{dob_mrz}{mrz_check_digit(dob_mrz)}{gender_mrz}{exp_mrz}{mrz_check_digit(exp_mrz)}KHM".ljust(29, '<') + "4"
        line3 = f"{name['last_en']}<<{name['first_en']}".ljust(30, '<')

        sample = {
            "nid": nid,
            "name_kh": f"{name['last_kh']} {name['first_kh']}",
            "name_en": f"{name['last_en']} {name['first_en']}",
            "dob": dob.strftime("%d.%m.%Y"),
            "gender": name["gender"],
            "height": str(random.randint(145, 185)),
            "pob": pob,
            "addr": addr,
            "issue_date": issue_date.strftime("%d.%m.%Y"),
            "exp_date": exp_date.strftime("%d.%m.%Y"),
            "mark": mark,
            "mrz1": line1, "mrz2": line2, "mrz3": line3
        }
        dataset.append(sample)

    random.shuffle(dataset)
    split_idx = int(num_samples * split_ratio)
    
    with open("train_clean.jsonl", "w", encoding="utf-8") as f:
        for item in dataset[:split_idx]: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open("val_clean.jsonl", "w", encoding="utf-8") as f:
        for item in dataset[split_idx:]: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("✅ Clean dataset generated!")

if __name__ == "__main__":
    # Because augmentation is on the fly, we don't need 100k samples. 50k is more than enough!
    build_clean_datasets(100_000, split_ratio=0.8)