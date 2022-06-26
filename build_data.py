
import re
import json
import requests
import pandas as pd
import preprocessing
from preprocessing import *
from sklearn.pipeline import Pipeline

RAW_DATA = '/data/raw_data.csv'
OUTPUT = '/data/dataset.csv'

#url link is used manually here as for now
def crawl_data(url):
    r = re.search(r"i\.(\d+)\.(\d+)", url)
    shop_id, item_id = r[1], r[2]
    ratings_url = "https://shopee.sg/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"

    offset = 0
    d = {"comment": []}
    while True:
        data = requests.get(
            ratings_url.format(shop_id=shop_id, item_id=item_id, offset=offset)
        ).json()

        # uncomment this to print all data:
        # print(json.dumps(data, indent=4))
        
        i = 1
        for i, rating in enumerate(data["data"]["ratings"], 1):
            # d["username"].append(rating["author_username"])
            # d["rating"].append(rating["rating_star"])
            # d["ctime"].append(rating["ctime"])
            d["comment"].append(rating["comment"])

            # print(rating["author_username"])
            # print(rating["rating_star"])
            # print(rating["ctime"])
            # print(rating["comment"])
            # print("-" * 100)
            

        if i % 20:
            break

        offset += 20
    

    raw_data = pd.DataFrame(d)
    raw_data.to_csv(RAW_DATA, index=False)


def build_dataset():
    dataset = pd.read_csv(RAW_DATA)
    dataset.dropna(subset = ["comment"], inplace=True)
    dataset.drop_duplicates(subset=['comment'])
    dataset.to_csv(OUTPUT,index=False)


def main():

    # url = "https://shopee.vn/-M%C3%A3-LIFEMALL995-gi%E1%BA%A3m-10-%C4%91%C6%A1n-99K-S%C3%A1ch-thi-v%C3%A0o-10-Combo-3-cu%E1%BB%91n-C%E1%BA%A5p-t%E1%BB%91c-789-m%C3%B4n-To%C3%A1n-V%C4%83n-Anh-i.313752154.15106042337?sp_atk=3b519927-0833-45eb-9620-e948e9fb2e2b&xptdk=3b519927-0833-45eb-9620-e948e9fb2e2b"

    # crawl_data(url=url)
    # build_dataset()
  
    pass



if __name__ == "__main__":
    main()


