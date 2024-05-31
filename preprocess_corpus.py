import os
import re
import html
import time
import random
import json
import requests

from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from googletrans import Translator

translator = Translator()

def translate_text(texts, dest="vi", batch_size=16, delay_per_batch=5):
    translator = Translator()
    is_str = False
    if isinstance(texts, str):
        texts = [texts]
        is_str = True

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tries = 5
        while tries > 0:
            try:
                results.extend(translator.translate(batch, dest=dest))
                time.sleep(random.randint(0, 3))
                break
            except Exception as e:
                tries -= 1
                print(e)
                print("Retrying...")
                time.sleep(random.randint(0, 3))

    results = [result.text for result in results]
    if is_str:
        return results[0]
    else:
        return results
def get_text_from_html_file(html_path):
    with open(html_path, "r") as f:
        html_text = f.read()
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text()


def parse_list_elements(list_element, item_separator="; "):
    assert list_element.name in ["ul", "ol"]

    content_list = []

    # Find all <li> elements within the <ul> or <ol> element
    li_elements = list_element.find_all("li")

    for li_element in li_elements:
        content_list.append(li_element.get_text())

    content_list = [c.strip() for c in content_list]
    content_list = [c for c in content_list if c != ""]
    return item_separator.join(content_list)


def filter_content_text(text):
    leading_terms = [
        "đặt lịch khám",
        "bệnh viện đa khoa",
        "cập nhật lần cuối",
        "Hà Nội",
        "TP.HCM",
        "liên hệ với chúng tôi",
        "TP. HÀ NỘI",
        "ĐỐI TÁC BẢO HIỂM",
        "HỆ THỐNG BỆNH VIỆN",
        "BÀI VIẾT LIÊN QUAN",
        "Để được tư vấn",
    ]
    for leading_term in leading_terms:
        if text.lower().startswith(leading_term.lower()):
            return False
    return True


def get_abstract(raw_html):
    h1_end_pos = raw_html.find("</h1>")
    h3_start_pos = raw_html.find("<h3>", h1_end_pos)
    abstract_data = raw_html[h1_end_pos + 5 : h3_start_pos]
    abstract_soup = BeautifulSoup(abstract_data, "html.parser")
    return abstract_soup.get_text().strip()


def get_all_subsections(soup, debug=False):
    subsections = []
    h2_tags = soup.find_all(["h2"])

    for tag in h2_tags:
        subsection_name = tag.get_text().strip()
        if not filter_content_text(subsection_name):
            continue
        content = []

        # travel through all the siblings of the h2 tag
        next_node = tag.next_sibling
        while next_node:
            if next_node.name == "h2":
                break
            else:
                if next_node.name == "p":
                    content.append(next_node.get_text().strip())
                elif next_node.name in ["ul", "ol"]:
                    list_content = parse_list_elements(next_node)
                    content.append(list_content)
                else:
                    content.append(next_node.get_text().strip())
                next_node = next_node.next_sibling

        content = [c for c in content if c != "" and filter_content_text(c)]
        subsection_content = "\n".join(content).strip()

        if debug:
            print(subsection_name.strip())
            print("\n".join(content))
            print()

        subsections.append(
            {
                "subsection_name": subsection_name,
                "subsection_content": subsection_content,
                "subsection_string": subsection_name + "\n" + subsection_content,
            }
        )

    return subsections


def preprocess_html_source_document(html_source):
    soup = BeautifulSoup(html_source, "html.parser")
    document = soup.get_text()
    lines = document.split("\n")

    # first line is the link
    link = lines[0]
    
    # if len(lines) < 5:
    #     # recrawl the html source from the link
    #     recrawl_html_source = requests.get(link).text
    #     soup = BeautifulSoup(recrawl_html_source, "html.parser")
    #     document = soup.get_text()
    #     lines = document.split("\n")

    # second line is blank
    assert lines[1] == "", f"lines[1]: {lines[1]}"

    # third line is "Trang chủ >"
    assert lines[2].strip().lower() == "Trang chủ >".lower(), f"lines[2]: {lines[2].strip()}"

    # fourth line is "CHUYÊN MỤC BỆNH HỌC >"
    assert lines[3].strip().lower() == "CHUYÊN MỤC BỆNH HỌC >".lower(), f"lines[3]: {lines[3].strip()}"

    # fifth line is category, or the title
    h1_idx = -1
    for i in range(5, 10):
        if "<h1>" in lines[i]:
            h1_idx = i
            break
    if "<h1>" in html_source.split("\n")[6]:
        category = lines[4].strip()[:-1].strip()
        # sixth line and seventh line are the same title
        title = lines[5].strip()
        assert lines[6].strip() == title, f"line[6]: {lines[6].strip()} || title: {title}"
    else:
        category = None
        title = lines[4].strip()
        assert lines[5].strip() == title, f"line[5]: {lines[5].strip()} || title: {title}"

    abstract = get_abstract(html_source)
    recrawl_html_source = requests.get(link).text
    subsections = get_all_subsections(BeautifulSoup(recrawl_html_source, "html.parser"))

    content = [
        title,
        abstract,
        *[subsection["subsection_string"] for subsection in subsections],
    ]
    content = "\n\n".join(content)

    return {
        "title": title,
        "category": category,
        "link": link,
        "abstract": abstract,
        "content": content,
        "subsections": subsections,
    }
def main():
    corpus_dir ="corpus"
    processed_dir="processed"
    os.makedirs(processed_dir, exist_ok=True)
    document_paths = os.listdir(corpus_dir)
    document_names = sorted(os.listdir(corpus_dir))
    error_indices = []
    for i in tqdm(range(0, len(document_names)), desc="Processing documents"):
        try:
            document_name = document_names[i]
            document_path = os.path.join(corpus_dir, document_name)
            html_src = open(document_path, "r").read()
            processed_document = preprocess_html_source_document(html_src)
            processed_document["name"] = document_name

            processed_save_path = os.path.join(processed_dir, document_name + ".json")
            with open(processed_save_path, "w") as f:
                json.dump(processed_document, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(e)
            print(f"Error at index {i} with document name {document_name}")
            error_indices.append(i)
            continue
if __name__ == "__main__":
    main()