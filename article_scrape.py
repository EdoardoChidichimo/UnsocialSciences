import requests
import time
import os
import csv
csv.field_size_limit(10**7)
import json
import argparse

from config import SELECTED_JOURNALS, MAX_TEAM_SIZE, FIELD_NAME, EMAIL

CROSSREF_API = "https://api.crossref.org/works"
EXCLUDED_TYPES = {"editorial", "correction", "letter", "retraction", "news", "other"}

author_index_map = {}
next_author_id = 0
author_log_file = None


def load_author_index():
    global next_author_id
    if os.path.exists(author_log_file):
        with open(author_log_file, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    idx, name = row
                    if idx.strip().isdigit():
                        author_index_map[name] = int(idx)
        if author_index_map:
            next_author_id = max(author_index_map.values()) + 1



def save_author(name):
    global next_author_id
    name = name.lower()
    if name not in author_index_map:
        author_index_map[name] = next_author_id
        with open(author_log_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([next_author_id, name])
        next_author_id += 1
    return author_index_map[name]


def is_probable_book_review(item, strict=False):
    type_ = item.get("type", "").lower()
    if type_ == "book-review":
        return True
    if strict and type_ == "review":
        return True

    title_fields = []
    for key in ['title', 'subtitle', 'container-title']:
        val = item.get(key, [])
        if isinstance(val, list):
            title_fields.extend(val)
        elif isinstance(val, str):
            title_fields.append(val)

    combined_text = " ".join(title_fields).lower()
    keywords = ["book review", "review of", "review essay"]
    if strict:
        keywords += ["review"]

    return any(kw in combined_text for kw in keywords)


def get_total_articles(journal_name):
    params = {
        "query.container-title": journal_name,
        "filter": "type:journal-article,from-pub-date:1925-01-01,until-pub-date:2024-12-31",
        "rows": 0
    }
    headers = {
        "User-Agent": f"CrossRefHarvester/1.0 (mailto:{EMAIL})"
    }

    for attempt in range(3):
        try:
            r = requests.get(CROSSREF_API, params=params, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json().get('message', {}).get('total-results', 0)
            elif r.status_code == 429:
                wait = min(2 ** attempt, 60)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"Error {r.status_code}")
        except Exception as e:
            print(f"Error fetching count: {e}")
            time.sleep(2)
    return 0


def init_stats_file(output_dir):
    stats_path = os.path.join(output_dir, f"{FIELD_NAME}/{FIELD_NAME}_journal_stats.csv")
    if not os.path.exists(stats_path):
        with open(stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["journal", "total_articles_found", "rejected", "final_saved"])


def log_journal_stats(output_dir, journal_name, total, rejected, saved):
    stats_path = os.path.join(output_dir, f"{FIELD_NAME}/{FIELD_NAME}_journal_stats.csv")
    with open(stats_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([journal_name, total, rejected, saved])


def get_articles_for_journal(journal_name, output_dir, strict_book_filter=False):
    total_results = get_total_articles(journal_name)
    print(f"{journal_name}: {total_results} results")

    cursor = "*"
    rows_per_request = 1000
    downloaded = 0
    rejected = 0

    safe_name = journal_name.replace("/", "_").replace(" ", "_")
    out_file = os.path.join(output_dir, f"{safe_name}.csv")
    state_file = os.path.join(output_dir, f"{safe_name}_state.json")

    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            downloaded = state.get("downloaded", 0)
            cursor = state.get("cursor", "*")
            print(f"Resuming {journal_name} from cursor {cursor[:20]}")

    write_header = not os.path.exists(out_file)
    headers = {"User-Agent": f"CrossRefHarvester/1.0 (mailto:{EMAIL})"}

    with open(out_file, "a", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["year", "authors"])

        stuck_counter = 0
        max_stuck_loops = 50
        prev_downloaded = downloaded

        while True:
            params = {
                "query.container-title": journal_name,
                "filter": "type:journal-article,from-pub-date:1925-01-01,until-pub-date:2024-12-31",
                "rows": rows_per_request,
                "cursor": cursor
            }
            try:
                r = requests.get(CROSSREF_API, params=params, headers=headers, timeout=90)
                if r.status_code != 200:
                    print(f"Failed request: {r.status_code}")
                    break
                data = r.json().get("message", {})
                cursor = data.get("next-cursor", None)
                items = data.get("items", [])

                if not items:
                    break

                for item in items:
                    if item.get("type", "").lower() in EXCLUDED_TYPES:
                        rejected += 1
                        continue
                    if is_probable_book_review(item, strict=strict_book_filter):
                        rejected += 1
                        continue

                    authors_raw = item.get("author", [])
                    if not authors_raw or len(authors_raw) >= MAX_TEAM_SIZE:
                        rejected += 1
                        continue

                    year = None
                    for field in ["published-print", "published-online", "created"]:
                        if field in item and "date-parts" in item[field]:
                            year = item[field]["date-parts"][0][0]
                            break

                    if year is None or year < 1925 or year > 2024:
                        rejected += 1
                        continue

                    indices = []
                    for a in authors_raw:
                        if "family" in a:
                            name = f"{a.get('given', '')} {a.get('family', '')}".strip().lower()
                            if name:
                                idx = save_author(name)
                                indices.append(str(idx))
                    if indices:
                        writer.writerow([year, ";".join(indices)])
                        downloaded += 1

                with open(state_file, 'w') as f_state:
                    json.dump({
                        "downloaded": downloaded,
                        "cursor": cursor
                    }, f_state)

                print(f"{journal_name}: Downloaded {downloaded} items")
                time.sleep(1.2)

                if downloaded == prev_downloaded:
                    stuck_counter += 1
                    print(f"No new data added. Stuck loop #{stuck_counter}")
                else:
                    stuck_counter = 0
                    prev_downloaded = downloaded

                if stuck_counter >= max_stuck_loops:
                    print("Detected stalled downloading. Exiting.")
                    break

                if not cursor:
                    break

            except requests.exceptions.ReadTimeout:
                print(f"Timeout occurred. Retrying in 10 seconds...")
                time.sleep(10)
                continue  # retry the loop

            except requests.exceptions.ConnectionError as ce:
                print(f"Connection error: {ce}. Retrying in 15 seconds...")
                time.sleep(15)
                continue

            except Exception as e:
                print(f"Unexpected error: {e}")
                break

    log_journal_stats(output_dir, journal_name, total_results, rejected, downloaded)
    print(f"Completed {journal_name}. Total downloaded: {downloaded}")


def process_journals(journal_list, output_dir, strict_book_filter=False):
    os.makedirs(output_dir, exist_ok=True)
    init_stats_file(output_dir)
    load_author_index()

    for journal in journal_list:
        get_articles_for_journal(journal.replace("_", " "), output_dir, strict_book_filter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-book-filter", action="store_true", help="Enable strict filtering to eliminate all potential book reviews")
    args = parser.parse_args()

    output_dir = os.path.join("data", FIELD_NAME)
    author_log_file = os.path.join(output_dir, f"{FIELD_NAME}_author_index.csv")

    print(f"Processing {len(SELECTED_JOURNALS)} journals for field '{FIELD_NAME}'...\n")
    process_journals(SELECTED_JOURNALS, output_dir, args.strict_book_filter)