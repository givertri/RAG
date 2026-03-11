import csv
import json

csv_file = '../rag_outputs.csv'
json_column_name = 'contexts'  # replace with your actual column name

ret_sum = 0.0
gen_sum = 0.0
count_ret = 0
count_gen = 0

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    i = 0
    for row in reader:
        # ---- JSON parsing ----
        try:
            data = json.loads(row[json_column_name])
            print(i, json.dumps(data, indent=4))
            i += 1
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON in row: {row}")

        # ---- Average calculation ----
        try:
            if row.get("ret_time"):
                ret_sum += float(row["ret_time"])
                count_ret += 1
        except ValueError:
            pass

        try:
            if row.get("gen_time"):
                gen_sum += float(row["gen_time"])
                count_gen += 1
        except ValueError:
            pass

# ---- Final averages ----
avg_ret = ret_sum / count_ret if count_ret else 0
avg_gen = gen_sum / count_gen if count_gen else 0

print(f"\nAverage ret_time: {avg_ret}")
print(f"Average gen_time: {avg_gen}")